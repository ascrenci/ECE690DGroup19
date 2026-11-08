"""Structured surgery on a HF T5: slice FFN channels, attention heads and vocabulary rows in place.

Heads are pruned with the same set in every layer of a stack because T5 shares layer 0's
relative-position bias across the whole stack. Assumes an untied lm_head (flan-t5).
A pruned model can't go through save_pretrained (HF rebuilds uniform dims), so it is saved as
state_dict + spec.json and reloaded with `load` into a fresh base-architecture model.
"""
import json

import torch
from torch import nn

STACKS = ("encoder", "decoder")


def blocks(model, stack):
    return getattr(model, stack).block


def ffn_of(block):
    return block.layer[-1].DenseReluDense


def attentions(block):
    """Self-attention, plus cross-attention in decoder blocks."""
    layers = block.layer[:-1]
    return [layers[0].SelfAttention] + ([layers[1].EncDecAttention] if len(layers) > 1 else [])


def _linear(old, rows=None, cols=None):
    w = old.weight.data
    if rows is not None:
        w = w[rows]
    if cols is not None:
        w = w[:, cols]
    new = nn.Linear(w.shape[1], w.shape[0], bias=False).to(w.device, w.dtype)
    new.weight.data.copy_(w)
    return new


def drop_layers(model, keep):
    """keep[stack] = indices of the blocks to keep (stacks not listed are untouched). Block 0 must stay: it
    owns the relative-position bias that every later block reuses."""
    for stack, idx in keep.items():
        assert idx[0] == 0, "block 0 holds the relative position bias"
        part = getattr(model, stack)
        part.block = nn.ModuleList([part.block[i] for i in idx])
        part.config.num_layers = len(idx)
        setattr(model.config, "num_layers" if stack == "encoder" else "num_decoder_layers", len(idx))


def prune_ffn(model, keep):
    """keep[stack] = one index tensor per layer: the d_ff channels to keep."""
    for stack in STACKS:
        for block, idx in zip(blocks(model, stack), keep[stack]):
            ffn = ffn_of(block)
            ffn.wi_0, ffn.wi_1 = _linear(ffn.wi_0, rows=idx), _linear(ffn.wi_1, rows=idx)
            ffn.wo = _linear(ffn.wo, cols=idx)


def prune_heads(model, keep):
    """keep[stack] = head indices kept in every layer of that stack."""
    for stack in STACKS:
        heads = torch.as_tensor(keep[stack]).cpu()  # CPU like torch.arange below; moved to the weights' device after
        for block in blocks(model, stack):
            for attn in attentions(block):
                dim = attn.key_value_proj_dim
                idx = (heads[:, None] * dim + torch.arange(dim)).flatten().to(attn.q.weight.device)
                attn.q, attn.k, attn.v = _linear(attn.q, rows=idx), _linear(attn.k, rows=idx), _linear(attn.v, rows=idx)
                attn.o = _linear(attn.o, cols=idx)
                attn.n_heads, attn.inner_dim = len(heads), len(heads) * dim
                if attn.has_relative_attention_bias:
                    old = attn.relative_attention_bias
                    new = nn.Embedding(old.num_embeddings, len(heads)).to(old.weight.device, old.weight.dtype)
                    new.weight.data.copy_(old.weight.data[:, heads.to(old.weight.device)])
                    attn.relative_attention_bias = new


def prune_vocab(model, kept):
    """Keep only these embedding and lm_head rows. `kept` is ascending and starts 0, 1, 2 so special ids don't move."""
    kept = torch.as_tensor(kept)
    emb = model.get_input_embeddings()
    rows = kept.to(emb.weight.device)
    new = nn.Embedding(len(kept), emb.embedding_dim).to(emb.weight.device, emb.weight.dtype)
    new.weight.data.copy_(emb.weight.data[rows])
    model.set_input_embeddings(new)
    model.lm_head = _linear(model.lm_head, rows=rows)
    model.config.vocab_size = len(kept)


def prunable_params(model):
    """Weights in the attention and FFN matrices (what channel pruning removes)."""
    total = 0
    for stack in STACKS:
        for block in blocks(model, stack):
            ffn = ffn_of(block)
            total += sum(m.weight.numel() for m in (ffn.wi_0, ffn.wi_1, ffn.wo))
            for attn in attentions(block):
                total += sum(m.weight.numel() for m in (attn.q, attn.k, attn.v, attn.o))
    return total


def spec_of(model):
    return {
        "ffn": {s: [ffn_of(b).wi_0.out_features for b in blocks(model, s)] for s in STACKS},
        "heads": {s: attentions(blocks(model, s)[0])[0].n_heads for s in STACKS},
        "vocab": model.config.vocab_size,
    }


def save(model, path, kept_ids=None):
    path.mkdir(parents=True, exist_ok=True)
    spec = {**spec_of(model), "kept_ids": None if kept_ids is None else [int(i) for i in kept_ids]}
    (path / "spec.json").write_text(json.dumps(spec))
    torch.save(model.state_dict(), path / "model.pt")


def load(model, path):
    """Reshape a fresh base-architecture `model` to the saved spec, load the weights; returns kept_ids or None."""
    spec = json.loads((path / "spec.json").read_text())
    drop_layers(model, {s: list(range(len(spec["ffn"][s]))) for s in STACKS})
    prune_ffn(model, {s: [torch.arange(w) for w in spec["ffn"][s]] for s in STACKS})
    prune_heads(model, {s: torch.arange(spec["heads"][s]) for s in STACKS})
    prune_vocab(model, torch.arange(spec["vocab"]))
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    return spec["kept_ids"]
