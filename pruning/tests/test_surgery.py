import torch
from conftest import logits

import surgery


def keep_all(model):
    cfg = model.config
    return (
        {s: [torch.arange(cfg.d_ff)] * cfg.num_layers for s in surgery.STACKS},
        {s: torch.arange(cfg.num_heads) for s in surgery.STACKS},
    )


def test_full_keep_is_identity(model):
    before = logits(model)
    ffn, heads = keep_all(model)
    surgery.prune_ffn(model, ffn)
    surgery.prune_heads(model, heads)
    surgery.prune_vocab(model, torch.arange(model.config.vocab_size))
    assert torch.allclose(before, logits(model), atol=1e-6)


def test_prune_ffn_equals_zeroing_dropped_channels(model, make_model):
    reference = make_model()
    idx = torch.tensor([1, 4, 5, 20])
    drop = torch.tensor([i for i in range(32) if i not in idx.tolist()])
    for stack in surgery.STACKS:
        for block in surgery.blocks(reference, stack):
            surgery.ffn_of(block).wo.weight.data[:, drop] = 0
    surgery.prune_ffn(model, {s: [idx, idx] for s in surgery.STACKS})
    assert surgery.ffn_of(surgery.blocks(model, "encoder")[0]).wi_0.out_features == 4
    assert torch.allclose(logits(reference), logits(model), atol=1e-5)


def test_prune_heads_equals_zeroing_dropped_heads(model, make_model):
    reference = make_model()
    heads = torch.tensor([0, 2])
    cols = torch.tensor([c for c in range(16) if c // 4 not in heads.tolist()])
    for stack in surgery.STACKS:
        for block in surgery.blocks(reference, stack):
            for attn in surgery.attentions(block):
                attn.o.weight.data[:, cols] = 0
    surgery.prune_heads(model, {s: heads for s in surgery.STACKS})
    first = surgery.attentions(surgery.blocks(model, "decoder")[0])[0]
    assert (first.n_heads, first.inner_dim, first.q.out_features) == (2, 8, 8)
    assert first.relative_attention_bias.weight.shape == (8, 2)
    assert torch.allclose(logits(reference), logits(model), atol=1e-5)


def test_prune_vocab_matches_original_logits_on_kept_ids(model, make_model):
    reference = make_model()
    kept = [0, 1, 2, 5, 9, 12]
    surgery.prune_vocab(model, kept)
    assert model.get_input_embeddings().num_embeddings == 6 and model.lm_head.out_features == 6
    assert model.config.vocab_size == 6
    new = logits(model, enc_ids=(3, 4, 1), dec_ids=(0, 3, 5))  # kept[3]=5, kept[4]=9, kept[5]=12
    old = logits(reference, enc_ids=(5, 9, 1), dec_ids=(0, 5, 12))
    assert torch.allclose(new, old[..., kept], atol=1e-5)


def test_prunable_params_counts_attention_and_ffn(model):
    cfg = model.config
    ffn = 3 * cfg.d_model * cfg.d_ff
    attn = 4 * cfg.d_model * cfg.num_heads * cfg.d_kv
    assert surgery.prunable_params(model) == 2 * (ffn + attn) + 2 * (ffn + 2 * attn)


def test_spec_save_and_load_round_trip(model, make_model, tmp_path):
    surgery.prune_ffn(model, {"encoder": [torch.arange(8), torch.arange(16)], "decoder": [torch.arange(4)] * 2})
    surgery.prune_heads(model, {"encoder": torch.tensor([1, 3]), "decoder": torch.tensor([0])})
    surgery.prune_vocab(model, [0, 1, 2, 7, 8])
    spec = surgery.spec_of(model)
    assert spec == {"ffn": {"encoder": [8, 16], "decoder": [4, 4]}, "heads": {"encoder": 2, "decoder": 1}, "vocab": 5}

    surgery.save(model, tmp_path / "m", kept_ids=[0, 1, 2, 7, 8])
    fresh = make_model(seed=7)
    assert surgery.load(fresh, tmp_path / "m") == [0, 1, 2, 7, 8]
    assert surgery.spec_of(fresh) == spec
    assert torch.allclose(logits(model, (3, 4, 1), (0, 3)), logits(fresh, (3, 4, 1), (0, 3)))


def test_save_without_kept_ids_loads_none(model, make_model, tmp_path):
    surgery.save(model, tmp_path / "m")
    assert surgery.load(make_model(seed=3), tmp_path / "m") is None
