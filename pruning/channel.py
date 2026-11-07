"""Method 1: plain channel pruning. Keep the FFN channels and attention heads with the largest L2 magnitude."""
import surgery
from surgery import STACKS


def ffn_importance(ffn):
    """Squared L2 magnitude of each d_ff channel across the three FFN matrices."""
    return ffn.wi_0.weight.norm(dim=1) ** 2 + ffn.wi_1.weight.norm(dim=1) ** 2 + ffn.wo.weight.norm(dim=0) ** 2


def head_importance(model, stack):
    """Squared L2 magnitude of each head's q/k/v/o slices, summed over the stack (cross-attention included)."""
    total = 0
    for block in surgery.blocks(model, stack):
        for attn in surgery.attentions(block):
            for w in (attn.q.weight, attn.k.weight, attn.v.weight, attn.o.weight.t()):
                total = total + w.reshape(attn.n_heads, -1).norm(dim=1) ** 2
    return total


def top(scores, k):
    """Indices of the k largest scores, ascending."""
    return scores.topk(k).indices.sort().values


def apply(model, ffn_widths, heads):
    """Prune to ffn_widths[stack] (a width per layer) and heads[stack] (a count), choosing by magnitude."""
    keep_ffn = {
        s: [top(ffn_importance(surgery.ffn_of(b)), w) for b, w in zip(surgery.blocks(model, s), ffn_widths[s])]
        for s in STACKS
    }
    keep_heads = {s: top(head_importance(model, s), heads[s]) for s in STACKS}  # scored before any slicing
    surgery.prune_ffn(model, keep_ffn)
    surgery.prune_heads(model, keep_heads)


def uniform(model, level):
    """Same keep ratio (1 - level) for every layer's FFN and every stack's heads: (ffn_widths, heads)."""
    cfg = model.config
    width = round(cfg.d_ff * (1 - level))
    count = max(1, round(cfg.num_heads * (1 - level)))
    return (
        {"encoder": [width] * cfg.num_layers, "decoder": [width] * cfg.num_decoder_layers},
        {"encoder": count, "decoder": count},
    )
