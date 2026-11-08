"""Method 4: decoder layer dropping. Remove the decoder blocks that change the hidden states the least.

Width pruning barely changes decode latency (the layers still run one after another per token); dropping
whole decoder blocks does. Block 0 is always kept: it owns T5's relative-position bias.
"""
import torch
import torch.nn.functional as F

import metrics


def influence(model, tok, rows, max_source=127, max_target=127, batch_size=32):
    """Per decoder block: mean (1 - cosine similarity) between its input and output hidden states.

    Higher means the block changes more. The last block's output has the final layer norm applied by T5.
    """
    model.eval()
    device = next(model.parameters()).device
    total, count = torch.zeros(len(model.decoder.block)), 0
    for enc, labels in metrics.batches(tok, rows, max_source, max_target, batch_size):
        labels = labels.to(device)
        with torch.no_grad():
            hidden = model(**enc.to(device), labels=labels, output_hidden_states=True).decoder_hidden_states
        mask = labels != -100
        for i in range(len(total)):
            total[i] += (1 - F.cosine_similarity(hidden[i], hidden[i + 1], dim=-1))[mask].sum().cpu()
        count += int(mask.sum())
    return (total / count).tolist()


def choose(scores, n_keep):
    """Block indices to keep, ascending: block 0 plus the n_keep - 1 highest-influence others."""
    ranked = sorted(range(1, len(scores)), key=lambda i: -scores[i])
    return [0] + sorted(ranked[: n_keep - 1])
