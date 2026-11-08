import torch

import channel
import surgery


def test_ffn_importance_ranks_by_l2_magnitude(model):
    ffn = surgery.ffn_of(surgery.blocks(model, "encoder")[0])
    ffn.wi_0.weight.data.zero_(), ffn.wi_1.weight.data.zero_(), ffn.wo.weight.data.zero_()
    ffn.wo.weight.data[:, 7] = 2.0  # channel 7 dominates
    ffn.wi_0.weight.data[3] = 1.0
    scores = channel.ffn_importance(ffn)
    assert scores.argmax() == 7 and scores[3] > 0 and scores[0] == 0


def test_head_importance_sums_over_layers_and_cross_attention(model):
    for stack in surgery.STACKS:
        for block in surgery.blocks(model, stack):
            for attn in surgery.attentions(block):
                attn.o.weight.data[:, 8:12] *= 10  # head 2 gets large output weights everywhere
    assert channel.head_importance(model, "encoder").argmax() == 2
    assert channel.head_importance(model, "decoder").argmax() == 2


def test_top_returns_ascending_indices():
    assert channel.top(torch.tensor([0.1, 5.0, 0.3, 4.0]), 2).tolist() == [1, 3]


def test_apply_keeps_highest_magnitude_channels_and_heads(model):
    ffn = surgery.ffn_of(surgery.blocks(model, "encoder")[0])
    ffn.wo.weight.data[:, 9] = 50.0
    channel.apply(model, {"encoder": [1, 8], "decoder": [8, 8]}, {"encoder": 2, "decoder": 3})
    new = surgery.ffn_of(surgery.blocks(model, "encoder")[0])
    assert new.wo.weight.shape == (16, 1) and torch.all(new.wo.weight == 50.0)
    assert surgery.spec_of(model)["heads"] == {"encoder": 2, "decoder": 3}


def test_uniform_scales_every_layer_and_keeps_at_least_one_head(model):
    ffn, heads = channel.uniform(model, 0.5)
    assert ffn == {"encoder": [16, 16], "decoder": [16, 16]} and heads == {"encoder": 2, "decoder": 2}
    assert channel.uniform(model, 0.99)[1] == {"encoder": 1, "decoder": 1}


def test_uniform_only_prunes_one_part(model):
    ffn, heads = channel.uniform(model, 0.5, only="ffn")
    assert ffn["encoder"] == [16, 16] and heads == {"encoder": 4, "decoder": 4}
    ffn, heads = channel.uniform(model, 0.5, only="heads")
    assert ffn["encoder"] == [32, 32] and heads == {"encoder": 2, "decoder": 2}
