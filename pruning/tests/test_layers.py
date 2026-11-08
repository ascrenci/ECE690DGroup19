import layers


def test_influence_gives_one_nonnegative_score_per_decoder_block(model, tok, rows):
    scores = layers.influence(model, tok, rows, 16, 16, batch_size=2)
    assert len(scores) == len(model.decoder.block) and all(s >= 0 for s in scores)


def test_choose_keeps_block_zero_and_the_highest_influence_others():
    scores = [9.0, 0.1, 0.5, 0.3, 0.9]  # block 0's score is ignored: it is always kept
    assert layers.choose(scores, 1) == [0]
    assert layers.choose(scores, 3) == [0, 2, 4]
    assert layers.choose(scores, 5) == [0, 1, 2, 3, 4]
