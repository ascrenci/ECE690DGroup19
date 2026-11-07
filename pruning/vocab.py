"""Method 3: vocabulary pruning. Drop embedding / lm_head rows for tokens the data never uses.

The 32k vocab is ~41% of flan-t5-small's weights (and is duplicated in both Core ML files), so this is
the biggest size lever. Token ids change, so a tokenizer wrapper remaps them: tokens outside the kept
set become <unk> on the way in, kept ids map back to original ids on the way out.
"""
import torch

import latex_chars
import surgery


def kept_ids(tok, rows, batch=1000):
    """Ascending ids of the specials plus every token in the prompt+spoken inputs and targets of `rows`."""
    ids = {tok.pad_token_id, tok.eos_token_id, tok.unk_token_id}
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        for texts in ([latex_chars.PROMPT + r["spoken"] for r in chunk], [r["target"] for r in chunk]):
            for seq in tok(texts, add_special_tokens=True).input_ids:
                ids.update(seq)
    return sorted(ids)


class Remapped:
    """Tokenizer view over a pruned vocabulary (see module docstring). Other attributes pass through."""

    def __init__(self, tok, kept, old_vocab):
        self.tok = tok
        self.kept = torch.as_tensor(kept)
        assert self.kept[:3].tolist() == [0, 1, 2], "specials must stay at ids 0, 1, 2"
        self.lut = torch.full((old_vocab,), tok.unk_token_id)
        self.lut[self.kept] = torch.arange(len(self.kept))

    def __getattr__(self, name):
        return getattr(self.tok, name)

    def _map(self, ids):
        if torch.is_tensor(ids):
            return self.lut[ids]
        return [self._map(x) if isinstance(x, list) else int(self.lut[x]) for x in ids]

    def __call__(self, *args, **kw):
        enc = self.tok(*args, **kw)
        enc["input_ids"] = self._map(enc["input_ids"])
        return enc

    def _back(self, ids):
        return self.kept[torch.as_tensor(ids).cpu()].tolist()

    def decode(self, ids, **kw):
        return self.tok.decode(self._back(ids), **kw)

    def batch_decode(self, batch, **kw):
        return self.tok.batch_decode(self._back(batch), **kw)


def apply(model, tok, kept):
    """Prune `model`'s vocabulary to `kept`; returns the matching tokenizer wrapper."""
    old = model.config.vocab_size
    surgery.prune_vocab(model, kept)
    return Remapped(tok, kept, old)
