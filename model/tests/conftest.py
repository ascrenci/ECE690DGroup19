"""Fakes for torch / transformers / datasets / coremltools so tests run anywhere, offline, instantly.

The fakes are installed into sys.modules when this file is imported, i.e. before any test module
imports the code under test. Tests prove our logic, not the real libraries; the macOS CI build
(build.yml) is the real end-to-end check.
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from latex_chars import UNPRODUCIBLE

UNK, EOS, PAD = 2, 1, 0


class Enc(dict):
    """Tokenizer output: dict + attribute access + .to() like transformers' BatchEncoding."""

    def __getattr__(self, key):
        return self[key]

    def to(self, _device):
        return self


class FakeTokenizer:
    """One token per character (id = ord). UNPRODUCIBLE and non-ASCII characters become <unk>."""

    unk_token_id = UNK

    def _ids(self, text):
        return [UNK if (c in UNPRODUCIBLE or ord(c) > 126) else ord(c) for c in text]

    def __call__(self, text=None, text_target=None, return_tensors=None, **_kw):
        text = text_target if text_target is not None else text
        if isinstance(text, list):
            if not text:
                raise IndexError("list index out of range")  # what a real fast tokenizer does on []
            return Enc(input_ids=[self._ids(t) for t in text])
        ids = self._ids(text)
        if return_tensors == "pt":
            return Enc(input_ids=np.array([ids]), attention_mask=np.ones((1, len(ids)), dtype=np.int32))
        return Enc(input_ids=ids)

    def decode(self, ids, skip_special_tokens=False, **_kw):
        out = []
        for i in ids:
            if i == UNK:
                out.append("<unk>")
            elif i in (PAD, EOS) and skip_special_tokens:
                continue
            else:
                out.append(chr(i))
        return "".join(out)

    def batch_decode(self, batch, **kw):
        return [self.decode(list(row), **kw) for row in batch]

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}")


class FakeModel:
    """Stands in for T5ForConditionalGeneration (and its encoder/decoder/lm_head)."""

    def __init__(self, vocab=8, d_model=4, tied=False, generated=(5, 6, 1)):
        self.config = SimpleNamespace(
            d_model=d_model, tie_word_embeddings=tied, pad_token_id=PAD, eos_token_id=EOS,
            decoder_start_token_id=PAD, vocab_size=vocab,
        )
        self.vocab, self.generated = vocab, list(generated)
        self.cuda_device = None
        self.logits = np.zeros((1, 1, vocab))
        self.encoder = lambda **_kw: (np.ones((1, 3, d_model)),)
        self.decoder = lambda **_kw: (np.ones((1, 5, d_model)),)
        self.lm_head = lambda h: np.zeros((h.shape[0], h.shape[1], vocab)) + 1
        self.config.save_pretrained = lambda path: (path / "config.json").write_text("{}")

    def eval(self):
        return self

    def to(self, device):
        self.cuda_device = device
        return self

    def parameters(self):
        yield SimpleNamespace(device="cpu")

    def generate(self, *_a, **kw):
        n = len(kw["input_ids"]) if "input_ids" in kw else 1
        return np.array([self.generated] * n)

    def __call__(self, **_kw):
        return SimpleNamespace(logits=self.logits)


class FakeDataset:
    """The tiny slice of datasets.Dataset that train_t5.py uses."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def __len__(self):
        return len(self.rows)

    def select(self, idx):
        return FakeDataset([self.rows[i] for i in idx])

    def map(self, fn, batched, remove_columns):
        batch = {k: [r[k] for r in self.rows] for k in self.column_names}
        out = fn(batch)
        n = len(next(iter(out.values())))
        return FakeDataset([{k: v[i] for k, v in out.items()} for i in range(n)])


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


class _Module:
    def __init__(self, *a, **k):
        pass

    def eval(self):
        return self


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class _Arr(np.ndarray):
    def numpy(self):
        return np.asarray(self)


def _arr(x, dtype=None):
    return np.asarray(x, dtype=dtype).view(_Arr)


torch = _module(
    "torch",
    float32=np.float32, int32=np.int32,
    zeros=lambda shape, dtype=None: _arr(np.zeros(shape), dtype),
    ones=lambda shape, dtype=None: _arr(np.ones(shape), dtype),
    tensor=lambda x: _arr(x),
    index_select=lambda t, dim, idx: np.take(t, idx, axis=dim),
    no_grad=_NoGrad,
    cuda=SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False),
    jit=SimpleNamespace(trace=lambda module, example, strict=False: module),
    nn=SimpleNamespace(Module=_Module),
)
sys.modules["torch.nn"] = torch.nn

transformers = _module(
    "transformers",
    AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_a, **_k: FakeTokenizer()),
    T5ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *_a, **_k: FakeModel()),
    DataCollatorForSeq2Seq=lambda *a, **k: SimpleNamespace(args=a, kwargs=k),
    Seq2SeqTrainingArguments=lambda **k: SimpleNamespace(**k),
)


class FakeTrainer:
    last = None

    def __init__(self, **kw):
        self.kw, self.resume, self.saved = kw, "unset", None
        FakeTrainer.last = self

    def train(self, resume_from_checkpoint=None):
        self.resume = resume_from_checkpoint

    def save_model(self, path):
        self.saved = path


transformers.Seq2SeqTrainer = FakeTrainer
_module("datasets", load_dataset=lambda *_a, **_k: FakeDataset([]))

# coremltools: records what convert() asks for; models "predict" via callables set by the test.
saved = []


class _MLModel:
    def __init__(self, path=None, compute_units=None):
        self.path = path

    def save(self, path):
        saved.append(path)

    def predict(self, feed):
        return _MLModel.predictor(self.path, feed)

    predictor = staticmethod(lambda path, feed: {})


_module(
    "coremltools",
    convert=lambda *_a, **_k: _MLModel(),
    TensorType=lambda *a, **k: SimpleNamespace(args=a, kwargs=k),
    target=SimpleNamespace(iOS17="iOS17"),
    precision=SimpleNamespace(FLOAT32="fp32"),
    ComputeUnit=SimpleNamespace(CPU_ONLY="cpu"),
    models=SimpleNamespace(MLModel=_MLModel),
)


@pytest.fixture
def tok():
    return FakeTokenizer()


@pytest.fixture
def coreml():
    saved.clear()
    yield sys.modules["coremltools"], saved
