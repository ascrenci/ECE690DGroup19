"""Real CPU torch + a tiny random T5 and a tiny word-level tokenizer (ids 0/1/2 = pad/eos/unk like T5)."""
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

WORDS = ["Convert", "spoken", "math", "to", "LaTeX", ":", "x", "y", "squared", "plus", "one", "two", "!", "2", "1", "+",
         "$", "{", "}", "equals", "r"]


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    """Never touch the GPU from tests (a training run may be using it)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def make_model():
    def _make(seed=0):
        torch.manual_seed(seed)
        cfg = T5Config(vocab_size=64, d_model=16, d_kv=4, d_ff=32, num_layers=2, num_decoder_layers=2, num_heads=4,
                       relative_attention_num_buckets=8, feed_forward_proj="gated-gelu", tie_word_embeddings=False,
                       pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
        return T5ForConditionalGeneration(cfg).eval()

    return _make


@pytest.fixture
def model(make_model):
    return make_model()


@pytest.fixture
def tok():
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
                                   model_input_names=["input_ids", "attention_mask"])


@pytest.fixture
def rows():
    return [
        {"spoken": "x squared plus one", "latex": "x ^ 2 + 1", "target": "x ! 2 + 1"},
        {"spoken": "y equals two", "latex": "y = 2", "target": "y equals 2"},
        {"spoken": "r plus x", "latex": "r + x", "target": "r + x"},
    ]


def logits(model, enc_ids=(5, 6, 7, 1), dec_ids=(0, 8, 9)):
    with torch.no_grad():
        return model(input_ids=torch.tensor([enc_ids]), decoder_input_ids=torch.tensor([dec_ids])).logits
