#!/usr/bin/env python3
"""Convert a T5 checkpoint to Core ML (encoder + decoder) for on-device use.

Outputs, under --out:
  T5Encoder.mlpackage     input_ids, attention_mask            -> last_hidden_state
  T5Decoder.mlpackage     decoder_input_ids, encoder_hidden_states,
                          encoder_attention_mask, position     -> logits (1, vocab)
  t5_tokenizer/           tokenizer.json, tokenizer_config.json, config.json, t5_meta.json

First pass, deliberately simple (pruning/quantization/KV-cache come later):
  * fixed sequence lengths (inputs are padded),
  * decoder recomputes the whole prefix every step (no KV cache),
  * float32 weights (T5 is fragile in fp16),
  * the decoder only returns logits for ONE position, selected by `position`,
    so we don't compute/copy a (max_dec x vocab) logits tensor every step.

Conversion works on any OS, but validation (running the Core ML models) needs macOS.
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

import latex_chars

PROMPT = latex_chars.PROMPT  # shared with training; also written to t5_meta.json for the app


class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]


class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        # t5-small ties input/output embeddings and rescales; flan-t5 does not.
        self.scale = model.config.d_model**-0.5 if model.config.tie_word_embeddings else 1.0

    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask, position):
        hidden = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]
        hidden = torch.index_select(hidden, 1, position) * self.scale  # (1, 1, d_model)
        return self.lm_head(hidden)[:, 0, :]  # (1, vocab)


def convert(model, enc_len, dec_len, out):
    import coremltools as ct

    d_model = model.config.d_model
    ids = torch.zeros((1, enc_len), dtype=torch.int32)
    mask = torch.ones((1, enc_len), dtype=torch.int32)
    dec_ids = torch.zeros((1, dec_len), dtype=torch.int32)
    hidden = torch.zeros((1, enc_len, d_model), dtype=torch.float32)
    pos = torch.zeros((1,), dtype=torch.int32)

    def to_ml(module, example, inputs, outputs, path):
        print(f"tracing {path.name} ...")
        traced = torch.jit.trace(module.eval(), example, strict=False)
        print(f"converting {path.name} ...")
        ml = ct.convert(
            traced,
            inputs=inputs,
            outputs=outputs,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32,
        )
        ml.save(str(path))

    to_ml(
        EncoderWrapper(model),
        (ids, mask),
        [
            ct.TensorType("input_ids", shape=(1, enc_len), dtype=np.int32),
            ct.TensorType("attention_mask", shape=(1, enc_len), dtype=np.int32),
        ],
        [ct.TensorType("last_hidden_state")],
        out / "T5Encoder.mlpackage",
    )
    to_ml(
        DecoderWrapper(model),
        (dec_ids, hidden, mask, pos),
        [
            ct.TensorType("decoder_input_ids", shape=(1, dec_len), dtype=np.int32),
            ct.TensorType("encoder_hidden_states", shape=(1, enc_len, d_model), dtype=np.float32),
            ct.TensorType("encoder_attention_mask", shape=(1, enc_len), dtype=np.int32),
            ct.TensorType("position", shape=(1,), dtype=np.int32),
        ],
        [ct.TensorType("logits")],
        out / "T5Decoder.mlpackage",
    )


def validate(model, tokenizer, enc_len, dec_len, out, max_new=24):
    """Compare Core ML greedy decoding against PyTorch. Needs macOS."""
    if platform.system() != "Darwin":
        print("skipping validation (Core ML runtime needs macOS)")
        return
    import coremltools as ct

    cpu = ct.ComputeUnit.CPU_ONLY
    enc = ct.models.MLModel(str(out / "T5Encoder.mlpackage"), compute_units=cpu)
    dec = ct.models.MLModel(str(out / "T5Decoder.mlpackage"), compute_units=cpu)

    text = PROMPT + "x squared plus y squared equals r squared"
    token_ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()  # includes </s>
    assert len(token_ids) <= enc_len, "validation prompt longer than encoder length"
    n = len(token_ids)
    ids = np.zeros((1, enc_len), dtype=np.int32)
    ids[0, :n] = token_ids
    mask = np.zeros((1, enc_len), dtype=np.int32)
    mask[0, :n] = 1

    hidden = enc.predict({"input_ids": ids, "attention_mask": mask})["last_hidden_state"]

    start, eos = model.config.decoder_start_token_id, model.config.eos_token_id
    seq = [start]
    first_logits = None
    for step in range(min(max_new, dec_len - 1)):
        dec_in = np.zeros((1, dec_len), dtype=np.int32)
        dec_in[0, : len(seq)] = seq
        logits = dec.predict(
            {
                "decoder_input_ids": dec_in,
                "encoder_hidden_states": hidden,
                "encoder_attention_mask": mask,
                "position": np.array([step], dtype=np.int32),
            }
        )["logits"]
        if first_logits is None:
            first_logits = logits[0]
        nxt = int(np.argmax(logits[0]))
        seq.append(nxt)
        if nxt == eos:
            break

    with torch.no_grad():
        ref_in = torch.tensor([token_ids])
        ref_logits = model(input_ids=ref_in, decoder_input_ids=torch.tensor([[start]])).logits[0, 0].numpy()
        ref = model.generate(ref_in, do_sample=False, max_new_tokens=max_new)[0].tolist()

    diff = float(np.abs(first_logits - ref_logits).max())
    print(f"first-step max |logit diff| vs PyTorch: {diff:.5f}")
    print(f"Core ML : {tokenizer.decode(seq, skip_special_tokens=True)!r}")
    print(f"PyTorch : {tokenizer.decode(ref, skip_special_tokens=True)!r}")
    if seq[: len(ref)] != ref[: len(seq)]:
        print("WARNING: greedy outputs differ")
    if diff > 0.05:
        sys.exit(f"validation failed: logits differ by {diff}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/flan-t5-small")
    ap.add_argument("--out", type=Path, default=Path("HelloWorld/Models"))
    ap.add_argument("--enc-len", type=int, default=128, help="max encoder tokens (padded)")
    ap.add_argument("--dec-len", type=int, default=128, help="max decoder tokens (padded)")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tok_dir = args.out / "t5_tokenizer"

    print(f"loading {args.model} ...")
    model = T5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    latex_chars.check_tokenizer(tokenizer)  # the LaTeX<->T5 character mapping relies on this vocab

    tokenizer.save_pretrained(tok_dir)
    model.config.save_pretrained(tok_dir)  # swift-transformers wants config.json next to the tokenizer
    (tok_dir / "t5_meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt": PROMPT,
                "maxEncoderLength": args.enc_len,
                "maxDecoderLength": args.dec_len,
                "padTokenId": model.config.pad_token_id,
                "eosTokenId": model.config.eos_token_id,
                "decoderStartTokenId": model.config.decoder_start_token_id,
                "vocabSize": model.config.vocab_size,
                # T5 output -> LaTeX, applied in order (see latex_chars.py)
                "latexDecode": [list(step) for step in latex_chars.DECODE_STEPS],
            },
            indent=2,
        )
    )

    convert(model, args.enc_len, args.dec_len, args.out)
    if not args.no_validate:
        validate(model, tokenizer, args.enc_len, args.dec_len, args.out)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
