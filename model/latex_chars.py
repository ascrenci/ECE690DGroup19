"""Reversible LaTeX <-> T5-safe text mapping.

T5's SentencePiece vocab cannot produce these ASCII characters (they become <unk>), measured
against google/flan-t5-small's tokenizer.json:

    \\  ^  `  {  }  ~  <

Everything else printable in ASCII round-trips (letters, digits, and  !"#$%&'()*+,-./:;=>?@[]_| ).
Whitespace is collapsed to single spaces and trimmed (leading/trailing), and non-ASCII
(e.g. U+2212, U+00B7) is not reliably producible, so rows containing it are dropped.

Scheme: "@" plays the role of the backslash, so commands read naturally (@frac, @alpha, @pm).
The frequent characters get one-character stand-ins; the rare ones get pseudo-commands.

    \\ -> @        { -> "        } -> ?        ^ -> !
    <  -> @lt      ~ -> @nbsp    ` -> @bq

The stand-in characters (@ " ? !) are therefore *reserved*: an equation that already contains
one is rejected (it'd be ambiguous), costing well under 1% of MathBridge rows.

Single source of truth: DECODE_STEPS is also written into t5_meta.json by convert_t5.py so the
Swift app applies exactly the same table.
"""
import re

# Task prefix. Baked into training data AND inference (t5_meta.json), so change it in one place only.
PROMPT = "Convert spoken math to LaTeX: "

# T5 -> LaTeX, applied top to bottom. Order matters: pseudo-commands before the bare "@".
DECODE_STEPS = [
    ("@lt", "<"),
    ("@nbsp", "~"),
    ("@bq", "`"),
    ("@", "\\"),
    ('"', "{"),
    ("?", "}"),
    ("!", "^"),
]

RESERVED = "@\"?!"
UNPRODUCIBLE = "\\^`{}~<"  # what T5 can't emit (verified by check_tokenizer)

_WS = re.compile(r"\s+")
_DOLLARS = re.compile(r"^\s*\$+\s*(.*?)\s*\$+\s*$", re.DOTALL)


def normalize(latex: str) -> str:
    """Strip $...$ wrapper (MathBridge wraps every equation) and collapse whitespace."""
    m = _DOLLARS.match(latex)
    if m:
        latex = m.group(1)
    return _WS.sub(" ", latex).strip()


def normalize_spoken(text: str) -> str:
    """Canonical form of the *input* side: collapse whitespace, drop trailing . , ; :

    Case is kept on purpose ("Xi" and "xi" are different symbols). The Swift app applies the same rule
    to Whisper output before calling the model.
    """
    return _WS.sub(" ", text).strip().rstrip(".,;: ")


def encode(latex: str) -> str:
    """LaTeX -> T5-safe text. Raises ValueError if it can't be represented unambiguously."""
    s = normalize(latex)
    if not s:
        raise ValueError("empty equation")
    if any(c in s for c in RESERVED):
        raise ValueError("contains a reserved character")
    if any(ord(c) > 126 for c in s):
        raise ValueError("non-ASCII character")
    for src, dst in reversed(DECODE_STEPS):
        s = s.replace(dst, src)
    return s


def decode(text: str) -> str:
    """T5 output -> LaTeX."""
    s = text.strip()
    for src, dst in DECODE_STEPS:
        s = s.replace(src, dst)
    return s


def check_tokenizer(tokenizer) -> None:
    """Fail loudly if the tokenizer can't produce what the mapping relies on.

    `tokenizer` is a transformers tokenizer. Verifies (a) every character we map TO round-trips,
    and (b) the characters in UNPRODUCIBLE really do not (so we notice if the vocab changes).
    """
    unk = tokenizer.unk_token_id

    def produces(c):
        ids = tokenizer("a" + c + "b", add_special_tokens=False).input_ids
        return unk not in ids and tokenizer.decode(ids, skip_special_tokens=True) == "a" + c + "b"

    targets = set("".join(src for src, _ in DECODE_STEPS))
    broken = sorted(c for c in targets if not produces(c))
    if broken:
        raise AssertionError(f"tokenizer cannot produce stand-in characters: {broken}")
    now_ok = [c for c in UNPRODUCIBLE if produces(c)]
    if now_ok:
        print(f"note: tokenizer can now produce {now_ok}; the mapping for these is unnecessary")


def roundtrip_ok(latex: str, tokenizer) -> bool:
    """True if `latex` survives LaTeX -> T5 text -> token ids -> text -> LaTeX unchanged.

    Use this to filter training data: it drops anything with reserved/non-ASCII characters and
    any equation that collides with a pseudo-command (e.g. a custom macro literally named \bq).
    """
    try:
        s = encode(latex)
    except ValueError:
        return False
    ids = tokenizer(s, add_special_tokens=False).input_ids
    if tokenizer.unk_token_id in ids:
        return False
    return decode(tokenizer.decode(ids, skip_special_tokens=True)) == normalize(latex)
