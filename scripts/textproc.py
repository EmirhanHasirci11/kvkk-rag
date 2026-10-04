"""Turkish text normalization and tokenization for lexical search (BM25)."""
import re

TOKEN_RE = re.compile(r"[a-zçğıöşüâîû0-9]+")


def tr_lower(s):
    # Python's lower() turns "I" into "i" and "İ" into "i̇", both wrong for Turkish
    return s.replace("I", "ı").replace("İ", "i").lower()


def stem_prefix5(word):
    # crude Turkish stemmer: keep the first 5 letters, so "silinmesi" and "silinir" match
    return word[:5]


STEMMERS = {
    "none": lambda w: w,
    "prefix5": stem_prefix5,
}


def tokenize(text, stem="none"):
    words = TOKEN_RE.findall(tr_lower(text))
    f = STEMMERS[stem]
    return [f(w) for w in words]
