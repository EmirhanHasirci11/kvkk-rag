"""Turkish text normalization and tokenization for lexical search (BM25)."""
import re
from functools import lru_cache

TOKEN_RE = re.compile(r"[a-zçğıöşüâîû0-9]+")


def tr_lower(s):
    # Python's lower() turns "I" into "i" and "İ" into "i̇", both wrong for Turkish
    return s.replace("I", "ı").replace("İ", "i").lower()


def stem_prefix5(word):
    # crude Turkish stemmer: keep the first 5 letters, so "silinmesi" and "silinir" match
    return word[:5]


@lru_cache(maxsize=None)
def _snowball():
    import snowballstemmer
    return snowballstemmer.stemmer("turkish")


@lru_cache(maxsize=None)
def stem_snowball(word):
    # Turkish Snowball stemmer: strips suffixes by rule, no dictionary
    return _snowball().stemWord(word)


@lru_cache(maxsize=None)
def _zeyrek():
    import logging
    import zeyrek
    # zeyrek logs every analysis it finds as a warning
    logging.getLogger("zeyrek.rulebasedanalyzer").setLevel(logging.ERROR)
    return zeyrek.MorphAnalyzer()


@lru_cache(maxsize=None)
def stem_zeyrek(word):
    # zeyrek (Python port of Zemberek) has no disambiguation, so take the lemma of its first
    # analysis; words it does not know stay as they are
    analyses = _zeyrek()._parse(word)    # word-level parse, skips zeyrek's NLTK tokenizer
    return tr_lower(analyses[0].dict_item.lemma) if analyses else word


STEMMERS = {
    "none": lambda w: w,
    "prefix5": stem_prefix5,
    "snow": stem_snowball,
    "zeyrek": stem_zeyrek,
}


def tokenize(text, stem="none"):
    words = TOKEN_RE.findall(tr_lower(text))
    f = STEMMERS[stem]
    return [f(w) for w in words]
