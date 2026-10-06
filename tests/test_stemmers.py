"""Run from the repo root: python -m pytest tests -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from textproc import STEMMERS, tokenize


def test_snowball_strips_suffixes():
    assert tokenize("verilerin yükümlülüğü", "snow") == ["ver", "yükümlülük"]


def test_zeyrek_lemma_and_unknown_word():
    assert tokenize("verilerin kişi", "zeyrek") == ["veri", "kişi"]
    # not in zeyrek's lexicon: kept as it is
    assert tokenize("hacklendi", "zeyrek") == ["hacklendi"]


def test_every_stemmer_handles_empty_text():
    assert all(tokenize("", stem) == [] for stem in STEMMERS)
