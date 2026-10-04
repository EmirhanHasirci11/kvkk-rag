"""Run from the repo root: python -m pytest tests -v"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bm25 import BM25
from textproc import tokenize

DOCS = [
    ["kişisel", "veri", "silme"],             # len 3
    ["veri", "veri", "güvenlik", "önlem"],    # len 4
    ["açık", "rıza"],                         # len 2
]                                             # N = 3, avgdl = 3


@pytest.fixture
def bm():
    return BM25(DOCS)


def test_idf_values(bm):
    assert bm.idf("veri") == pytest.approx(math.log(1 + 1.5 / 2.5))   # df = 2
    assert bm.idf("rıza") == pytest.approx(math.log(1 + 2.5 / 1.5))   # df = 1


def test_rare_term_has_higher_idf(bm):
    assert bm.idf("rıza") > bm.idf("veri")


def test_idf_positive_even_if_term_is_everywhere():
    bm = BM25([["a", "b"], ["a"], ["a", "c"]])
    assert bm.idf("a") > 0


def test_exact_scores(bm):
    # doc 0: tf=1, len=avgdl  -> 1 * 2.5 / (1 + 1.5 * 1.00) = 1.0
    assert bm.score(["veri"], 0) == pytest.approx(math.log(1.6) * 1.0)
    # doc 1: tf=2, len=4      -> 2 * 2.5 / (2 + 1.5 * 1.25) = 5 / 3.875
    assert bm.score(["veri"], 1) == pytest.approx(math.log(1.6) * 5 / 3.875)


def test_no_matching_terms_scores_zero(bm):
    assert bm.score(["veri"], 2) == 0
    assert bm.score(["olmayan"], 0) == 0


def test_query_terms_add_up(bm):
    both = bm.score(["kişisel", "veri"], 0)
    assert both == pytest.approx(bm.score(["kişisel"], 0) + bm.score(["veri"], 0))


def test_shorter_doc_wins_with_same_tf():
    bm = BM25([["veri", "x"], ["veri", "x", "y", "z", "w", "v"]])
    assert bm.score(["veri"], 0) > bm.score(["veri"], 1)


def test_b_zero_turns_off_length_normalization():
    bm = BM25([["veri", "x"], ["veri", "x", "y", "z", "w", "v"]], b=0)
    assert bm.score(["veri"], 0) == pytest.approx(bm.score(["veri"], 1))


def test_tf_saturates():
    bm = BM25([["veri"] * 1 + ["x"] * 9, ["veri"] * 5 + ["x"] * 5, ["veri"] * 9 + ["x"]])
    s1, s5, s9 = (bm.score(["veri"], i) for i in range(3))
    assert s1 < s5 < s9
    assert (s9 - s5) < (s5 - s1)          # each extra occurrence adds less
    assert s9 < bm.idf("veri") * (bm.k1 + 1)  # upper bound


def test_search_sorted_and_cut(bm):
    res = bm.search(["veri", "silme"], k=2)
    assert len(res) == 2
    assert [i for i, _ in res] == [0, 1]
    assert res[0][1] >= res[1][1]


def test_search_k_larger_than_corpus(bm):
    assert len(bm.search(["veri"], k=10)) == 3


def test_tokenize_turkish_case():
    assert tokenize("İLGİLİ Kişi IRAK") == ["ilgili", "kişi", "ırak"]


def test_tokenize_punctuation_and_stem():
    assert tokenize("Madde 12, (1) verilerin;") == ["madde", "12", "1", "verilerin"]
    assert tokenize("silinmesi silinir", stem="prefix5") == ["silin", "silin"]
