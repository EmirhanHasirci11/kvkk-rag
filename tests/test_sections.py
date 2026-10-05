"""Run from the repo root: python -m pytest tests -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chunking import chunk_words
from sections import chunk_sections, section_marks

LAW = "Kanun adı\nAmaç\nMADDE 1 - (1) a b c\nKapsam\nMADDE 2 - (1) d e f g h\nGEÇİCİ\nMADDE 1 - (1) i j"


def test_article_marks_and_temporary():
    words = LAW.split()
    marks = section_marks(LAW)
    assert [label for _, label in marks] == ["Madde 1", "Madde 2", "Geçici Madde 1"]
    # each index points at the word "MADDE"
    assert all(words[i] == "MADDE" for i, _ in marks)


def test_temporary_on_same_line():
    assert section_marks("GEÇİCİ MADDE 3 - (1) x")[0][1] == "Geçici Madde 3"


def test_guide_sections():
    text = "Giriş metni\n1. BAŞLIK\nx y\n1.1. Alt başlık\nz\n3) BAŞKA\nw"
    assert [label for _, label in section_marks(text)] == ["1", "1.1", "3"]


def test_law_ignores_numbered_lines():
    text = "MADDE 1 - (1) x\n1. madde içi liste\ny"
    assert [label for _, label in section_marks(text)] == ["Madde 1"]


def test_chunk_carries_last_article():
    meta = chunk_sections(LAW, size=4, overlap=0)
    assert len(meta) == len(chunk_words(LAW, 4, 0))
    # chunk 0 starts at the title: no label yet, but covers Madde 1
    assert meta[0] == {"section": None, "sections": ["Madde 1"]}
    # chunk 1 = "(1) a b c": starts mid-article, carries Madde 1
    assert meta[1]["section"] == "Madde 1"
    # chunk 2 = "Kapsam MADDE 2 - (1)": still Madde 1 at its first word, then Madde 2
    assert meta[2] == {"section": "Madde 1", "sections": ["Madde 1", "Madde 2"]}


def test_chunk_starting_on_heading():
    text = "MADDE 1 - a MADDE 2 - c d"
    meta = chunk_sections(text, size=4, overlap=0)
    assert meta[1] == {"section": "Madde 2", "sections": ["Madde 2"]}
