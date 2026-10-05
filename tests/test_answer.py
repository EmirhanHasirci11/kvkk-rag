"""Run from the repo root: python -m pytest tests -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from answer import is_no_info, parse_citations
from run_answers import citation_check

CHUNKS = [
    {"rank": 1, "doc": "kvkk_kanun_6698", "section": "Madde 11", "sections": ["Madde 11", "Madde 12"],
     "text": "... veri sorumlusu bu durumu en kısa sürede ilgilisine ve Kurula bildirir. ..."},
    {"rank": 2, "doc": "rehber_veri_guvenligi", "section": "3.1", "sections": ["3.1"], "text": "güvenlik duvarı"},
]


def test_parse_citations_matches_chunks():
    cites = parse_citations("Kurula bildirilir [kvkk_kanun_6698, Madde 12]. Ayrıca [rehber_veri_guvenligi, 3.1] "
                            "ve tekrar [kvkk_kanun_6698,  madde 12].", CHUNKS)
    assert cites == [{"doc": "kvkk_kanun_6698", "section": "Madde 12", "chunk_ranks": [1]},
                     {"doc": "rehber_veri_guvenligi", "section": "3.1", "chunk_ranks": [2]}]


def test_citation_not_in_retrieved_chunks():
    assert parse_citations("[kvkk_kanun_6698, Madde 30]", CHUNKS)[0]["chunk_ranks"] == []


def test_citation_check():
    fact = [["bu durumu en kısa sürede ilgilisine ve Kurula bildirir."]]
    r = {"retrieved": CHUNKS, "citations": parse_citations("x [kvkk_kanun_6698, Madde 12]", CHUNKS)}
    assert citation_check(r, fact) == "pass"
    r["citations"] = parse_citations("x [rehber_veri_guvenligi, 3.1]", CHUNKS)
    assert citation_check(r, fact) == "fail"
    r["citations"] = []
    assert citation_check(r, fact) == "no_citation"


def test_no_info():
    assert is_no_info("Bu konuda verilen metinlerde bilgi yok.")
    assert not is_no_info("Kurula bildirilir.")
