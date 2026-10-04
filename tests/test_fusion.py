"""Run from the repo root: python -m pytest tests -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fusion import rrf


def test_item_in_both_lists_rises():
    # "b" is only 2nd in each list, but it is the only item both lists agree on
    assert rrf([["a", "b"], ["c", "b"]])[0] == "b"


def test_single_list_keeps_order():
    assert rrf([["a", "b", "c", "d"]]) == ["a", "b", "c", "d"]


def test_top_cuts_the_result():
    assert rrf([["a", "b", "c"], ["d", "e", "f"]], top=2) == ["a", "d"]
    assert len(rrf([["a", "b", "c"]], top=10)) == 3


def test_two_lists_worked_example():
    # x: 1/61 + 1/63, y: 2/62, and 1/61 + 1/63 > 2/62
    assert rrf([["x", "y", "z"], ["w", "y", "x"]])[:2] == ["x", "y"]
