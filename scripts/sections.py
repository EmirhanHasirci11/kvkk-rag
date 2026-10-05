"""Article / section labels for word chunks.

Laws, regulations and the communiqué have "MADDE n -" headings, labelled "Madde n"
("Geçici Madde n" when preceded by "GEÇİCİ"). The two guides have numbered headings
at the start of a line ("3.1. ..." or "3) ..."), labelled with the bare number ("3.1"),
the same form eval/golden.jsonl uses in its source field.

A chunk that starts mid-article carries the last label seen before its first word.
Text before the first heading has no label (None).
"""
import re

from chunking import chunk_words

ARTICLE_RE = re.compile(r"\b(GEÇİCİ\s+)?MADDE\s+(\d+)\s*[-–]")
SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]\s+\S")


def section_marks(text: str) -> list[tuple[int, str]]:
    """(index of the heading's first word in text.split(), label), in text order."""
    articles, sections = [], []
    offset = 0
    prev_line = ""
    for line in text.splitlines():
        for m in ARTICLE_RE.finditer(line):
            # "GEÇİCİ" sometimes sits alone on the line before "MADDE n -"
            temporary = bool(m.group(1)) or prev_line.strip() == "GEÇİCİ"
            label = ("Geçici Madde " if temporary else "Madde ") + m.group(2)
            articles.append((offset + len(line[:m.start()].split()), label))
        m = SECTION_RE.match(line)
        if m:
            sections.append((offset + len(line[:m.start(1)].split()), m.group(1)))
        offset += len(line.split())
        if line.strip():
            prev_line = line
    if offset != len(text.split()):
        raise ValueError("word offsets do not match text.split()")
    # documents with articles are laws; numbered lines in them are not headings
    return articles if articles else sections


def chunk_sections(text: str, size: int, overlap: int = 0) -> list[dict]:
    """Labels for each chunk of chunk_words(text, size, overlap), in the same order.

    section:  label in force at the chunk's first word (carried forward), or None
    sections: every label the chunk covers, starting with `section`
    """
    marks = section_marks(text)
    n_words = len(text.split())
    n_chunks = len(chunk_words(text, size, overlap))
    step = size - overlap
    out = []
    for c in range(n_chunks):
        start, end = c * step, min(c * step + size, n_words)
        current = None
        for idx, label in marks:
            if idx > start:
                break
            current = label
        inside = [label for idx, label in marks if start < idx < end]
        covered = ([current] if current else []) + inside
        out.append({"section": current, "sections": list(dict.fromkeys(covered))})
    return out
