# Corpus

Official Turkish data protection texts used for retrieval experiments. About 22,000 words from 150 PDF pages.

| File | Document | Source |
|---|---|---|
| `kvkk_kanun_6698.txt` | 6698 sayılı Kişisel Verilerin Korunması Kanunu (consolidated, includes the 2024 amendments) | mevzuat.gov.tr |
| `yonetmelik_silme_yok_etme_anonim.txt` | Kişisel Verilerin Silinmesi, Yok Edilmesi veya Anonim Hale Getirilmesi Hakkında Yönetmelik | mevzuat.gov.tr |
| `teblig_aydinlatma.txt` | Aydınlatma Yükümlülüğünün Yerine Getirilmesinde Uyulacak Usul ve Esaslar Hakkında Tebliğ | mevzuat.gov.tr |
| `rehber_veri_guvenligi.txt` | Kişisel Veri Güvenliği Rehberi (Teknik ve İdari Tedbirler) | kvkk.gov.tr |
| `rehber_ozel_nitelikli.txt` | Özel Nitelikli Kişisel Verilerin İşlenmesine İlişkin Rehber | kvkk.gov.tr |

Word counts, page counts and source URLs are in `manifest.json`.

## How the text was cleaned

`scripts/extract_corpus.py` rebuilds everything in `text/` from the PDFs in `raw/`.

- Text is extracted with PyMuPDF, block by block.
- Page numbers, running headers, footnotes and footnote markers are removed using font size.
- Words hyphenated across line breaks are joined, and paragraphs split across pages are merged.
- Covers, tables of contents, the summary tables and bibliography of the security guide, and the amendment and staffing tables at the end of the legal texts are removed.
- Legal texts are restructured so every article starts on its own line with its title, and every clause `(1)` and item `a)` starts on a new line.
- Typographic ligatures are replaced with plain letters. Turkish characters and quotation marks are kept as they are.

## Known gaps

- Footnote markers are removed, so a few sentences in the guides read slightly differently from the PDF.
- Not included yet: Veri Sorumlusuna Başvuru Usul ve Esasları Hakkında Tebliğ, Aydınlatma Yükümlülüğünün Yerine Getirilmesi Rehberi.


# Retrieval results

## Setup

- **Corpus:** the 5 documents above (21,741 words, 150 PDF pages).
- **Questions:** `eval/golden.jsonl` holds 50 questions. Each has one or more verbatim evidence sentences from the corpus (with optional alternatives), a source, and a type: paraphrase (21), senaryo / scenario (22), dogrudan / direct (4) or coklu / multi-fact (3). Questions q001-q030 are the dev set. q031-q050 were written later as a held-out test set (kept unchanged in `eval/test_v1.jsonl`) and were added to `golden.jsonl` after the test run.
- **Metric:** Recall@k is the share of evidence facts that appear verbatim (whitespace-normalized) in at least one of the top-k chunks. Multi-fact questions score a fraction.
- **Chunking:** fixed-size word chunks, cut per document. 192 words with 48 overlap was the best of the four configurations tried (64/0, 64/32, 128/32, 192/48) on R@3, R@5 and R@10, but not on R@1.
- **Models:** `intfloat/multilingual-e5-base` (`query: ` and `passage: ` prefixes) and `trmteb/turkish-embedding-model`, cosine similarity.
- **BM25:** written from scratch in `scripts/bm25.py` (k1 = 1.5, b = 0.75). Tokens are Turkish-lowercased words. `bm25-none` uses them as they are; `bm25-prefix5` cuts each word to its first 5 letters as a crude stemmer.
- **Hybrid:** Reciprocal Rank Fusion (k = 60). Each retriever passes its top 50 candidates, the fused list is cut to 10. In all hybrid runs, the BM25 side is bm25-prefix5 (first 5 letters stemming). bm25-none is never used in a hybrid.

Code: `scripts/retriever.py` (`HybridRetriever`), `scripts/eval_retriever.py`, notebooks `01` (dev experiments) and `02` (held-out test). Raw numbers are in `results/`.

## Dev results (30 questions, 192/48)

From `results/v4_summary.csv`.

| System | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| e5 | 0.22 | 0.48 | 0.52 | 0.72 |
| tr | 0.17 | 0.43 | 0.53 | 0.65 |
| bm25-none | 0.07 | 0.13 | 0.17 | 0.30 |
| bm25-prefix5 | 0.17 | 0.40 | 0.40 | 0.50 |
| hyb-e5+bm25 | 0.33 | 0.50 | 0.57 | 0.65 |
| hyb-tr+bm25 | 0.30 | 0.47 | 0.57 | 0.67 |
| hyb-e5+tr+bm25 | 0.33 | 0.57 | 0.62 | 0.72 |

## Held-out test results (20 questions, 192/48)

From `results/test_v1_summary.csv`. Configuration frozen before the run, one run.

| System | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| e5 | 0.45 | 0.50 | 0.70 | 0.75 |
| hyb-e5+bm25 | 0.45 | 0.65 | 0.75 | 0.85 |
| hyb-e5+tr+bm25 | 0.45 | 0.55 | 0.60 | 0.75 |

## Combined results (dev + test combined, 50 q, 192/48)

From `results/combined_50q.csv`, written by `scripts/eval_combined.py`. BM25 is computed on all 50 questions. The e5 and hybrid rows are the mean of the per-question values from `results/test_v1_per_question.csv` (30 dev + 20 test).

| System | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| bm25-none | 0.18 | 0.26 | 0.30 | 0.42 |
| bm25-prefix5 | 0.28 | 0.46 | 0.48 | 0.60 |
| e5 | 0.31 | 0.49 | 0.59 | 0.73 |
| hyb-e5+bm25 | 0.38 | 0.56 | 0.64 | 0.73 |

BM25 R@10 is 0.42 without stemming and 0.60 with prefix5. `e5` and `hyb-e5+bm25` are tied at R@10 (0.73); the hybrid is ahead at R@1, R@3 and R@5. These numbers include the test questions that were also looked at when choosing the system (see Decision), so they are a summary, not an independent test.

## Findings

- Stemming matters for BM25: `bm25-none` to `bm25-prefix5` raises R@10 from 0.30 to 0.50.
- On dev, hybrid retrieval improves the top ranks (R@1 0.22 to 0.33) but not R@10 (0.65 and 0.72 against 0.72 for e5 alone).
- The R@1 gain did not repeat on test: all three systems score 0.45.
- The three-way hybrid (e5 + tr + BM25) did not hold up on test: R@10 0.75 and R@5 0.60, against 0.85 and 0.75 for `hyb-e5+bm25`.
- q007 and q014 are missed by every system on dev.

## Decision

`hyb-e5+bm25`. On dev the two hybrids differ by about two questions (R@3 0.50 against 0.57, R@10 0.65 against 0.72), which is within noise at n = 30, so the cheaper one was chosen: it needs one embedding model instead of two. The test result points the same way (R@10 0.85 against 0.75), but the choice was not made strictly before looking at the test results, so the test set is not a fully independent confirmation.

## Limitations

- Small n. One question is worth 0.03 on dev and 0.05 on test. Differences under about 0.10 should not be read as real.
- One person wrote all questions.
- The test questions were written evidence-first, and every system scores higher on test than on dev. Absolute numbers are not comparable across the two sets.
- Verbatim-evidence recall checks that the labeled sentence is in the top-k chunks. It does not measure whether an answer would be correct, and other passages that support an answer are not counted.
- Concept gaps are not solved. q007 and q014 describe a situation without using the legal terms, and neither BM25 nor the embeddings bridge that.

Roadmap: see docs/ROADMAP.md
