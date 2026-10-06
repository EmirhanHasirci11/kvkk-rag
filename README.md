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


# Generation results

## Setup

- **Pipeline:** `hyb-e5+bm25` top-5 chunks (192/48) -> `gemini-3.8-flash` (google-genai SDK, temperature 0) -> short Turkish answer with a citation after every claim. Prompt rules: answer only from the given chunks, cite as `[doc, Madde n]` (guides: `[doc, 3.1]`), say "Bu konuda verilen metinlerde bilgi yok." when the chunks don't answer it, quote the text and give no legal advice.
- **Chunk metadata:** every chunk carries its doc id, the article (`Madde n`, `Geçici Madde n`) or guide section (`3.1`) in force at its first word, and every label it covers (`scripts/sections.py`). Adding it did not change retrieval: dev R@1/3/5/10 reproduced as 0.33 / 0.50 / 0.57 / 0.65.
- **Grading:** the user graded every answer as correct / partial / wrong. An answer is correct when it answers the question from the given chunks with a valid citation, also when it cites a passage other than the gold evidence that says the same thing. "Bilgi yok" on a golden question is wrong.
- **Citation check (automatic):** pass when every gold fact is contained in a chunk the answer cites.
- **Abstention set:** `eval/abstention.jsonl`, 10 questions whose answer is not in the corpus (cookie banners, VERBİS threshold, a 72-hour breach deadline, GDPR fines, ...). Expected: "bilgi yok", or quoting the related text without inventing the missing number.
- **Cost:** about $0.004 per question. Prices and the $8 budget cap are in `scripts/llm.py`, every call is logged in `results/llm_usage.csv`.

Code: `scripts/llm.py`, `scripts/answer.py` (`python scripts/answer.py "soru"`), `scripts/run_answers.py`. Answers and grades are in `results/answers_v0*` and `results/answers_abstention_v0*`.

## Results

Golden set (50 q), `answers_v0`:

| Grade | Count | Share |
|---|---|---|
| correct | 33 | 0.66 |
| partial | 0 | 0.00 |
| wrong | 17 | 0.34 |

All 17 wrong answers are "bilgi yok". In 14 of them the evidence was not in the top-5 chunks; in 3 (q008, q020, q039) it was, and the model still abstained. No answered question was graded wrong.

Citation check against the grades:

| citation_check | Count | Graded correct |
|---|---|---|
| pass | 29 | 29 |
| fail | 4 | 4 |
| no_citation | 17 | 0 |

The 4 fails cite a different passage that states the same rule (for example the guide quoting the law's definition), so the citation check gives 0.58 where the graded accuracy is 0.66.

Abstention set (10 q), `answers_abstention_v0`: 10 of 10 answered "bilgi yok" and none invented a number or a rule. In a001 and a007 the related text (Madde 12 "en kısa sürede", the password advice in the security guide) was in the top-5 but was not quoted.

## Findings

- Retrieval is the bottleneck: 14 of the 17 failures are questions whose evidence was not retrieved. When the evidence was in the top-5, the model answered correctly in 29 of 32 questions.
- The model errs on the side of abstaining: no invented answers in 60 questions, but 3 unnecessary abstentions on the golden set.
- The citation check is a usable lower bound: no false passes, 4 false fails.

## Limitations

- One person wrote the questions and graded the answers.
- All 50 golden questions were seen while building the retriever, so these are not held-out numbers.
- 10 abstention questions: one question is worth 0.10.
- Temperature 0 is not fully deterministic: the same q004 call used 420 and 686 output tokens in two runs (same answer).


# Query rewriting

## Setup

- **Rewrite:** before retrieval, `gemini-3.8-flash` rewrites the question into the wording of the KVKK texts (`scripts/rewrite.py`). Rules: don't answer, add no numbers or facts, map everyday words to legal terms. The prompt's only examples are general term mappings ("müşteri" -> "ilgili kişi", "şirket" -> "veri sorumlusu"), nothing from `eval/golden.jsonl`. The rewrite is only a search query; the answer prompt still gets the original question.
- **Systems** (all `hyb-e5+bm25`, 192/48, depth 50): `orig` the question alone, `rw` the rewrite alone, `fuse` RRF over the dense and BM25 lists of both (`HybridRetriever.search_multi`).
- **Check:** `orig` reproduces the saved `hyb-e5+bm25` values in `results/test_v1_per_question.csv` for every question and every k.
- Rewrites are cached in `results/rewrites_v1.jsonl`, so the retrieval eval and the answer run use the same rewrites.

Code: `scripts/eval_rewrite.py`, `python scripts/run_answers.py --rewrite --tag v1`.

## Retrieval results (50 q)

From `results/rewrite_v1_summary.csv`.

| System | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| orig | 0.38 | 0.56 | 0.64 | 0.73 |
| rw | 0.37 | 0.67 | 0.75 | 0.78 |
| fuse | 0.47 | 0.67 | 0.71 | 0.81 |

At R@5, `rw` gains 10 questions and loses 4; `fuse` gains 5 and loses 1 (q050). `fuse` was chosen for the answer run: it loses the fewest questions and is best at R@1 and R@10. The choice was made after seeing all 50 questions.

## Answer results

| | v0 (orig) | v1 (fuse) |
|---|---|---|
| Golden, correct (user graded) | 33 / 50 | 35 / 50 |
| Golden, evidence in top-5 | 32 | 36 |
| Golden, "bilgi yok" with the evidence in top-5 | 3 | 5 |
| Abstention, "bilgi yok" (no invented answer) | 10 / 10 | 10 / 10 |

v0 -> v1: q001, q030, q031, q048, q049 became correct; q012, q019, q050 became "bilgi yok" (q050 lost its evidence; in q012 the evidence was still in the top-5).

## Findings

- Rewriting helps retrieval: `fuse` raises R@1 from 0.38 to 0.47 and R@10 from 0.73 to 0.81, with one question lost at R@5.
- The answer gain is small: +2 questions net (+5, -3), within noise at n = 50. Part of the retrieval gain is lost to the model abstaining while the evidence is in its chunks.
- Some rewrites add the model's own KVKK knowledge, not only legal wording (q013 adds "Veri Sorumluları Sicili", q031 adds the Madde 4 principle). On the abstention set this did not lead to invented answers: the rewrites turned "kaç saat?" into "yasal süre kaç saattir?" but added no number.

## Limitations

- All 50 golden questions had been seen before, and `fuse` was picked on them. The held-out check is step 6.
- Every rewrite is one extra LLM call per question (about $0.003).

Roadmap: see docs/ROADMAP.md
