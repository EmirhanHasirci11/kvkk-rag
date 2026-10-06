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


# Prompt v1

## Setup

In v1, 5 of the 15 "bilgi yok" answers had the evidence in the top-5. Most of them asked about a concrete case the texts don't name (".env", "by phone", "staff ID photos") while a general rule in the chunks covers it, and the v0 prompt ("answer only from the texts, no interpretation") made the model abstain.

The v1 prompt (`PROMPTS["v1"]` in `scripts/answer.py`) adds: if a general rule in the texts covers the concrete case, quote it with its citation and say that the case itself is not regulated separately, adding no conclusion, number or period beyond the rule; answer the part of a question the texts cover; say "bilgi yok" only when the texts have no related rule, definition or explanation.

Retrieval is the same as v1 (`fuse`, same cached rewrites; the retrieved chunks are identical for every question), so only the prompt changes.

Code: `python scripts/run_answers.py --rewrite --rewrites results/rewrites_v1.jsonl --prompt v1 --tag v2`.

## Results

| | v1 (prompt v0) | v2 (prompt v1) |
|---|---|---|
| Golden, correct (user graded) | 35 / 50 | 43 / 50 |
| Golden, "bilgi yok" | 15 | 6 |
| Golden, citation check pass | 30 | 35 |
| Abstention, correct (user graded) | 10 / 10 | 10 / 10 |
| Abstention, plain "bilgi yok" | 10 | 4 |

v1 -> v2: q008, q012, q013, q019, q020, q026, q039 and q044 became correct; no question went from correct to wrong. All 6 remaining "bilgi yok" answers are retrieval misses (q004, q005, q007, q014, q024, q029). q050 is answered but wrong: it lists general rules on special-category data and misses the guide's conclusion.

On the abstention set, 6 answers now quote a related rule instead of a plain "bilgi yok" (a001: Madde 12 "en kısa sürede"; a007: the password advice; a008: the guide's criteria for retention periods), each saying the asked number or case is not in the texts. None invents a number, deadline or threshold; every quoted rule was checked against the corpus.

## Findings

- The prompt fixed the abstentions it targeted: with the same chunks, correct answers go from 35 to 43 and the remaining failures are all retrieval misses plus one weak answer.
- Being less conservative did not lead to invented answers on the abstention set.
- Side effects: 10 answers that were already correct now add an unneeded "bu somut durum ayrıca düzenlenmemiştir" line, and q050 cites `[rehber_ozel_nitelikli, başlık öncesi]`. "başlık öncesi" is the chunk header's label for text before the first heading, not a real section; the header wording should change in the next prompt version.

## Limitations

- The v1 prompt was written after reading the failures on these same 50 questions, so 43 / 50 is an optimistic number. The held-out check is step 6.
- The grading counts an answer as correct when it answers from a valid cited rule, even if it is not the gold passage (q026, q044). A stricter grader would score some of these as partial.


# Reranker

## Setup

Fixed before the first run (docstring of `scripts/eval_rerank.py`):

- **Candidates:** top-30 of `orig` (question alone) and of `fuse` (question + the cached rewrite from step 3).
- **Rerankers:** `BAAI/bge-reranker-v2-m3` (0.6B, multilingual) and, as a small baseline, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (0.1B, trained on mMARCO, which has no Turkish). Both score (original question, chunk) pairs with max length 512, so a 192-word chunk is not cut.
- **Adoption rule:** a reranker goes into the pipeline only if it raises R@5 on all 50 questions by at least 0.06 (3 questions). R@5 is what the answer model sees.
- **Check:** the no-rerank rows reproduce `results/rewrite_v1_per_question.csv` for every question.

Code: `scripts/reranker.py`, `scripts/eval_rerank.py`. Raw numbers in `results/rerank_v1_*`.

## Results (50 q)

| System | R@1 | R@3 | R@5 | R@10 | R@30 (ceiling) |
|---|---|---|---|---|---|
| orig | 0.38 | 0.56 | 0.64 | 0.73 | 0.87 |
| orig + mmarco | 0.42 | 0.62 | 0.70 | 0.74 | 0.87 |
| orig + bge | 0.50 | 0.69 | 0.81 | 0.83 | 0.87 |
| fuse | 0.47 | 0.67 | 0.71 | 0.81 | 0.96 |
| fuse + mmarco | 0.44 | 0.66 | 0.76 | 0.78 | 0.96 |
| fuse + bge | 0.52 | 0.73 | 0.80 | 0.88 | 0.96 |

At R@5, `orig + bge` gains 10 questions and loses 1 (q020); `fuse + bge` gains 7 and loses 2 (q017, q020).

## Findings

- `bge-reranker-v2-m3` passes the adoption rule on both candidate sets (+0.17 on `orig`, +0.09 on `fuse`). The small mMARCO model does not on `fuse` (+0.05) and loses questions the hybrid already had (q006, q042).
- With the reranker, the LLM rewrite adds little at the top: `orig + bge` and `fuse + bge` are tied at R@5 (0.81 and 0.80). The rewrite still widens the candidate pool (R@30 0.87 to 0.96), which shows at R@10 (0.83 against 0.88).
- `orig + bge` reaches 0.81 at R@5 without any LLM call, against 0.71 for `fuse`, the step 3 system.
- Speed: bge took 1012 s for 50 questions on CPU (about 20 s per question for ~40 distinct candidates) and 33 s on an RTX 3080 (about 0.7 s per question); mMARCO took 96 s and 5 s. The GPU rerun gave exactly the same recall for every question, and the hybrid retrieval also reproduces exactly on GPU (same top-5 chunks as the CPU runs for all 50 questions).

## Answer results

`fuse` top-30 -> bge rerank -> top-5 -> prompt v1 (`--rerank --prompt v1 --tag v3`). Everything except the reranker is the same as v2, and the top-5 of every question matches the `fuse + bge` row above.

| | v2 (no rerank) | v3 (rerank) |
|---|---|---|
| Golden, correct (user graded) | 43 / 50 | 47 / 50 |
| Golden, "bilgi yok" | 6 | 1 |
| Golden, citation check pass | 35 | 40 |
| Abstention, correct (user graded) | 10 / 10 | 10 / 10 |

v2 -> v3: q004, q005, q024 and q050 became correct; no question went from correct to wrong. q004 ("Sitemiz hacklendi..."), missed by every system since the start, is answered from Madde 12. The 3 wrong answers: q029 is a retrieval miss ("bilgi yok"); q007 and q014 list general rules and miss the rule that answers them (aydınlatma and açık rıza taken separately; the Madde 28 exemption for personal and household use). On the abstention set, 5 answers are a plain "bilgi yok" and 5 quote a related rule while saying the asked number or case is not in the texts; nothing is invented.

## Progress (golden set, user graded)

| Version | Retrieval | Prompt | Correct |
|---|---|---|---|
| v0 | orig | v0 | 33 / 50 |
| v1 | fuse (rewrite) | v0 | 35 / 50 |
| v2 | fuse | v1 | 43 / 50 |
| v3 | fuse + bge rerank | v1 | 47 / 50 |
| v4 | fuse + bge rerank, bm25-snow | v1 | 49 / 50 |

Abstention set: 10 / 10 in every version. v4 is described under Stemming below.

## Limitations

- The setup and the adoption rule were fixed before the run and nothing was tuned, but `fuse` was chosen and the v1 prompt was written on these same 50 questions. 47 / 50 is not a held-out number; step 6 is.
- The grading counts an answer as correct when it answers from a valid cited rule, even if it is not the gold passage.


# Stemming

## Setup

Fixed before the first run (docstring of `scripts/eval_stemming.py`):

- **Stemmers:** `bm25-none` (no stemming), `bm25-p5` (first 5 letters, the `bm25-prefix5` used so far), `bm25-snow` (Turkish Snowball, `snowballstemmer`), `bm25-zeyrek` (lemma from `zeyrek`, a Python port of Zemberek; it has no disambiguation, so the lemma of its first analysis is used and unknown words stay as they are).
- **Levels:** BM25 alone; the `orig` hybrid (e5 + BM25); the pipeline, `fuse` top-30 -> bge rerank.
- **Adoption rule:** a stemmer replaces p5 only if it raises the pipeline R@5 by at least 0.06 (3 questions).
- **Check:** the p5 rows reproduce `results/rerank_v1_per_question.csv`.

Code: `scripts/textproc.py`, `scripts/eval_stemming.py`. Raw numbers in `results/stemming_v1_*`.

## Results (50 q)

| Level | Stemmer | R@1 | R@3 | R@5 | R@10 | R@30 |
|---|---|---|---|---|---|---|
| BM25 alone | bm25-none | 0.18 | 0.26 | 0.30 | 0.42 | 0.64 |
| | bm25-p5 | 0.28 | 0.46 | 0.48 | 0.60 | 0.81 |
| | bm25-snow | 0.30 | 0.42 | 0.52 | 0.60 | 0.77 |
| | bm25-zeyrek | 0.28 | 0.44 | 0.52 | 0.62 | 0.75 |
| Hybrid (orig) | bm25-none | 0.18 | 0.36 | 0.48 | 0.63 | 0.90 |
| | bm25-p5 | 0.38 | 0.56 | 0.64 | 0.73 | 0.87 |
| | bm25-snow | 0.36 | 0.56 | 0.65 | 0.71 | 0.87 |
| | bm25-zeyrek | 0.40 | 0.60 | 0.66 | 0.73 | 0.87 |
| Pipeline (fuse + bge) | bm25-none | 0.54 | 0.72 | 0.82 | 0.88 | 0.94 |
| | bm25-p5 | 0.52 | 0.73 | 0.80 | 0.88 | 0.96 |
| | bm25-snow | 0.52 | 0.72 | 0.86 | 0.90 | 0.96 |
| | bm25-zeyrek | 0.54 | 0.73 | 0.84 | 0.88 | 0.94 |

R@30 is the candidate pool before reranking (for the pipeline, the `fuse` top-30).

Pipeline at R@5: `bm25-snow` gains q014, q017 and q019 and loses none (+0.06, exactly at the threshold), so it replaces p5. `bm25-zeyrek` gains 2, `bm25-none` 1.

## Answer results

`--rerank --stem snow --prompt v1 --tag v4`, everything else as v3; the top-5 of every question matches the `pipe-snow` row.

| | v3 (bm25-p5) | v4 (bm25-snow) |
|---|---|---|
| Golden, correct (user graded) | 47 / 50 | 49 / 50 |
| Golden, citation check pass | 40 | 43 |
| Abstention, correct (user graded) | 10 / 10 | 10 / 10 |

v3 -> v4: q014 now quotes the Madde 28 exemption for personal and household use; q029 became correct; nothing went from correct to wrong. q007 is still wrong (now "bilgi yok"). The q029 answer lists general rules (use only for the stated purpose, keep the notice up to date) and not the gold rule (a new purpose needs a new notice); answers of that kind were graded wrong in v3 (q007, q014), so with the v3 standard v4 is 48 / 50.

## Findings

- Stemming matters a lot for BM25 alone (R@5 0.30 without, about 0.50 with any stemmer), less in the hybrid, and little once the reranker sees the candidates: without any stemming the pipeline still reaches R@5 0.82.
- Snowball and zeyrek beat the 5-letter cut slightly at the BM25 and hybrid levels, but the differences are 1-3 questions.
- `bm25-snow` passed the adoption rule exactly at the threshold; in the answers it adds one or two questions.

## Limitations

- The gain is at the threshold of what this set can show, and the questions are not held out.
- zeyrek's first analysis is often not the right lemma ("Kurula" -> "kurulamak"); a disambiguation step might change its numbers.


# Final held-out test

## Setup

- **Questions:** `eval/test_v2.jsonl`, 20 new questions (q051-q070), and `eval/abstention_v2.jsonl`, 5 new out-of-corpus questions (a011-a015). Both were committed before the run (`53f42bb`, `b364e42`).
- **How they were made:** evidence sentences were drawn at random (seed 2026) from sections that no golden question uses; fragments, headings, case law, Kurum organisation articles and second sentences from an already used section were skipped by a rule fixed before drawing. Claude drafted one everyday-language question per sentence, and the user rewrote every question in their own words (typos kept). The abstention questions are Claude's drafts.
- **Frozen pipeline** (`scripts/run_heldout.py`): final = v4 (`fuse` top-30 -> bge rerank -> top-5, `bm25-snow`, prompt v1); baseline = v0 (question alone, hybrid top-5, `bm25-p5`, prompt v0). One run; the script refuses a second run with the same tag and records the commit in `results/heldout_v2_config.json`.
- **Grading:** by the user, same rule as before.

## Retrieval (20 q)

From `results/heldout_v2_retrieval_summary.csv`.

| System | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| orig | 0.15 | 0.30 | 0.35 | 0.65 |
| rw-only | 0.50 | 0.65 | 0.70 | 0.75 |
| fuse | 0.20 | 0.55 | 0.60 | 0.70 |
| orig + bge | 0.30 | 0.50 | 0.55 | 0.70 |
| fuse + bge | 0.30 | 0.50 | 0.55 | 0.70 |
| fuse + bge + snow (final) | 0.30 | 0.50 | 0.55 | 0.75 |

## Answers

| | v0 (baseline) | final (v4) |
|---|---|---|
| Held-out, correct (user graded) | 9 / 20 | 12 / 20 |
| Held-out, "bilgi yok" | 10 | 3 |
| Held-out abstention, correct | 5 / 5 | 5 / 5 |
| Golden set (seen), correct | 33 / 50 | 49 / 50 |

v0 -> final: q051, q054, q055, q057 and q059 became correct; q052 and q068 went from correct to wrong.

In the final system, all 11 questions whose evidence was in the top-5 were answered correctly. All 8 wrong answers had no evidence in the top-5: 3 say "bilgi yok" (q056, q064, q069) and 5 answer from general rules and miss the rule that answers them (q052, q058, q062, q065, q068). q054 is correct without its evidence in the top-5.

On the 5 new abstention questions the final system says "bilgi yok" once and quotes a related general rule 4 times, each time saying the asked case is not in the texts; nothing is invented.

## Findings

- The 50-question numbers were optimistic. Final accuracy drops from 49 / 50 on the questions used to build the system to 12 / 20 on new ones; the baseline drops less (33 / 50 to 9 / 20).
- The final system still beats the baseline by 3 questions (+0.15), with 2 questions lost. With n = 20 that is a modest gain, not a decisive one.
- Generation is not the problem on new questions: when the evidence is retrieved, the answer is right (11 of 11). Retrieval is.
- The reranker did not carry over: it lifted R@5 from 0.71 to 0.80 on the golden set but lowers it from 0.60 to 0.55 here. The LLM rewrite did carry over: `rw-only` has the best R@5 (0.70, against 0.35 for the question alone).
- No invented answers on any abstention question, seen or new.

## Limitations

- n = 20: one question is worth 0.05.
- The questions are evidence-first (sampled sentence, then question), drafted by Claude and rewritten by the user, not written cold by a user.
- 12 of the 20 evidence sentences come from the special-category guide, because most unused sentences are there; the communiqué has none.
- By the project rule, nothing is changed after this result. `rw-only` looking better here is a finding for the next version, which would need its own new held-out set.

Roadmap: see docs/ROADMAP.md
