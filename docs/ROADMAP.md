## Done
- Corpus extraction (5 KVKK texts), golden set (50 q), check_golden
- Retrieval eval: chunking, e5 / tr embeddings, BM25 from scratch + prefix5, RRF hybrid
- Held-out test (20 q), chosen setup: hyb-e5+bm25 (BM25 side = bm25-prefix5), 192/48, depth 50
- Step 1 Generation v0: gemini-3.8-flash, chunk metadata, answer.py, run_answers.py (v0: 33/50)
- Step 2 Answer evaluation: user grading, citation check, abstention set (10 q)
- Step 3 Query rewriting: fuse (question + rewrite) (v1: 35/50); prompt v1 (v2: 43/50)
- Step 4 Reranker: bge-reranker-v2-m3 on the top-30 (v3: 47/50)
- Step 5 Stemming: bm25-snow replaces bm25-p5 (v4: 49/50)
- Step 6 Final held-out test (20 + 5 new q): final 12/20 vs baseline 9/20; abstention 5/5 both.
  Every wrong final answer is a retrieval miss; rw-only retrieval (R@5 0.70) beat the final (0.55)

## Next steps (in order)
1. Generation v0
   - Pipeline: question -> HybridRetriever top-5 -> LLM -> answer in Turkish + citations
   - Every chunk must carry metadata: doc id and article (MADDE n / section). Carry the last seen
     "MADDE n" or section number forward when a chunk starts mid-article.
   - Prompt rules: answer only from the given chunks, cite as [doc, Madde n],
     say "Bu konuda verilen metinlerde bilgi yok" when the chunks don't answer it.
   - LLM: Gemini Flash via Google AI Studio API key (prepaid, paid tier; free-tier data is used for training). Code-level budget cap: $8.
     Keep the LLM behind one small interface so it can be swapped.
   - Deliverable: scripts/answer.py (CLI: python scripts/answer.py "soru")
2. Answer evaluation
   - Citation check (automatic): does a cited chunk contain the gold evidence?
   - Answer grading: run all 50 questions, save answers to results/answers_v0.jsonl,
     user grades each as correct / partial / wrong in a CSV. LLM-as-judge only after
     comparing it against the user's grades.
   - Abstention: add ~10 out-of-scope questions (answer not in corpus), expected
     behavior is "bilgi yok". Measure how often the system makes something up.
3. Query rewriting
   - Target: vocabulary-gap questions (q004, q007, q014 type). LLM rewrites the question
     into legal wording before retrieval. Compare R@k with and without rewriting.
4. Reranker
   - Open-source cross-encoder on top-30 hybrid candidates. Compare R@1 / R@3.
5. Better Turkish stemming
   - Compare prefix5 vs snowballstemmer vs zeyrek for BM25 and in the hybrid.
   - Name variants explicitly: bm25-p5, bm25-snow, bm25-zeyrek.
6. Final held-out test
   - New 20 questions (q051+), written question-first, evidence found after.
   - Freeze the full pipeline, run once, report all systems.
7. Serving
   - FastAPI endpoint, pgvector for embeddings, Docker, latency and cost table.

## After the project (optional)
- Jev (TypeSafe) as a listwise reranker, compared with the step 4 cross-encoder.
  Needs API access (waitlist). Turkish support unverified. Public law texts only.

## Rules for every step
- Never change settings after looking at a held-out test result.
- Save each run as a new versioned file in results/, never overwrite old ones.
- Sanity check: when code moves, old numbers must reproduce exactly before new runs.
- 1 question = 2-5 points; don't claim wins smaller than that.
- .env must never be committed; check git status before every commit.
