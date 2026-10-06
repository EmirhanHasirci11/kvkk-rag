"""Hybrid retriever: e5 dense search and BM25 (prefix5 stemmer), fused with Reciprocal Rank Fusion.

Same pipeline as the hyb-e5+bm25 system in notebooks 01 and 02.
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

from bm25 import BM25
from chunking import chunk_words
from fusion import rrf
from sections import chunk_sections
from textproc import tokenize

MODEL_NAME = "intfloat/multilingual-e5-base"
STEM = "prefix5"


class HybridRetriever:
    def __init__(self, texts: dict[str, str], size=192, overlap=48, depth=50, k_rrf=60, stem=STEM,
                 dense="numpy", dsn=None):
        """texts maps document name to full text. Documents are chunked separately, in dict order.
        stem: BM25 stemmer, a key of textproc.STEMMERS.
        dense: "numpy" keeps the chunk vectors in memory; "pgvector" keeps them in Postgres (scripts/pgstore.py)."""
        self.depth = depth
        self.k_rrf = k_rrf
        self.stem = stem
        # chunk each document separately so no chunk mixes two documents
        self.chunks = [{"doc": doc, "text": c, **meta}
                       for doc, t in texts.items()
                       for c, meta in zip(chunk_words(t, size, overlap), chunk_sections(t, size, overlap), strict=True)]
        self.model = SentenceTransformer(MODEL_NAME)
        self.index, self.vecs = None, None
        if dense == "pgvector":
            from pgstore import PgVectorIndex
            self.index = PgVectorIndex(dsn)
            self.index.sync(self.chunks, self._encode_chunks, MODEL_NAME)
        else:
            self.vecs = self._encode_chunks()
        self.bm25 = BM25([tokenize(c["text"], self.stem) for c in self.chunks])

    def _encode_chunks(self):
        return self.model.encode(["passage: " + c["text"] for c in self.chunks],
                                 normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    def ranked_lists(self, query: str) -> list[list[int]]:
        """Dense and BM25 rankings of the top `depth` chunk indices for one query, best first."""
        q = self.model.encode(["query: " + query], normalize_embeddings=True, show_progress_bar=False)[0]
        if self.index is not None:
            dense = self.index.search(q, self.depth)
        else:
            dense = np.argsort(-(self.vecs @ q))[:self.depth].tolist()    # cosine similarity, vectors are normalized
        sparse = [i for i, _ in self.bm25.search(tokenize(query, self.stem), k=self.depth)]
        return [dense, sparse]

    def search(self, query: str, k=5) -> list[dict]:
        return self.search_multi([query], k)

    def search_multi(self, queries: list[str], k=5) -> list[dict]:
        """RRF over the dense and BM25 rankings of every query, e.g. a question and its rewrite."""
        fused = rrf([ranked for q in queries for ranked in self.ranked_lists(q)], k=self.k_rrf, top=k)
        # section: article / guide section at the chunk's first word, sections: all it covers
        return [{"doc": self.chunks[i]["doc"], "section": self.chunks[i]["section"],
                 "sections": self.chunks[i]["sections"], "text": self.chunks[i]["text"], "rank": r}
                for r, i in enumerate(fused, 1)]
