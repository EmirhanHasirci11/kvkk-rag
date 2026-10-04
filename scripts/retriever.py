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
from textproc import tokenize

MODEL_NAME = "intfloat/multilingual-e5-base"
STEM = "prefix5"


class HybridRetriever:
    def __init__(self, texts: dict[str, str], size=192, overlap=48, depth=50, k_rrf=60):
        """texts maps document name to full text. Documents are chunked separately, in dict order."""
        self.depth = depth
        self.k_rrf = k_rrf
        # chunk each document separately so no chunk mixes two documents
        self.chunks = [{"doc": doc, "text": c} for doc, t in texts.items() for c in chunk_words(t, size, overlap)]
        self.model = SentenceTransformer(MODEL_NAME)
        self.vecs = self.model.encode(["passage: " + c["text"] for c in self.chunks],
                                      normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        self.bm25 = BM25([tokenize(c["text"], STEM) for c in self.chunks])

    def search(self, query: str, k=5) -> list[dict]:
        q = self.model.encode(["query: " + query], normalize_embeddings=True, show_progress_bar=False)[0]
        dense = np.argsort(-(self.vecs @ q))[:self.depth].tolist()    # cosine similarity, vectors are normalized
        sparse = [i for i, _ in self.bm25.search(tokenize(query, STEM), k=self.depth)]
        fused = rrf([dense, sparse], k=self.k_rrf, top=k)
        return [{"doc": self.chunks[i]["doc"], "text": self.chunks[i]["text"], "rank": r}
                for r, i in enumerate(fused, 1)]
