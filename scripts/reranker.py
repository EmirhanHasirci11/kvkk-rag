"""Cross-encoder reranking of retrieved chunks (ROADMAP step 4)."""
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, model_name=MODEL_NAME, max_length=512):
        # chunks are 192 words, about 300-450 tokens, so 512 keeps the whole chunk
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=8,
                                                     show_progress_bar=False)]

    def rerank(self, query: str, chunks: list[dict], k=None) -> list[dict]:
        """Chunks sorted by cross-encoder score, best first; ties keep the retrieval order."""
        s = self.scores(query, [c["text"] for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: -s[i])[:k]
        return [{**chunks[i], "rank": r, "rerank_score": s[i]} for r, i in enumerate(order, 1)]
