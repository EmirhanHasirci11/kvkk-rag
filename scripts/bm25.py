"""Okapi BM25, written from scratch.

    score(q, d) = sum over query terms t of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))      # always positive
"""
import math
from collections import Counter


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """docs: list of token lists."""
        self.k1 = k1
        self.b = b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(self.len) / self.N if self.N else 0.0
        self.df = Counter(t for tf in self.tf for t in tf)

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query, i):
        tf, dl = self.tf[i], self.len[i]
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        total = 0.0
        for t in query:
            f = tf.get(t, 0)
            if f:
                total += self.idf(t) * f * (self.k1 + 1) / (f + norm)
        return total

    def search(self, query, k=10):
        """Top k (doc index, score) pairs, best first. Ties keep corpus order."""
        scored = [(i, self.score(query, i)) for i in range(self.N)]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
