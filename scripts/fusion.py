"""Rank fusion for hybrid retrieval.

Reciprocal Rank Fusion (RRF):

    score(d) = sum over ranked lists L containing d of  1 / (k + rank_L(d))      (rank starts at 1)

k = 60 is the constant proposed in Cormack, Clarke and Buettcher, "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009). It damps the
influence of the very top ranks, so one list cannot dominate on its own.
Only ranks are used, never raw scores, so BM25 and cosine scores need no normalization.
"""


def rrf(ranked_lists, k=60, top=10):
    """Fuse several ranked lists of hashable items into one list of at most `top` items."""
    scores = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # dicts keep insertion order and sorted() is stable, so ties go to the item seen first
    fused = sorted(scores, key=lambda item: -scores[item])
    return fused[:top]
