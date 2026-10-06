"""Chunk embeddings in Postgres with pgvector (ROADMAP step 7).

Start the database with `docker compose up -d` from the repo root. DATABASE_URL overrides the
default connection string. The table is rebuilt only when the chunk texts or the embedding model
change (a hash is kept in `meta`), so a restart does not re-embed the corpus.

With ~150 chunks an exact scan is fast; the search orders by cosine distance and then by id,
so ties are broken the same way every time.
"""
import hashlib
import json
import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DEFAULT_DSN = "postgresql://kvkk:kvkk@localhost:5432/kvkk"


class PgVectorIndex:
    def __init__(self, dsn=None, dim=768):
        self.conn = psycopg.connect(dsn or os.environ.get("DATABASE_URL", DEFAULT_DSN), autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute("CREATE TABLE IF NOT EXISTS chunks (id integer PRIMARY KEY, doc text, section text, "
                          f"sections text[], text text, embedding vector({dim}))")
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, value text)")

    def sync(self, chunks: list[dict], encode, model_name: str) -> bool:
        """Store the chunks and their embeddings unless they are already there. True if it re-embedded."""
        digest = hashlib.sha256((model_name + json.dumps([c["text"] for c in chunks], ensure_ascii=False))
                                .encode("utf-8")).hexdigest()
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'corpus_hash'").fetchone()
        if row and row[0] == digest:
            return False
        vecs = encode()
        with self.conn.transaction():
            self.conn.execute("TRUNCATE chunks")
            with self.conn.cursor() as cur:
                cur.executemany("INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s)",
                                [(i, c["doc"], c["section"], c["sections"], c["text"], np.asarray(v, dtype=np.float32))
                                 for i, (c, v) in enumerate(zip(chunks, vecs, strict=True))])
            self.conn.execute("INSERT INTO meta VALUES ('corpus_hash', %s) "
                              "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (digest,))
        return True

    def search(self, qvec, k: int) -> list[int]:
        """Chunk ids by cosine distance (<=>), closest first."""
        rows = self.conn.execute("SELECT id FROM chunks ORDER BY embedding <=> %s, id LIMIT %s",
                                 (np.asarray(qvec, dtype=np.float32), k)).fetchall()
        return [r[0] for r in rows]
