"""
brain/arc/ingest_arc.py
-----------------------
Step 3 of the ARC evaluation track: index the focused ARC corpus subset into a
SEPARATE Qdrant collection (`arc_corpus`).

A separate collection keeps the ARC science sentences fully isolated from the
`academic_papers` collection used by the main RAG system — the two evaluations
never share an index.

Schema mirrors the main indexer: named dense vector (BGE-M3, 1024-d, cosine) +
named sparse vector (BM25, IDF). Each sentence is one point.

Usage (from brain/):
    python arc/ingest_arc.py
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ARC_DIR = Path(__file__).resolve().parent
_DATA = _ARC_DIR / "data"
_BRAIN_ROOT = _ARC_DIR.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

try:
    from dotenv import load_dotenv
    for _ep in (_BRAIN_ROOT / ".env", _BRAIN_ROOT.parent / ".env"):
        if _ep.exists():
            load_dotenv(_ep)
            break
except ImportError:
    pass

import os
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from FlagEmbedding import BGEM3FlagModel

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
ARC_COLLECTION = "arc_corpus"
DENSE_MODEL = "BAAI/bge-m3"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_SIZE = 1024
BATCH = 64


def main() -> int:
    subset = _DATA / "arc_corpus_subset.jsonl"
    if not subset.exists():
        print(f"[arc-ingest] corpus subset not found: {subset}")
        return 1

    # Iterate physical lines (split only on \n). Do NOT use splitlines(): some
    # ARC Corpus sentences contain Unicode line separators (  /  )
    # which splitlines() would treat as line breaks, splitting JSON records.
    with open(subset, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    texts = [r["text"] for r in rows]
    print(f"[arc-ingest] {len(texts)} sentences to index into '{ARC_COLLECTION}'")

    client = QdrantClient(url=QDRANT_URL)

    # Recreate the ARC collection fresh each run (it is small and disposable).
    if client.collection_exists(ARC_COLLECTION):
        client.delete_collection(ARC_COLLECTION)
    client.create_collection(
        collection_name=ARC_COLLECTION,
        vectors_config={"dense": models.VectorParams(
            size=DENSE_SIZE, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF)},
    )
    print(f"[arc-ingest] created collection '{ARC_COLLECTION}'")

    print(f"[arc-ingest] loading embedding models...")
    dense_model = BGEM3FlagModel(DENSE_MODEL, use_fp16=False)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    print(f"[arc-ingest] embedding {len(texts)} sentences...")
    dense_out = dense_model.encode(
        texts, batch_size=8, max_length=512,
        return_dense=True, return_sparse=False, return_colbert_vecs=False,
    )["dense_vecs"]
    sparse_out = list(sparse_model.embed(texts))

    points = []
    for i, (row, dvec, svec) in enumerate(zip(rows, dense_out, sparse_out)):
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"arc_{i}")),
            payload={"text": row["text"],
                     "matched_question_ids": row.get("matched_question_ids", [])},
            vector={
                "dense": dvec.tolist() if hasattr(dvec, "tolist") else list(dvec),
                "sparse": models.SparseVector(
                    indices=svec.indices.tolist(),
                    values=svec.values.tolist()),
            },
        ))

    for start in range(0, len(points), BATCH):
        client.upsert(collection_name=ARC_COLLECTION,
                      points=points[start:start + BATCH], wait=True)
        print(f"  upserted {min(start + BATCH, len(points))}/{len(points)}")

    info = client.get_collection(ARC_COLLECTION)
    print(f"[arc-ingest] done. '{ARC_COLLECTION}' now holds {info.points_count} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
