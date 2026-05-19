"""
brain/multihop/ingest_multihop.py
---------------------------------
Index a multi-hop QA corpus subset into a dedicated Qdrant collection.

Schema mirrors the main indexer and the ARC indexer: named dense vector
(BGE-M3, 1024-d, cosine) + named sparse vector (BM25, IDF). Each context
paragraph is one point. The collection is recreated fresh each run.

Usage (from brain/):
    python multihop/ingest_multihop.py --corpus multihop/data/hotpotqa_corpus.jsonl \\
        --collection hotpotqa_corpus
    python multihop/ingest_multihop.py --corpus multihop/data/2wiki_corpus.jsonl \\
        --collection 2wiki_corpus
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_MH_DIR = Path(__file__).resolve().parent
_BRAIN_ROOT = _MH_DIR.parent
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
DENSE_MODEL = "BAAI/bge-m3"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_SIZE = 1024
BATCH = 64


def main() -> int:
    ap = argparse.ArgumentParser("Ingest a multi-hop QA corpus into Qdrant")
    ap.add_argument("--corpus", required=True, help="Path to the *_corpus.jsonl")
    ap.add_argument("--collection", required=True, help="Qdrant collection name")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"[mh-ingest] corpus not found: {corpus}")
        return 1

    with open(corpus, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    texts = [r["text"] for r in rows]
    print(f"[mh-ingest] {len(texts)} paragraphs -> '{args.collection}'")

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(args.collection):
        client.delete_collection(args.collection)
    client.create_collection(
        collection_name=args.collection,
        vectors_config={"dense": models.VectorParams(
            size=DENSE_SIZE, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF)},
    )
    print(f"[mh-ingest] created collection '{args.collection}'")

    print("[mh-ingest] loading embedding models...")
    dense_model = BGEM3FlagModel(DENSE_MODEL, use_fp16=False)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    print(f"[mh-ingest] embedding {len(texts)} paragraphs...")
    dense_out = dense_model.encode(
        texts, batch_size=8, max_length=512,
        return_dense=True, return_sparse=False, return_colbert_vecs=False,
    )["dense_vecs"]
    sparse_out = list(sparse_model.embed(texts))

    points = []
    for i, (row, dvec, svec) in enumerate(zip(rows, dense_out, sparse_out)):
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{args.collection}_{i}")),
            payload={
                "text": row["text"],
                "source_file": row.get("source_file", "Unknown"),
                "content_type": "text",
                "section_header": row.get("source_file", ""),
                "page_number": 1,
            },
            vector={
                "dense": dvec.tolist() if hasattr(dvec, "tolist") else list(dvec),
                "sparse": models.SparseVector(
                    indices=svec.indices.tolist(),
                    values=svec.values.tolist()),
            },
        ))

    for start in range(0, len(points), BATCH):
        client.upsert(collection_name=args.collection,
                      points=points[start:start + BATCH], wait=True)
        print(f"  upserted {min(start + BATCH, len(points))}/{len(points)}")

    info = client.get_collection(args.collection)
    print(f"[mh-ingest] done. '{args.collection}' holds {info.points_count} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
