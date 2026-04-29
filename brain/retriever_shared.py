import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from FlagEmbedding import BGEM3FlagModel, FlagReranker

from state_shared import GraphState
from qdrant_config import (
    QDRANT_URL,
    COLLECTION_NAME,
    DENSE_EMBED_MODEL,
    SPARSE_EMBED_MODEL,
    DENSE_PREFETCH_LIMIT,
    SPARSE_PREFETCH_LIMIT,
    FINAL_FUSION_LIMIT,
    RERANKER_MODEL,
    RERANK_TOP_K,
)

load_dotenv()

client = QdrantClient(url=QDRANT_URL)

dense_model = BGEM3FlagModel(
    DENSE_EMBED_MODEL,
    use_fp16=False,
)

sparse_model = SparseTextEmbedding(model_name=SPARSE_EMBED_MODEL)

print(f"[*] Loading cross-encoder reranker: {RERANKER_MODEL} ...")
reranker_model = FlagReranker(RERANKER_MODEL, use_fp16=False)


def _get_dense_query_embedding(query: str) -> list[float]:
    """
    Generate a dense embedding for the query using local BGE-M3.
    """
    output = dense_model.encode(
        [query],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    dense_vec = output["dense_vecs"][0]
    return dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)


def _get_sparse_query_embedding(query: str):
    """
    Generate a sparse embedding for the query using Qdrant/bm25 via FastEmbed.
    """
    return list(sparse_model.embed([query]))[0]


def retrieve_docs(query: str) -> list[dict]:
    """
    Shared hybrid retrieval:
    dense BGE-M3 + sparse BM25 + Qdrant RRF fusion.

    Returns the full ranked retrieval list in original order.
    """
    dense_vec = _get_dense_query_embedding(query)
    sparse_vec = _get_sparse_query_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=DENSE_PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=SPARSE_PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=FINAL_FUSION_LIMIT,
        with_payload=True,
    )

    retrieved_docs = []
    for point in results.points:
        retrieved_docs.append(
            {
                "text": point.payload.get("text", ""),
                "metadata": point.payload,
                "score": point.score,
            }
        )

    # Cross-Encoder Reranking
    if retrieved_docs:
        pairs = [[query, doc["text"]] for doc in retrieved_docs]
        
        # compute_score returns float if len(pairs)==1, else list[float]
        rerank_scores = reranker_model.compute_score(pairs, normalize=True)
        if isinstance(rerank_scores, float):
            rerank_scores = [rerank_scores]
            
        for idx, doc in enumerate(retrieved_docs):
            doc["rerank_score"] = rerank_scores[idx]
            
        # Sort by rerank_score descending
        retrieved_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Keep top K
        retrieved_docs = retrieved_docs[:RERANK_TOP_K]

    return retrieved_docs


def retrieve_and_store(state: GraphState):
    """
    Shared retriever node for simple baseline or other architectures.
    """
    queries = state.get("search_queries")
    if not queries:
        queries = [state["search_query"]]
        
    print(f"\n[Shared Retriever] Retrieving context for {len(queries)} sub-queries: {queries}")

    all_docs = []
    seen_texts = set()

    for q in queries:
        docs = retrieve_docs(q)
        for d in docs:
            text = d.get("text", "")
            if text not in seen_texts:
                seen_texts.add(text)
                all_docs.append(d)

    # Sort merged docs by rerank_score globally across all sub-queries
    all_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    
    # Keep only the top K globally
    all_docs = all_docs[:RERANK_TOP_K]

    print(f"[Shared Retriever] Retrieved and merged {len(all_docs)} unique docs.")
    for idx, doc in enumerate(all_docs[:10]):
        original_score = doc.get("score", 0.0)
        rerank_score = doc.get("rerank_score")
        if rerank_score is not None:
            print(f"  -> Doc {idx + 1}: rerank_score={rerank_score:.4f} (qdrant={original_score:.4f})")
        else:
            print(f"  -> Doc {idx + 1}: score={original_score:.4f}")

    return {
        "retrieved_docs": all_docs
    }