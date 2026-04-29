from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState

llm = build_groq_llm(temperature=0.0)


def _build_weak_context(docs: list[dict]) -> str:
    blocks = []

    for idx, doc in enumerate(docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown Source")
        section = metadata.get("section_header", "Unknown Section")
        page = metadata.get("page_number", "Unknown Page")

        block = (
            f"WEAK DOC [{idx + 1}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Text:\n{doc.get('text', '')}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def rewrite_query(state: GraphState):
    original_query = state["original_query"]
    weak_docs = state.get("weak_signal_docs", [])
    retries = state.get("crag_retries", 0)

    print("\n[Final Combined] Rewriting retrieval query...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    weak_context = _build_weak_context(weak_docs) if weak_docs else "No weak documents available."

    prompt = f"""You are rewriting a user question into better retrieval queries for academic document search.

Original user question:
{original_query}

Weak or partial retrieved evidence:
---
{weak_context}
---

Instructions:
- Rewrite ONLY for retrieval quality.
- Preserve the original user intent exactly.
- If the question contains multiple distinct entities (e.g. comparing two different papers or algorithms), decompose the question into multiple distinct sub-queries (one for each entity).
- If the question is about a single topic, just output one rewritten query.
- Prefer short keyword-rich phrasing useful for hybrid search.
- Do NOT answer the question.
- Do NOT add commentary.
- You MUST output ONLY a valid JSON array of strings. Example: ["query 1", "query 2"]

Now output the JSON array only.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    import json
    try:
        # Strip markdown code blocks if present
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
                
        search_queries = json.loads(content)
        if not isinstance(search_queries, list):
            search_queries = [str(search_queries)]
    except Exception as e:
        print(f"[Final Combined] Failed to parse JSON ({e}). Falling back to string.")
        search_queries = [content]

    if not search_queries:
        search_queries = [original_query]

    rewritten_query = " | ".join(search_queries)

    print(f"[Final Combined] Original query:  {original_query}")
    print(f"[Final Combined] Rewritten queries: {search_queries}")

    return {
        "search_query": rewritten_query,
        "search_queries": search_queries,
        "crag_retries": retries + 1,
    }