from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState

llm = build_groq_llm(temperature=0.0)


def _build_context_blocks(docs: list[dict]) -> str:
    blocks = []

    for idx, doc in enumerate(docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown Source")
        section = metadata.get("section_header", "Unknown Section")
        page = metadata.get("page_number", "Unknown Page")
        
        content_type = metadata.get("content_type", "text")
        display_type = "Prose Text"
        if content_type == "figure_description" or metadata.get("has_image_description"):
            display_type = "AI-Generated Figure/Visual Description"
        elif content_type == "table" or metadata.get("has_table"):
            display_type = "Markdown Table"

        block = (
            f"DOCUMENT [{idx + 1}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Content Type: {display_type}\n"
            f"Text:\n{doc.get('text', '')}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


import json

def audit_answer(state: GraphState):
    """
    Claim-Level Citation Verification Auditor.
    Splits the answer into claims, verifies each against chunks, and removes unsupported claims.
    """
    query = state["original_query"]
    answer = state.get("generation", "").strip()
    selected_docs = state.get("graded_docs", [])
    verify_retries = state.get("verify_retries", 0)

    print("\n[Final Combined] Auditing generated answer (Claim-Level)...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    if not answer or not selected_docs:
        print("[Final Combined] Empty answer or no docs. Marking FAIL.")
        return {
            "citations_pass": False,
            "auditor_feedback": "The answer is empty or not grounded in retrieved evidence.",
            "verify_retries": verify_retries + 1,
        }

    context = _build_context_blocks(selected_docs)

    prompt = f"""You are a strict academic auditor. Your job is to perform Claim-Level Citation Verification.

User Question:
{query}

Retrieved Documents:
---
{context}
---

Generated Answer:
{answer}

Instructions:
1. Break the Generated Answer down into distinct claims (sentences or assertions).
2. For each claim, check if it is fully supported by the Retrieved Documents.
3. If a claim is unsupported, overly broad, or contradicts the documents, mark it as UNSUPPORTED. Otherwise, mark it as SUPPORTED.
4. Rewrite the final answer by completely removing all UNSUPPORTED claims. Ensure the revised answer still flows logically, but do not add new information.
5. If the revised answer is completely empty (all claims unsupported) or fails to address the User Question after stripping, set decision to "FAIL". Otherwise, set decision to "PASS".

Output EXACTLY in the following JSON format. Do not include markdown code blocks, just raw JSON:
{{
  "claims": [
    {{
      "claim": "Text of the claim",
      "status": "SUPPORTED or UNSUPPORTED",
      "supporting_doc_indices": [1, 2]
    }}
  ],
  "decision": "PASS or FAIL",
  "feedback": "Short feedback on what was removed or why it failed.",
  "revised_answer": "The new answer text with unsupported claims stripped out."
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        audit_data = json.loads(content)
        decision = audit_data.get("decision", "FAIL").upper()
        feedback = audit_data.get("feedback", "No feedback provided.")
        revised_answer = audit_data.get("revised_answer", "")
        
        passed = (decision == "PASS") and bool(revised_answer.strip())
        
        # Log claims for terminal output
        claims = audit_data.get("claims", [])
        for c in claims:
            print(f"  -> Claim: '{c.get('claim', '')[:50]}...' => {c.get('status', 'UNKNOWN')}")
            
    except json.JSONDecodeError:
        passed = False
        feedback = "Failed to parse JSON output from auditor."
        revised_answer = answer

    if passed:
        print("[Final Combined] Audit PASS.")
        return {
            "citations_pass": True,
            "auditor_feedback": feedback,
            "generation": revised_answer
        }

    print("[Final Combined] Audit FAIL.")
    print(f"  -> Feedback: {feedback}")

    return {
        "citations_pass": False,
        "auditor_feedback": feedback,
        "verify_retries": verify_retries + 1,
    }