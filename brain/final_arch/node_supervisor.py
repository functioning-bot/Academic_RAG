from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState
from config import MAX_REWRITE_ROUNDS, MAX_AUDIT_RETRIES

llm = build_groq_llm(temperature=0.0)

def supervisor_agent(state: GraphState):
    """
    Supervisor node that decides the next node in the pipeline.
    """
    query = state.get("original_query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    graded_docs = state.get("graded_docs", [])
    candidate_docs = state.get("candidate_docs", [])
    generation = state.get("generation", "")
    auditor_feedback = state.get("auditor_feedback", "")
    citations_pass = state.get("citations_pass", False)
    
    crag_retries = state.get("crag_retries", 0)
    verify_retries = state.get("verify_retries", 0)

    retrieval_sufficient = state.get("retrieval_sufficient", False)

    print(f"\n[Final Combined] Supervisor Planner evaluating state...")
    
    # 1. Start of flow
    if not retrieved_docs and crag_retries == 0:
        print("  -> Fast Path: retrieve_original")
        return {"supervisor_directive": "retrieve_original"}
        
    # 2. End of flow
    if citations_pass and generation:
        print("  -> Fast Path: end (Audit passed)")
        return {"supervisor_directive": "end"}
        
    if verify_retries > MAX_AUDIT_RETRIES:
        print("  -> Fast Path: end (Max audit retries reached)")
        return {"supervisor_directive": "end"}

    # 3. Deterministic Fast Paths for linear progressions
    if candidate_docs and not graded_docs and retrieval_sufficient:
        print("  -> Fast Path: grade_documents")
        return {"supervisor_directive": "grade_documents"}

    if graded_docs and not generation:
        print("  -> Fast Path: generate")
        return {"supervisor_directive": "generate"}

    if generation and not auditor_feedback and not citations_pass:
        # It generated an answer, but hasn't been audited yet
        print("  -> Fast Path: audit_answer")
        return {"supervisor_directive": "audit_answer"}

    # Dynamic LLM routing for complex states
    prompt = f"""You are the Supervisor Agent of an Academic RAG pipeline.
Your job is to look at the current state of the pipeline and decide the exact next node to run.

--- CURRENT STATE ---
Original Query: "{query}"
Retrieved Docs Count: {len(retrieved_docs)}
Candidate Docs Count: {len(candidate_docs)}
Graded Docs Count: {len(graded_docs)}
Generation Present: {"Yes" if generation else "No"}
Retrieval Evaluated as Sufficient: {"Yes" if retrieval_sufficient else "No"}
Auditor Feedback: "{auditor_feedback}"
Citations Passed Audit: {"Yes" if citations_pass else "No"}
Rewrite Retries: {crag_retries} / {MAX_REWRITE_ROUNDS}
Audit Retries: {verify_retries} / {MAX_AUDIT_RETRIES}
---------------------

Choose EXACTLY ONE of the following node names to run next:
- "evaluate_retrieval" (if Retrieved Docs Count > 0 but Candidate Docs Count is 0)
- "rewrite_query" (if Retrieval Evaluated as Sufficient is No AND Rewrite Retries < 3)
- "grade_documents" (if Candidate Docs Count > 0 AND Graded Docs Count is 0 AND Retrieval Evaluated as Sufficient is Yes)
- "generate" (if Graded Docs Count > 0 AND Generation Present is No. OR if there is Auditor Feedback to fix the answer)
- "audit_answer" (if Generation Present is Yes but it hasn't passed audit yet)
- "end" (if Citations Passed Audit is Yes, OR retries are exhausted)

Output ONLY the exact node name, nothing else.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()
    
    valid_decisions = [
        "evaluate_retrieval", "rewrite_query", "grade_documents", 
        "generate", "audit_answer", "end", "retrieve_original"
    ]
    
    found = False
    for val in valid_decisions:
        if val in decision:
            decision = val
            found = True
            break
            
    if not found:
        decision = "end"
        
    print(f"  -> Supervisor Decision: {decision}")
    
    return {"supervisor_directive": decision}
