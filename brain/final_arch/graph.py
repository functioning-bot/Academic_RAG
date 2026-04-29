from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langgraph.graph import StateGraph, START, END

from state_shared import GraphState
from retriever_shared import retrieve_and_store
from node_retrieval_evaluator import evaluate_retrieval
from node_rewriter import rewrite_query
from node_context_selector import select_best_context
from node_grader import grade_documents
from node_generator import generate
from node_auditor import audit_answer
from node_supervisor import supervisor_agent


def route_from_supervisor(state: GraphState):
    """
    Dynamic hub router reading the decision made by the supervisor LLM.
    """
    directive = state.get("supervisor_directive", "end")
    
    # Map the LLM's chosen string to the exact LangGraph node name
    mapping = {
        "retrieve_original": "retrieve_original",
        "evaluate_retrieval": "evaluate_retrieval",
        "rewrite_query": "rewrite_query",
        "retrieve_rewritten": "retrieve_rewritten",
        "select_best_context": "select_best_context",
        "grade_documents": "grade_documents",
        "generate": "generate",
        "audit_answer": "audit_answer",
        "end": END
    }
    
    return mapping.get(directive, END)


def build_graph():
    workflow = StateGraph(GraphState)

    # 1. The Hub Node
    workflow.add_node("supervisor", supervisor_agent)

    # 2. The Spoke Nodes
    workflow.add_node("retrieve_original", retrieve_and_store)
    workflow.add_node("evaluate_retrieval", evaluate_retrieval)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve_rewritten", retrieve_and_store)
    workflow.add_node("select_best_context", select_best_context)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("audit_answer", audit_answer)

    # 3. Entry Point
    workflow.add_edge(START, "supervisor")

    # 4. Hub-to-Spoke Routing
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "retrieve_original": "retrieve_original",
            "evaluate_retrieval": "evaluate_retrieval",
            "rewrite_query": "rewrite_query",
            "retrieve_rewritten": "retrieve_rewritten",
            "select_best_context": "select_best_context",
            "grade_documents": "grade_documents",
            "generate": "generate",
            "audit_answer": "audit_answer",
            END: END,
        },
    )

    # 5. Spoke-to-Hub Return
    workflow.add_edge("retrieve_original", "supervisor")
    workflow.add_edge("evaluate_retrieval", "supervisor")
    workflow.add_edge("rewrite_query", "retrieve_rewritten") # specific internal sub-flow
    workflow.add_edge("retrieve_rewritten", "select_best_context") # specific internal sub-flow
    workflow.add_edge("select_best_context", "supervisor")
    workflow.add_edge("grade_documents", "supervisor")
    workflow.add_edge("generate", "supervisor")
    workflow.add_edge("audit_answer", "supervisor")

    return workflow.compile()