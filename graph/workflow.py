"""LangGraph workflow wiring for MedScope agents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.research_agent import research_agent_node
from agents.router_agent import router_agent_node
from agents.simplifier_agent import simplifier_agent_node


class MedScopeState(TypedDict, total=False):
	"""Shared state passed between graph nodes."""

	user_query: str
	query_type: str
	fetched_articles: List[Dict[str, Any]]
	indexed_results: List[Dict[str, Any]]
	final_response: str
	sources: List[Dict[str, Any]]
	error: Optional[str]
	metadata: Dict[str, Any]


VALID_QUERY_TYPES = {"definition", "symptoms", "treatment", "causes", "overview"}


def router_node(state: MedScopeState) -> MedScopeState:
	"""Route user query into a supported medical query type."""
	result = router_agent_node(state)
	query_type = (result.get("query_type") or "").strip().lower()
	error = result.get("error")

	if query_type in VALID_QUERY_TYPES and not error:
		return {
			"query_type": query_type,
			"error": None,
			"metadata": result.get("metadata") or state.get("metadata") or {},
		}

	# Out-of-scope or invalid classifier result.
	return {
		"query_type": "unknown",
		"error": error
		or "I can only help with general medical information requests, not personal diagnosis or non-medical topics.",
		"metadata": result.get("metadata") or state.get("metadata") or {},
	}


def research_node(state: MedScopeState) -> MedScopeState:
	"""Retrieve peer-reviewed evidence chunks for the classified query."""
	result = research_agent_node(state)

	indexed_results = result.get("indexed_results") or []
	error = result.get("error")
	if error or not indexed_results:
		return {
			"fetched_articles": result.get("fetched_articles") or [],
			"indexed_results": indexed_results,
			"error": error or "I could not find enough relevant studies for this request.",
			"metadata": result.get("metadata") or state.get("metadata") or {},
		}

	return {
		"fetched_articles": result.get("fetched_articles") or [],
		"indexed_results": indexed_results,
		"error": None,
		"metadata": result.get("metadata") or state.get("metadata") or {},
	}


def simplification_node(state: MedScopeState) -> MedScopeState:
	"""Rewrite scientific content into plain language and return used sources."""
	result = simplifier_agent_node(state)
	return {
		"final_response": result.get("final_response") or "",
		"sources": result.get("sources") or [],
		"error": result.get("error"),
		"metadata": result.get("metadata") or state.get("metadata") or {},
	}


def error_node(state: MedScopeState) -> MedScopeState:
	"""Always provide a readable, user-friendly fallback response."""
	raw_error = (state.get("error") or "").strip()
	friendly_reason = raw_error or "Something went wrong while preparing your medical summary."

	message = (
		"I could not complete your request this time. "
		f"Reason: {friendly_reason} "
		"Please try rephrasing your question as a general medical information request."
	)
	return {"final_response": message}


def _route_after_router(state: MedScopeState) -> str:
	"""Send unknown/out-of-scope queries to error handling."""
	query_type = (state.get("query_type") or "").strip().lower()
	if state.get("error") or query_type not in VALID_QUERY_TYPES:
		return "error"
	return "research"


def _route_after_research(state: MedScopeState) -> str:
	"""Send retrieval failures to error handling; otherwise simplify."""
	if state.get("error"):
		return "error"
	if not state.get("indexed_results"):
		return "error"
	return "simplify"


def build_workflow():
	"""Create and compile the MedScope state graph."""
	graph = StateGraph(MedScopeState)

	graph.add_node("router_node", router_node)
	graph.add_node("research_node", research_node)
	graph.add_node("simplification_node", simplification_node)
	graph.add_node("error_node", error_node)

	graph.add_conditional_edges(
		"router_node",
		_route_after_router,
		{
			"error": "error_node",
			"research": "research_node",
		},
	)
	graph.add_conditional_edges(
		"research_node",
		_route_after_research,
		{
			"error": "error_node",
			"simplify": "simplification_node",
		},
	)
	graph.add_edge("simplification_node", END)
	graph.add_edge("error_node", END)

	graph.set_entry_point("router_node")
	return graph.compile()


workflow = build_workflow()
