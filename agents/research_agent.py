"""Research agent node powered by Smolagents.

This agent is intended to run as the second node in a LangGraph workflow.
It reads the medical query from graph state and writes retrieved results back
to MedScopeState-compatible keys.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from smolagents import LiteLLMModel, ToolCallingAgent, tool

from indexing.llamaindex import LlamaIndexPipeline
from tools.pubmed_tool import PubMedPipeline

load_dotenv()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Role: You are a medical research agent. Your job is to find relevant, peer-reviewed scientific information about a specific medical condition.
Process: Always call the PubMed tool first with the illness name. Then evaluate the results - are there enough articles? Is the content relevant? Then call the LlamaIndex retrieval tool to extract the most relevant passages.
Quality criteria: The retrieved chunks should directly address the query. If results seem off-topic or too sparse, try calling the PubMed tool again with a slightly more specific query before proceeding.
Constraints: Only use content from the tools. Do not generate or invent medical information. Do not synthesize or simplify - just retrieve and return the relevant chunks.
Output format: Return the retrieved chunks exactly as the LlamaIndex tool provides them, with all metadata intact."""


class ResearchState(TypedDict, total=False):
	"""Shared state contract for the research node."""

	user_query: str
	query_type: str
	fetched_articles: List[Dict[str, Any]]
	indexed_results: List[Dict[str, Any]]
	final_response: str
	sources: List[Dict[str, Any]]
	error: Optional[str]
	metadata: Dict[str, Any]

	# Backward-compatible aliases that may still exist upstream.
	original_user_query: str
	origibal_user_query: str


@dataclass
class _RuntimeContext:
	"""Holds reusable pipeline instances for tool calls."""

	pubmed_pipeline: PubMedPipeline
	retrieval_pipeline: LlamaIndexPipeline
	last_fetched_articles: List[Dict[str, Any]]
	last_indexed_results: List[Dict[str, Any]]


_RUNTIME_CONTEXT: Optional[_RuntimeContext] = None


def _get_runtime_context() -> _RuntimeContext:
	"""Lazily initialize expensive pipeline objects once per process."""
	global _RUNTIME_CONTEXT
	if _RUNTIME_CONTEXT is None:
		_RUNTIME_CONTEXT = _RuntimeContext(
			pubmed_pipeline=PubMedPipeline(),
			retrieval_pipeline=LlamaIndexPipeline(),
			last_fetched_articles=[],
			last_indexed_results=[],
		)
	return _RUNTIME_CONTEXT


def _reset_runtime_outputs() -> None:
	"""Reset tool output buffers before each agent run."""
	context = _get_runtime_context()
	context.last_fetched_articles = []
	context.last_indexed_results = []


@tool
def pubmed_search_tool(illness_name: str) -> List[Dict[str, Any]]:
	"""Searches PubMed and PubMed Central for peer-reviewed medical literature.

	It accepts a single illness or condition name as a string.
	It returns a list of article dictionaries with pmid, title, content,
	and content_type fields.
	It should be called first before any retrieval happens
	"""
	context = _get_runtime_context()
	query = (illness_name or "").strip()
	if not query:
		context.last_fetched_articles = []
		return []

	articles = context.pubmed_pipeline.fetch_articles(query=query, max_results=20)
	context.last_fetched_articles = articles
	return articles


@tool
def llamaindex_retrieval_tool(
	articles: List[Dict[str, Any]], query: str
) -> List[Dict[str, Any]]:
	"""Indexes and retrieves the most relevant passages from a list of articles.

	It accepts the article list from the PubMed tool and the original query string.
	It returns a list of text chunks with source metadata.
	It should only be called after the PubMed tool has returned validate results
	"""
	if not articles:
		return []

	context = _get_runtime_context()
	raw_results = context.retrieval_pipeline.process_and_retrieve(articles, query)

	# Normalize fields expected by downstream state while preserving metadata.
	normalized: List[Dict[str, Any]] = []
	for item in raw_results:
		metadata = item.get("metadata", {}) or {}
		normalized.append(
			{
				"text": item.get("chunk_text", ""),
				"pmid": metadata.get("pmid"),
				"title": metadata.get("title"),
				"content_type": metadata.get("content_type"),
				"relevance_score": item.get("relevance_score"),
				"metadata": metadata,
			}
		)
	context.last_indexed_results = normalized
	return normalized


def _build_model() -> LiteLLMModel:
	"""Create the Gemini-backed LiteLLM model for Smolagents."""
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY is required to run the research agent.")

	return LiteLLMModel(
		model_id="gemini/gemini-2.5-flash-reasoning",
		api_key=api_key,
	)


def _build_agent() -> ToolCallingAgent:
	"""Construct the research ToolCallingAgent with required tools and behavior."""
	model = _build_model()
	return ToolCallingAgent(
		tools=[pubmed_search_tool, llamaindex_retrieval_tool],
		model=model,
		max_steps=5,
		system_prompt=SYSTEM_PROMPT,
	)


def _extract_original_query(state: ResearchState) -> str:
	"""Resolve the query key defensively from the incoming graph state."""
	return (
		state.get("original_user_query")
		or state.get("origibal_user_query")
		or state.get("user_query")
		or ""
	)


def _safe_parse_results(result: Any) -> List[Dict[str, Any]]:
	"""Convert agent output into a list of chunk dictionaries."""
	if isinstance(result, list):
		return [item for item in result if isinstance(item, dict)]

	if isinstance(result, dict):
		indexed_results = result.get("indexed_results")
		if isinstance(indexed_results, list):
			return [item for item in indexed_results if isinstance(item, dict)]
		return [result]

	if isinstance(result, str):
		cleaned = result.strip()
		try:
			parsed = json.loads(cleaned)
			if isinstance(parsed, list):
				return [item for item in parsed if isinstance(item, dict)]
			if isinstance(parsed, dict):
				data = parsed.get("indexed_results", parsed)
				if isinstance(data, list):
					return [item for item in data if isinstance(item, dict)]
				if isinstance(data, dict):
					return [data]
		except json.JSONDecodeError:
			logger.warning("Agent output was not valid JSON; returning empty indexed_results.")
	return []


def run_research_agent(state: ResearchState) -> ResearchState:
	"""Execute the Smolagents research workflow and write MedScope state keys."""
	query_type = state.get("query_type", "")
	original_query = _extract_original_query(state)
	metadata = dict(state.get("metadata") or {})

	if not original_query:
		logger.warning("No original user query found in state.")
		metadata["research_agent"] = {
			"status": "failed",
			"reason": "missing_user_query",
		}
		return {
			"fetched_articles": [],
			"indexed_results": [],
			"error": "No user_query provided to research agent.",
			"metadata": metadata,
		}

	task = (
		"You are running inside a medical research workflow.\n"
		f"query_type: {query_type}\n"
		f"original_user_query: {original_query}\n\n"
		"Instructions:\n"
		"1) Call pubmed_search_tool first.\n"
		"2) Evaluate if results are relevant/sufficient. If sparse or off-topic, retry pubmed_search_tool with a more specific query.\n"
		"3) Call llamaindex_retrieval_tool only after PubMed results exist, using the article list and the original user query.\n"
		"4) Return ONLY JSON: an array of chunk dictionaries, no extra text."
	)

	try:
		_reset_runtime_outputs()
		agent = _build_agent()
		result = agent.run(task)
		context = _get_runtime_context()
		fetched_articles = context.last_fetched_articles
		indexed_results = _safe_parse_results(result) or context.last_indexed_results

		metadata["research_agent"] = {
			"status": "ok",
			"query_type": query_type,
			"fetched_article_count": len(fetched_articles),
			"indexed_result_count": len(indexed_results),
		}

		return {
			"fetched_articles": fetched_articles,
			"indexed_results": indexed_results,
			"error": None,
			"metadata": metadata,
		}
	except Exception as exc:
		logger.exception("Research agent execution failed: %s", exc)
		metadata["research_agent"] = {
			"status": "failed",
			"query_type": query_type,
			"reason": str(exc),
		}
		return {
			"fetched_articles": [],
			"indexed_results": [],
			"error": f"Research agent failed: {exc}",
			"metadata": metadata,
		}


# LangGraph-compatible alias for node registration.
def research_agent_node(state: ResearchState) -> ResearchState:
	"""LangGraph node entrypoint."""
	return run_research_agent(state)

