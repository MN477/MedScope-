"""Router agent node powered by Smolagents.

This agent classifies a raw user medical query into one supported query type.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from smolagents import LiteLLMModel, ToolCallingAgent

load_dotenv()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Role: You are a medical query classifier. Your only job is to determine what type of medical information a user is asking for.

Output rule: You must respond with exactly one of these five words and nothing else: definition, symptoms, treatment, causes, overview. If the query is not a medical question or is asking for personal diagnosis or advice, respond with exactly: unknown.

Classification guidance:
definition - "what is", "what does X mean", "explain X", "tell me about X". The user wants to understand what the condition fundamentally is.
symptoms - "what are the symptoms of", "how does X feel", "signs of X", "what happens when you have X". The user wants to know what experiencing the condition is like.
causes - "what causes", "why do people get", "risk factors for", "how do you develop X". The user wants to understand origin and risk.
treatment - "how is X treated", "what medication for X", "can X be cured", "how do doctors treat X". The user wants to know what can be done about it.
overview - broad queries that do not fit cleanly into any one category, or queries that explicitly ask for a general summary. "Tell me everything about X", "give me information about X".

No elaboration: Do not explain your reasoning. Do not add punctuation. Return only the single classification word.

Framework rule: The punctuation restriction applies only to the final classification content. You must still emit a valid final_answer tool call using the exact function-call format required by the agent runtime.
"""


ALLOWED_QUERY_TYPES = {
	"definition",
	"symptoms",
	"treatment",
	"causes",
	"overview",
	"unknown",
}


class RouterState(TypedDict, total=False):
	"""Shared state contract for the router node."""

	user_query: str
	query_type: str
	error: Optional[str]
	metadata: Dict[str, Any]

	# Backward-compatible aliases that may still exist upstream.
	original_user_query: str
	origibal_user_query: str


def _build_model() -> LiteLLMModel:
	"""Create an Ollama-backed LiteLLM model for Smolagents."""
	api_base = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
	model_name = os.getenv("OLLAMA_MODEL", "mistral")

	return LiteLLMModel(
		model_id=f"ollama/{model_name}",
		api_base=api_base,
	)


def _build_agent() -> ToolCallingAgent:
	"""Construct an LLM-only query classification agent."""
	return ToolCallingAgent(
		tools=[],
		model=_build_model(),
		max_steps=2,
		instructions=SYSTEM_PROMPT,
	)


def _extract_user_query(state: RouterState) -> str:
	"""Resolve query key defensively from incoming state."""
	return (
		state.get("user_query")
		or state.get("original_user_query")
		or state.get("origibal_user_query")
		or ""
	)


def _normalize_label(raw: Any) -> str:
	"""Normalize model output to a supported router label."""
	text = str(raw or "").strip().lower()
	text = re.sub(r"[^a-z]", "", text)
	if text in ALLOWED_QUERY_TYPES:
		return text
	return "unknown"


def _extract_label_from_error(error_text: str) -> str:
	"""Recover a valid label from provider error payloads when possible."""
	if not error_text:
		return "unknown"

	match = re.search(r"\b(definition|symptoms|treatment|causes|overview|unknown)\b", error_text.lower())
	if match:
		label = match.group(1)
		return label if label in ALLOWED_QUERY_TYPES else "unknown"
	return "unknown"


def run_router_agent(state: RouterState) -> RouterState:
	"""Classify user query type and flag out-of-scope requests."""
	user_query = _extract_user_query(state).strip()
	metadata = dict(state.get("metadata") or {})

	if not user_query:
		metadata["router_agent"] = {
			"status": "failed",
			"reason": "missing_user_query",
		}
		return {
			"query_type": "unknown",
			"error": "Query is out of scope for medical information classification.",
			"metadata": metadata,
		}

	task = (
		"Classify this user query. Return only one word.\n"
		f"user_query: {user_query}"
	)

	try:
		agent = _build_agent()
		raw_result = agent.run(task)
		query_type = _normalize_label(raw_result)

		error: Optional[str] = None
		if query_type == "unknown":
			error = "Query is out of scope or requests personal diagnosis/advice."

		metadata["router_agent"] = {
			"status": "ok",
			"query_type": query_type,
		}

		return {
			"query_type": query_type,
			"error": error,
			"metadata": metadata,
		}
	except Exception as exc:
		logger.exception("Router agent execution failed: %s", exc)
		recovered_query_type = _extract_label_from_error(str(exc))
		if recovered_query_type != "unknown":
			metadata["router_agent"] = {
				"status": "ok_with_fallback",
				"query_type": recovered_query_type,
				"fallback_reason": "Recovered label from provider function-call error payload.",
			}
			return {
				"query_type": recovered_query_type,
				"error": None,
				"metadata": metadata,
			}

		metadata["router_agent"] = {
			"status": "failed",
			"reason": str(exc),
		}
		return {
			"query_type": "unknown",
			"error": f"Router agent failed: {exc}",
			"metadata": metadata,
		}


# LangGraph-compatible alias for node registration.
def router_agent_node(state: RouterState) -> RouterState:
	"""LangGraph node entrypoint."""
	return run_router_agent(state)
