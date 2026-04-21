"""Simplifier agent node powered by Smolagents.

This agent is intended to run after retrieval. It rewrites complex,
scientifically accurate medical content into plain language while preserving
factual fidelity to provided chunks.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from smolagents import LiteLLMModel, ToolCallingAgent

load_dotenv()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Identity and purpose:
You are a medical communication specialist. Your role is to take peer-reviewed scientific content and explain it clearly to a patient with no medical background. You do not diagnose, recommend treatments, or provide personal medical advice. You explain what science currently understands about a condition in terms anyone can understand.

Language rules:
- Never use a medical term without immediately defining it in plain language.
- Keep sentences under 20 words where possible.
- Use active voice.
- Use concrete analogies when explaining biological processes.
- Do not use Latin or Greek medical terminology without translation.
- Write at roughly an eighth-grade reading level.

Structure rules by query type:
- definition: Start with one sentence answering what the condition is, then explain what goes wrong in the body, then who it typically affects.
- symptoms: Start with the most common symptoms in plain language, group related symptoms together, and distinguish early versus advanced symptoms only if supported by evidence.
- causes: Explain known causes or risk factors, distinguish what is controllable versus not controllable, and avoid blame.
- treatment: Explain main treatment categories, what each generally involves, and clearly state treatment choices must be made with a doctor.
- overview: Cover definition, causes, symptoms, and treatment briefly in that order.

Content rules:
- Only use information that appears in the provided chunks.
- If chunks do not contain enough information for part of the query, explicitly say the information is not available in the provided studies.
- Do not contradict the provided sources.

Mandatory disclaimer:
Every response must end with a clear, non-alarming note that this is educational information, that individual situations vary, and that the person should speak with a healthcare provider for advice specific to them.

Output contract:
Return valid JSON only with this exact schema:
{
  "final_response": "string",
  "used_pmids": ["PMID1", "PMID2"]
}

Source usage rules:
- used_pmids must include only PMIDs from chunks actually used to produce final_response.
- used_pmids must be deduplicated.
- Never invent PMIDs.
"""


class SimplifierState(TypedDict, total=False):
	"""Shared state contract for the simplifier node."""

	user_query: str
	query_type: str
	indexed_results: List[Dict[str, Any]]
	final_response: str
	sources: List[Dict[str, Any]]
	error: Optional[str]
	metadata: Dict[str, Any]


def _build_model() -> LiteLLMModel:
	"""Create an Ollama-backed LiteLLM model for Smolagents."""
	api_base = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
	model_name = os.getenv("OLLAMA_MODEL", "mistral")

	return LiteLLMModel(
		model_id=f"ollama/{model_name}",
		api_base=api_base,
	)


def _build_agent() -> ToolCallingAgent:
	"""Construct an LLM-only ToolCallingAgent with strict prompt behavior."""
	return ToolCallingAgent(
		tools=[],
		model=_build_model(),
		max_steps=2,
		instructions=SYSTEM_PROMPT,
	)


def _extract_chunk_text(chunk: Dict[str, Any]) -> str:
	"""Return the primary text field from a retrieval chunk."""
	return (chunk.get("text") or chunk.get("chunk_text") or "").strip()


def _extract_chunk_source(chunk: Dict[str, Any]) -> Dict[str, str]:
	"""Extract PMID/title from chunk-level or nested metadata fields."""
	metadata = chunk.get("metadata", {}) or {}

	pmd = (
		chunk.get("source_pmid")
		or chunk.get("pmid")
		or metadata.get("source_pmid")
		or metadata.get("pmid")
	)
	title = (
		chunk.get("source_title")
		or chunk.get("title")
		or metadata.get("source_title")
		or metadata.get("title")
	)

	pmid = str(pmd).strip() if pmd is not None else ""
	source_title = str(title).strip() if title is not None else ""
	return {"pmid": pmid, "title": source_title}


def _coerce_json_object(raw: Any) -> Dict[str, Any]:
	"""Parse the agent output into a JSON object when possible."""
	if isinstance(raw, dict):
		return raw

	if isinstance(raw, str):
		text = raw.strip()
		if text.startswith("```"):
			text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
			text = re.sub(r"```$", "", text).strip()

		try:
			parsed = json.loads(text)
			return parsed if isinstance(parsed, dict) else {}
		except json.JSONDecodeError:
			logger.warning("Simplifier output was not valid JSON.")

	return {}


def _format_sources(used_pmids: List[str], source_index: Dict[str, str]) -> List[Dict[str, str]]:
	"""Format deduplicated source citations for PMIDs used in final response."""
	sources: List[Dict[str, str]] = []
	for pmid in used_pmids:
		title = source_index.get(pmid, "")
		citation = f"PMID {pmid}: {title}" if title else f"PMID {pmid}"
		sources.append(
			{
				"pmid": pmid,
				"title": title,
				"citation": citation,
			}
		)
	return sources


def _build_task(query_type: str, user_query: str, indexed_results: List[Dict[str, Any]]) -> str:
	"""Build a strict, single-turn task containing retrieval evidence."""

	evidence_blocks: List[Dict[str, Any]] = []
	for idx, chunk in enumerate(indexed_results, start=1):
		source = _extract_chunk_source(chunk)
		text = _extract_chunk_text(chunk)
		if not text:
			continue
		evidence_blocks.append(
			{
				"chunk_id": idx,
				"source_pmid": source["pmid"],
				"source_title": source["title"],
				"text": text,
			}
		)

	payload = {
		"query_type": query_type,
		"user_query": user_query,
		"indexed_results": evidence_blocks,
	}

	return (
		"You are inside a medical simplification workflow.\n"
		"Use only the provided evidence chunks.\n"
		"Return JSON only matching the required schema.\n\n"
		f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=True)}"
	)


def run_simplifier_agent(state: SimplifierState) -> SimplifierState:
	"""Execute LLM-only simplification and return response with used sources."""
	query_type = (state.get("query_type") or "overview").strip().lower()
	user_query = (state.get("user_query") or "").strip()
	indexed_results = state.get("indexed_results") or []
	metadata = dict(state.get("metadata") or {})

	if not indexed_results:
		metadata["simplifier_agent"] = {
			"status": "failed",
			"reason": "missing_indexed_results",
		}
		return {
			"final_response": (
				"I do not have enough retrieved study content to provide a reliable plain-language summary yet. "
				"Please try again after retrieving scientific sources for your question.\n\n"
				"This information is for education, not personal medical advice. "
				"Each person is different, so please speak with a healthcare provider for advice specific to your situation."
			),
			"sources": [],
			"error": "No indexed_results provided to simplifier agent.",
			"metadata": metadata,
		}

	source_index: Dict[str, str] = {}
	for chunk in indexed_results:
		source = _extract_chunk_source(chunk)
		pmid = source["pmid"]
		if pmid and pmid not in source_index:
			source_index[pmid] = source["title"]

	task = _build_task(query_type=query_type, user_query=user_query, indexed_results=indexed_results)

	try:
		agent = _build_agent()
		raw_result = agent.run(task)
		parsed = _coerce_json_object(raw_result)

		final_response = str(parsed.get("final_response") or "").strip()
		raw_used_pmids = parsed.get("used_pmids")
		used_pmids = (
			[str(item).strip() for item in raw_used_pmids if str(item).strip()]
			if isinstance(raw_used_pmids, list)
			else []
		)

		# Keep only valid PMIDs present in provided chunks and preserve order.
		seen = set()
		valid_used_pmids: List[str] = []
		for pmid in used_pmids:
			if pmid in source_index and pmid not in seen:
				seen.add(pmid)
				valid_used_pmids.append(pmid)

		sources = _format_sources(valid_used_pmids, source_index)

		if not final_response:
			raise ValueError("Simplifier produced an empty final_response.")

		metadata["simplifier_agent"] = {
			"status": "ok",
			"query_type": query_type,
			"input_chunk_count": len(indexed_results),
			"used_source_count": len(sources),
		}

		return {
			"final_response": final_response,
			"sources": sources,
			"error": None,
			"metadata": metadata,
		}
	except Exception as exc:
		logger.exception("Simplifier agent execution failed: %s", exc)
		metadata["simplifier_agent"] = {
			"status": "failed",
			"query_type": query_type,
			"reason": str(exc),
		}
		return {
			"final_response": (
				"I could not generate a reliable plain-language explanation from the retrieved studies right now. "
				"Please try again.\n\n"
				"This information is for education, not personal medical advice. "
				"Each person is different, so please speak with a healthcare provider for advice specific to your situation."
			),
			"sources": [],
			"error": f"Simplifier agent failed: {exc}",
			"metadata": metadata,
		}


# LangGraph-compatible alias for node registration.
def simplifier_agent_node(state: SimplifierState) -> SimplifierState:
	"""LangGraph node entrypoint."""
	return run_simplifier_agent(state)
