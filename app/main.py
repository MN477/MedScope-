"""Streamlit interface for running the MedScope workflow end-to-end."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# Ensure project root is importable when running: streamlit run app/main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from graph.workflow import workflow  # noqa: E402


def _build_initial_state(user_query: str) -> Dict[str, Any]:
	"""Return the workflow state shape expected by MedScope graph."""
	return {
		"user_query": user_query,
		"query_type": "",
		"fetched_articles": [],
		"indexed_results": [],
		"final_response": "",
		"sources": [],
		"error": None,
		"metadata": {},
	}


def _render_sources(sources: Any) -> None:
	"""Render source citations cleanly in the UI."""
	if not isinstance(sources, list) or not sources:
		st.info("No sources were returned for this response.")
		return

	st.subheader("Sources")
	for idx, source in enumerate(sources, start=1):
		if not isinstance(source, dict):
			st.write(f"{idx}. {source}")
			continue

		citation = source.get("citation")
		pmid = source.get("pmid")
		title = source.get("title")

		if citation:
			st.write(f"{idx}. {citation}")
		elif pmid and title:
			st.write(f"{idx}. PMID {pmid}: {title}")
		elif pmid:
			st.write(f"{idx}. PMID {pmid}")
		else:
			st.write(f"{idx}. {json.dumps(source, ensure_ascii=True)}")


def main() -> None:
	"""Run the Streamlit app."""
	st.set_page_config(page_title="MedScope", page_icon="🩺", layout="centered")
	st.title("MedScope")
	st.caption("Ask a medical question and get a plain-language answer grounded in retrieved studies.")

	default_query = "What causes Parkinson's disease?"
	user_query = st.text_area(
		"Enter your medical question",
		value=default_query,
		height=120,
		placeholder="Example: What are the symptoms of multiple sclerosis?",
	)

	run_clicked = st.button("Run Full Workflow", type="primary", use_container_width=True)
	show_debug = st.checkbox("Show debug state", value=False)

	if not run_clicked:
		return

	query = user_query.strip()
	if not query:
		st.error("Please enter a query before running the workflow.")
		return

	initial_state = _build_initial_state(query)

	with st.spinner("Running router, research, and simplification agents..."):
		try:
			final_state = workflow.invoke(initial_state)
		except Exception as exc:
			st.error(f"Workflow execution failed: {exc}")
			return

	final_response = (final_state.get("final_response") or "").strip()
	error_message = final_state.get("error")
	query_type = final_state.get("query_type") or ""

	st.subheader("Answer")
	if final_response:
		st.write(final_response)
	else:
		st.warning("The workflow completed but returned an empty response.")

	if query_type:
		st.caption(f"Detected query type: {query_type}")

	if error_message:
		st.warning(f"Workflow note: {error_message}")

	_render_sources(final_state.get("sources"))

	if show_debug:
		st.subheader("Debug State")
		st.json(final_state)


if __name__ == "__main__":
	main()
