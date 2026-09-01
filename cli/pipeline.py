"""
cli/pipeline.py

Thin entry point for the batch research pipeline (no case-fact
extraction stage -- OFFENCES/ALLEGATION are used as-is). This is
the modularized equivalent of the old research-pipeline.py: same
steps, same behavior, logic now lives in core/.

Run from the project root:

    python -m cli.pipeline

(also works as `python cli/pipeline.py` -- the project root is
added to sys.path below regardless of how it's launched.)
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vectorstore.query_vector_db import load_corpus, BM25Okapi, tokenize, chromadb, embedding_functions, DB_PATH, COLLECTION_NAME

from core.config import (
    OFFENCES,
    ALLEGATION,
    PLANNER_MODEL,
    RANKER_MODEL,
    REASONING_MODEL,
    N_SEARCH_QUERIES,
    MAX_CANDIDATES_TO_RANK,
)
from core.retrieval import retrieve_statutes, build_statute_context
from core.planning import generate_research_plan
from core.web_search import search_tavily_parallel, deduplicate_results
from core.ranking import rank_precedents, select_final_candidates
from core.reasoning import final_legal_reasoning
from core.output import (
    print_statutes,
    print_plan,
    print_candidates,
    print_ranking,
    print_final_candidates,
    save_final_output,
)


def main():
    # ========================================================
    # INITIALIZATION
    # ========================================================

    print("Loading corpus + vector DB...")
    corpus_ids, corpus_docs = load_corpus()
    bm25 = BM25Okapi([tokenize(doc) for doc in corpus_docs])

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_collection(
        COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    print(f"Ready. {collection.count()} sections indexed.")
    print(f"Planner: {PLANNER_MODEL}")
    print(f"Ranker: {RANKER_MODEL}")
    print(f"Reasoner: {REASONING_MODEL}")
    print()
    print("Legal RAG + Precedent Research Agent ready.")

    # ========================================================
    # CASE
    # ========================================================

    print()
    print("=" * 70)
    print("CASE")
    print("-" * 70)
    print("Offences:")
    print(OFFENCES.strip())
    print()
    print("Allegation:")
    print(ALLEGATION.strip())
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 1 — HYBRID STATUTE RETRIEVAL")
    print("=" * 70)

    statute_results = retrieve_statutes(
        OFFENCES,
        ALLEGATION,
        collection,
        bm25,
        corpus_ids,
    )

    if not statute_results:
        print("No statutes retrieved.")
        return

    print_statutes(statute_results)
    statute_context = build_statute_context(statute_results, collection)

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 2 — GROQ 120B LEGAL RESEARCH PLANNER")
    print("=" * 70)

    try:
        research_plan = generate_research_plan(
            OFFENCES,
            ALLEGATION,
            statute_context,
        )
    except Exception as exc:
        print()
        print("[FATAL] Research planner failed.")
        print(exc)
        return

    print_plan(research_plan)

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 3 — TAVILY JUDICIAL RESEARCH SEARCH")
    print("=" * 70)

    queries = research_plan.search_queries
    if len(queries) != N_SEARCH_QUERIES:
        raise RuntimeError(
            f"Internal error: expected {N_SEARCH_QUERIES} queries."
        )

    print(f"Running {len(queries)} Tavily searches in parallel...")
    tavily_results = search_tavily_parallel(queries)
    print()
    print(f"Tavily returned {len(tavily_results)} raw candidates.")

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    tavily_results = deduplicate_results(tavily_results)
    print(
        f"After URL deduplication: "
        f"{len(tavily_results)} unique candidates."
    )

    # ========================================================
    # CANDIDATE LIMIT
    # ========================================================

    candidates = tavily_results[:MAX_CANDIDATES_TO_RANK]
    print()
    print("Python compression:")
    print("  - whitespace normalization")
    print("  - content length limiting")
    print("  - metadata preservation")
    print("  - NO semantic rejection")

    print()
    print(f"Sending {len(candidates)} candidates to 20B.")

    if not candidates:
        print("No Tavily candidates were returned.")
        selected_items = []
        ranking = None
    else:
        print_candidates(candidates)

        # ====================================================
        # STEP 4
        # ====================================================

        print()
        print("=" * 70)
        print(
            "STEP 4 — GROQ 20B RECALL-FIRST RANKER + "
            "SOURCE CLASSIFIER"
        )
        print("=" * 70)

        ranking = None
        try:
            ranking = rank_precedents(
                OFFENCES,
                ALLEGATION,
                research_plan,
                candidates,
            )
            print_ranking(ranking)
        except Exception as exc:
            print()
            print("[WARNING] 20B ranking failed.")
            print(exc)
            print()
            print("Using deterministic Python fallback.")

        # ====================================================
        # PYTHON FINAL SELECTION
        # ====================================================

        selected_items = select_final_candidates(candidates, ranking)

    print_final_candidates(selected_items)

    # ========================================================
    # STEP 5
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 5 — GROQ 120B FINAL LEGAL REASONING")
    print("=" * 70)

    try:
        final_result = final_legal_reasoning(
            OFFENCES,
            ALLEGATION,
            statute_context,
            research_plan,
            selected_items,
        )
    except Exception as exc:
        print()
        print("[FATAL] Final legal reasoning failed.")
        print(exc)
        return

    # ========================================================
    # FINAL JSON
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL JSON")
    print("=" * 70)

    final_json = final_result.model_dump()
    print(json.dumps(final_json, indent=2, ensure_ascii=False))

    # ========================================================
    # SAVE
    # ========================================================

    save_final_output(final_result)

    print()
    print("=" * 70)
    print("LEGAL RESEARCH PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
