"""
cli/agent.py

Thin entry point for the case-fact-extraction-first agent: takes
a raw, messy free-form legal query, extracts structured case
facts first (core/extraction.py), then runs the same
retrieval -> plan -> search -> rank -> reason pipeline as
cli/pipeline.py.

This is the modularized equivalent of the old agent.py.

final_legal_reasoning_with_query() below is intentionally NOT in
core/reasoning.py: it is agent.py's own variant of that function,
genuinely different from core.reasoning.final_legal_reasoning
(it also takes the raw user query so the reasoner can tailor its
analysis to the exact question asked). Duplicating one function
here is the honest alternative to silently changing its behavior
by forcing it into the shared version.

Run from the project root:

    python -m cli.agent

(also works as `python cli/agent.py` -- the project root is added
to sys.path below regardless of how it's launched.)
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vectorstore.query_vector_db import load_corpus, BM25Okapi, tokenize, chromadb, embedding_functions, DB_PATH, COLLECTION_NAME

from core.config import (
    RAW_USER_QUERY,
    PLANNER_MODEL,
    RANKER_MODEL,
    REASONING_MODEL,
    EXTRACTOR_MODEL,
    N_SEARCH_QUERIES,
    MAX_CANDIDATES_TO_RANK,
    MAX_REASONER_CANDIDATE_CHARS,
    REASONING_MODEL,
)
from core.llm import groq_structured
from core.models import ResearchPlan, LegalResearchOutput
from core.extraction import extract_case_facts
from core.retrieval import retrieve_statutes, build_statute_context
from core.planning import generate_research_plan
from core.web_search import search_tavily_parallel, deduplicate_results
from core.ranking import rank_precedents, select_final_candidates
from core.reasoning import build_final_precedent_context
from core.output import (
    print_extraction,
    print_statutes,
    print_plan,
    print_candidates,
    print_ranking,
    print_final_candidates,
    save_final_output,
)


# ============================================================
# AGENT-SPECIFIC FINAL REASONING
# (differs from core.reasoning.final_legal_reasoning by also
# taking the raw original_user_query -- see module docstring)
# ============================================================

def final_legal_reasoning_with_query(
    offence_text: str,
    allegation_text: str,
    statute_context: str,
    research_plan: ResearchPlan,
    selected_items,
    original_user_query: str,
):
    """
    Final legal reasoning stage.

    NOTE: This function now receives the original_user_query so
    the reasoner can tailor its analysis to the exact question
    the user asked, not just the extracted facts.
    """

    precedent_context = build_final_precedent_context(selected_items)

    system_prompt = """
You are the FINAL LEGAL REASONER in an Indian legal research system.

You receive:

1. The ORIGINAL user query (exactly as asked).
2. Factual allegations extracted from that query.
3. Alleged offences.
4. ORIGINAL BNS / BNSS / BSA statute text.
5. The relevant sections and historical equivalents identified
   by the research plan.
6. Python-reconstructed judicial research candidates (top 3-5,
   already deduplicated and capped by source type).
7. 20B candidate assessments, including source_type.

============================================================
SOURCE HIERARCHY
============================================================

Use the evidence carefully.

CURRENT STATUTORY LAW:
    Original retrieved statutory text.

JUDICIAL AUTHORITY:
    Actual Supreme Court / High Court / other Indian judicial
    decisions contained in the research material -- check
    source_type before treating anything as an authority.

SECONDARY MATERIAL:
    Blogs, articles, news, legal commentary. These may inform
    background understanding but are NOT judicial authority.

Search results are NOT automatically precedents.

============================================================
PRECEDENT RULE
============================================================

Never fabricate:

- case names
- citations
- courts
- dates
- holdings
- quotations
- factual similarities

A blog discussing a judgment is not itself the judgment.

If the search material only identifies a case but does not provide
enough information to establish the holding, say so.

============================================================
STATUTE RULE
============================================================

The ORIGINAL statute text is primary.

Do not blindly trust the planner.

If a planner interpretation conflicts with the actual supplied
statutory text, prefer the statutory text.

============================================================
HISTORICAL LAW
============================================================

Do not infer historical legal meaning from section numbers alone.

Distinguish:

CURRENT LAW

from

HISTORICAL PROVISIONS USED TO LOCATE OLDER CASE LAW.

Criminal intimidation is a particularly important example:
do not casually collapse the historical definition and punishment
provisions into one provision.

============================================================
FACTUAL RULE
============================================================

The complainant's allegations are allegations.

The police officer denies them.

Do not treat disputed allegations as established facts.

Distinguish:

- alleged facts
- evidence
- legal elements
- established facts

============================================================
EVIDENCE COMPLETENESS (CRITICAL)
============================================================

You MUST distinguish between:

"exhaustive_search_no_authority_found"
    -- use ONLY if you are confident the research candidates
       represent a genuinely thorough attempt and still found
       nothing relevant.

"limited_search_may_have_missed_authority"
    -- use whenever the search was narrow, the candidate set was
       small, or you cannot rule out that relevant authority
       exists but simply was not retrieved. This is the SAFE
       DEFAULT for a five-query, single-round search.

Do NOT write a sentence like "no Supreme Court judgments were
found" when you actually mean "no Supreme Court judgments were
retrieved by this search." Those are different claims. Say
explicitly that no directly relevant authority was IDENTIFIED
AMONG THE RETRIEVED CANDIDATES, not that none exists.

============================================================
PER-CLAIM CONFIDENCE (CRITICAL)
============================================================

Do NOT assign a single confidence level to the whole output.

Each entry in precedent_analysis must state:

- the claim itself
- its support: statutory_text / judicial_authority /
  secondary_source / unverified
- its confidence: high / moderate / low

A claim resting on the plain statutory text can be "high"
confidence even if a nearby precedent claim is "low" confidence
because it rests on a blog with no reproduced holding. Do not
average these into one number -- keep them separate so the reader
can see exactly which parts of the analysis are solid and which
are speculative.

============================================================
MANDATORY OUTPUT FIELDS (DO NOT OMIT ANY)
============================================================

Your JSON must contain exactly this structure. Do not omit any field:

- laws (string)
- sections_applied (string)
- precedents (string)
- legal_research (object containing ALL of the following):
  - issues (array of strings)
  - applicable_law (array of strings)
  - precedent_analysis (array of objects)
  - application (array of strings)   <-- MUST BE PRESENT, use [] if empty
  - uncertainties (array of strings)
  - research_queries (array of strings)
  - evidence_completeness (string)

If there is nothing to say for "application", return an empty array [],
but do NOT omit the key.

============================================================
OUTPUT
============================================================

Return ONLY the Pydantic schema.

Do not add extra fields.

Do not invent precedent.
"""

    historical_context = [
        item.model_dump() for item in research_plan.historical_equivalents
    ]

    user_prompt = f"""
============================================================
ORIGINAL USER QUERY
============================================================

{original_user_query}

============================================================
EXTRACTED CASE FACTS
============================================================

OFFENCES:

{offence_text}

ALLEGATION:

{allegation_text}

============================================================
ORIGINAL RAG STATUTES
============================================================

{statute_context}

============================================================
RELEVANT RETRIEVED SECTIONS
============================================================

{json.dumps(research_plan.relevant_retrieved_sections, indent=2, ensure_ascii=False)}

============================================================
HISTORICAL EQUIVALENTS
============================================================

{json.dumps(historical_context, indent=2, ensure_ascii=False)}

============================================================
JUDICIAL RESEARCH CANDIDATES
============================================================

{precedent_context}

============================================================
FINAL TASK
============================================================

Produce the final legal research analysis.

Pay particular attention to:

1. Whether the alleged facts satisfy the legal elements of each
   potential offence.

2. Whether retrieved sections are substantive or procedural.

3. Whether historical equivalents are genuinely equivalent.

4. Whether search results actually contain judicial authorities
   (check source_type).

5. Whether a precedent supports the specific legal proposition
   attributed to it.

6. What remains uncertain because the facts are disputed.

7. Whether your evidence_completeness claim is accurate -- a
   five-query single-round search is almost always
   "limited_search_may_have_missed_authority" unless the
   candidates genuinely cover the space well.

8. Frame your answer in light of the ORIGINAL USER QUERY above.
   The user may have asked a specific question; ensure your
   analysis addresses it directly.

Do not invent authorities simply to make the answer appear complete.
"""

    return groq_structured(
        model=REASONING_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_class=LegalResearchOutput,
        max_retries=2,
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
    print(f"Extractor: {EXTRACTOR_MODEL}")
    print(f"Planner: {PLANNER_MODEL}")
    print(f"Ranker: {RANKER_MODEL}")
    print(f"Reasoner: {REASONING_MODEL}")
    print()
    print("Legal RAG + Precedent Research Agent ready.")

    # ========================================================
    # STEP 0 — CASE FACT EXTRACTION
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 0 — CASE FACT EXTRACTION")
    print("=" * 70)

    try:
        extraction = extract_case_facts(RAW_USER_QUERY)
    except Exception as exc:
        print()
        print("[FATAL] Case fact extraction failed.")
        print(exc)
        return

    print_extraction(extraction)

    # Feed extracted facts into the rest of the pipeline
    offences = extraction.alleged_offences
    allegation = extraction.factual_allegations

    # ========================================================
    # CASE
    # ========================================================

    print()
    print("=" * 70)
    print("CASE (FROM EXTRACTOR)")
    print("-" * 70)
    print("Offences:")
    print(offences.strip())
    print()
    print("Allegation:")
    print(allegation.strip())
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 1 — HYBRID STATUTE RETRIEVAL")
    print("=" * 70)

    statute_results = retrieve_statutes(
        offences,
        allegation,
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
            offences,
            allegation,
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
                offences,
                allegation,
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
        final_result = final_legal_reasoning_with_query(
            offences,
            allegation,
            statute_context,
            research_plan,
            selected_items,
            original_user_query=RAW_USER_QUERY,
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
