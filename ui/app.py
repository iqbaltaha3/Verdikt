"""
app.py — Indian Legal Research Agent
====================================

Professional Streamlit interface for an Indian legal research pipeline.

Pipeline:

    Case Fact Extraction
            ↓
    Statute Retrieval
            ↓
    Legal Research Planning
            ↓
    Judicial Research
            ↓
    Precedent Ranking
            ↓
    Final Legal Reasoning
            ↓
    Structured Legal Research Report

Run (from the project root, so `core`/`vectorstore` are importable):

    streamlit run ui/app.py

Environment variables required:

    GROQ_API_KEY
    TAVILY_API_KEY
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
load_dotenv()

# `streamlit run ui/app.py` puts this file's own directory (ui/) on
# sys.path, not the project root -- so `core` and `vectorstore`
# wouldn't be importable otherwise. Add the project root explicitly,
# regardless of the current working directory the app is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from pydantic import BaseModel, Field
from groq import Groq
from tavily import TavilyClient


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Legal Research",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

EXTRACTOR_MODEL = "openai/gpt-oss-120b"
PLANNER_MODEL = "openai/gpt-oss-120b"
RANKER_MODEL = "openai/gpt-oss-20b"
REASONING_MODEL = "openai/gpt-oss-120b"

N_STATUTE_RESULTS = 8
N_SEARCH_QUERIES = 5
TAVILY_RESULTS_PER_QUERY = 3

MAX_RESULT_CONTENT_CHARS = 700
MAX_CANDIDATES_TO_RANK = 15

MIN_SELECTED_PRECEDENTS = 3
MAX_SELECTED_PRECEDENTS = 5

MAX_STATUTE_TEXT_CHARS = 1200
MAX_REASONER_CANDIDATE_CHARS = 500


SOURCE_TYPE_AUTHORITY_CAPS: Dict[str, int] = {
    "supreme_court_judgment": 100,
    "high_court_judgment": 90,
    "tribunal_decision": 75,
    "judgment_repository": 65,
    "legal_commentary": 35,
    "legal_blog": 20,
    "news_report": 15,
    "generic_information": 10,
    "video": 0,
    "social_media": 0,
    "unknown": 25,
}


SOURCE_TYPE_LABELS = {
    "supreme_court_judgment": "Supreme Court Judgment",
    "high_court_judgment": "High Court Judgment",
    "tribunal_decision": "Tribunal Decision",
    "judgment_repository": "Judgment Repository",
    "legal_commentary": "Legal Commentary",
    "legal_blog": "Legal Blog",
    "news_report": "News Report",
    "video": "Video",
    "social_media": "Social Media",
    "generic_information": "General Information",
    "unknown": "Unclassified",
}


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Overall application */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1400px;
    }

    /* Header */

    .report-title {
        font-size: 2.2rem;
        font-weight: 650;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }

    .report-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.75rem;
    }

    /* Section headings */

    .section-heading {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* Small metadata */

    .metadata-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .metadata-value {
        font-size: 0.95rem;
        margin-top: 0.15rem;
    }

    /* Information panels */

    .notice {
        padding: 1rem 1.1rem;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        background: #f8fafc;
        margin: 0.5rem 0 1rem 0;
    }

    .warning {
        padding: 1rem 1.1rem;
        border: 1px solid #f5d78e;
        border-radius: 0.5rem;
        background: #fffbeb;
        margin: 0.75rem 0;
    }

    .success {
        padding: 1rem 1.1rem;
        border: 1px solid #bbf7d0;
        border-radius: 0.5rem;
        background: #f0fdf4;
        margin: 0.75rem 0;
    }

    /* Legal text */

    .legal-text {
        line-height: 1.7;
        font-size: 0.96rem;
    }

    /* Candidate cards */

    .candidate-title {
        font-weight: 600;
        font-size: 0.95rem;
    }

    .candidate-domain {
        color: #64748b;
        font-size: 0.8rem;
    }

    /* Reduce excessive spacing */

    div[data-testid="stExpander"] {
        margin-bottom: 0.45rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# CLIENTS
# ============================================================

@st.cache_resource
def get_groq_client(api_key: str):
    if not api_key:
        return None
    return Groq(api_key=api_key)


@st.cache_resource
def get_tavily_client(api_key: str):
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

# ============================================================
# PYDANTIC SCHEMAS
# ============================================================
#
# These are byte-identical to the ones used by the CLI pipeline
# and agent, so they now live in core.models instead of being
# redefined here.

from core.models import (
    CaseFactExtraction,
    HistoricalEquivalent,
    ResearchPlan,
    SourceType,
    CandidateAssessment,
    PrecedentRanking,
    PrecedentAnalysisItem,
    FinalLegalResearch,
    LegalResearchOutput,
)

# strict_schema() is also byte-identical to the CLI version.
from core.llm import strict_schema


# ============================================================
# GROQ STRUCTURED CALL
# ============================================================
#
# app.py's own version (kept local — its retry/error-handling
# behavior genuinely differs from the CLI's core.llm.groq_structured).

def groq_structured(
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema_class,
    max_retries: int = 2,
):

    schema = strict_schema(schema_class)

    client = get_groq_client(GROQ_API_KEY)

    if client is None:
        raise RuntimeError(
            "Groq client is not configured. "
            "Set GROQ_API_KEY in the environment."
        )

    last_error = None

    for attempt in range(max_retries + 1):

        try:

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_class.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
                reasoning_effort="medium",
            )

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError(
                    "Empty structured response."
                )

            parsed = json.loads(content)

            return schema_class.model_validate(parsed)

        except Exception as exc:

            last_error = exc

            err_str = str(exc)

            if (
                attempt == max_retries
                and (
                    "json_validate_failed" in err_str
                    or "does not match" in err_str
                )
            ):

                try:

                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        response_format={
                            "type": "json_object"
                        },
                        reasoning_effort="medium",
                    )

                    content = response.choices[0].message.content

                    parsed = json.loads(content)

                    if "legal_research" in parsed:

                        lr = parsed["legal_research"]

                        if "application" not in lr:
                            lr["application"] = []

                        parsed["legal_research"] = lr

                    return schema_class.model_validate(parsed)

                except Exception as repair_exc:

                    last_error = repair_exc

            if attempt < max_retries:
                time.sleep(1.5)

    raise RuntimeError(
        f"Structured model call failed after "
        f"{max_retries + 1} attempts: {last_error}"
    )


# ============================================================
# STEP 0 — CASE FACT EXTRACTION
# ============================================================

def extract_case_facts(raw_query: str) -> CaseFactExtraction:

    system_prompt = """
You are the CASE FACT EXTRACTOR in an Indian legal research system.

Your task is to extract structured information from a raw legal query.

Identify:

1. A concise summary.
2. Parties involved.
3. Factual allegations.
4. Alleged offences or legal wrongs.
5. Key legal issues.
6. Disputed facts.
7. Procedural context.

RULES:

- Never invent facts.
- Distinguish allegations from established facts.
- If information is missing, say so.
- Do not determine guilt or liability.
- Return only the requested schema.
"""

    user_prompt = f"""
RAW LEGAL QUERY
===============

{raw_query}

Extract the case facts into the requested structure.
"""

    return groq_structured(
        model=EXTRACTOR_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_class=CaseFactExtraction,
        max_retries=2,
    )


# ============================================================
# STEP 1 — STATUTE RETRIEVAL (real Chroma + BM25 hybrid search)
# ============================================================
#
# This used to be retrieve_statutes_stub(): a handful of
# hardcoded keyword-matched fake sections, completely
# disconnected from the actual BNS/BNSS/BSA vector DB. It now
# uses the same real hybrid BM25 + Chroma retrieval as the CLI
# pipeline/agent (core.retrieval), against the real chroma_db/
# built by vectorstore/build_vector_db.py.

from vectorstore.query_vector_db import (
    load_corpus,
    BM25Okapi,
    tokenize,
    chromadb,
    embedding_functions,
    DB_PATH,
    COLLECTION_NAME,
)
from core.retrieval import retrieve_statutes as _retrieve_statutes_real
from core.retrieval import build_statute_context as _build_statute_context_real


@st.cache_resource
def get_retrieval_resources():
    """
    Load the BM25 index and open the persistent Chroma collection
    once per Streamlit process, so every research run reuses them
    instead of reloading the corpus / re-embedding on every click.
    """
    corpus_ids, corpus_docs = load_corpus()
    bm25 = BM25Okapi([tokenize(doc) for doc in corpus_docs])

    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_collection(
        COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    return collection, bm25, corpus_ids


def retrieve_statutes(offence_text: str, allegation_text: str):
    collection, bm25, corpus_ids = get_retrieval_resources()
    return _retrieve_statutes_real(
        offence_text,
        allegation_text,
        collection,
        bm25,
        corpus_ids,
    )


def build_statute_context(results):
    collection, _bm25, _corpus_ids = get_retrieval_resources()
    return _build_statute_context_real(results, collection)


# ============================================================
# STEP 2 — RESEARCH PLANNER
# ============================================================

def generate_research_plan(
    offence_text: str,
    allegation_text: str,
    statute_context: str,
):

    system_prompt = """
You are the LEGAL RESEARCH PLANNER in an Indian legal research system.

Determine:

1. Actual alleged offences.
2. Relevant retrieved statutory sections.
3. Irrelevant retrieved sections.
4. Historical equivalents under earlier statutes.
5. Legal concepts requiring research.
6. Research questions.
7. Search queries.

RULES:

- Do not infer meaning from section numbers alone.
- Historical provisions should only be identified where genuinely relevant.
- Search queries must be useful for locating Indian judicial authority.
- Return exactly five search queries.
- At least two queries should use historical IPC/CrPC terminology where appropriate.
- Do not invent authorities.
"""

    user_prompt = f"""
OFFENCES
========

{offence_text}

ALLEGATION
==========

{allegation_text}

RETRIEVED STATUTES
==================

{statute_context}

Create the legal research plan.
"""

    plan = groq_structured(
        model=PLANNER_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_class=ResearchPlan,
        max_retries=2,
    )

    plan.search_queries = [
        q.strip()
        for q in plan.search_queries
        if q.strip()
    ]

    if len(plan.search_queries) != N_SEARCH_QUERIES:
        raise RuntimeError(
            f"Planner returned "
            f"{len(plan.search_queries)} queries; "
            f"expected {N_SEARCH_QUERIES}."
        )

    return plan


# ============================================================
# TEXT UTILITIES + TAVILY RESULT COMPRESSION
# ============================================================
#
# normalize_whitespace / infer_domain / compress_tavily_result are
# byte-identical to the CLI's versions, so they're imported from
# core.web_search instead of being redefined here.

from core.web_search import normalize_whitespace, infer_domain, compress_tavily_result


# ============================================================
# TAVILY SEARCH
# ============================================================

def tavily_search(
    query: str,
    candidate_id_start: int,
):

    try:

        client = get_tavily_client(
            TAVILY_API_KEY
        )

        if client is None:
            raise RuntimeError(
                "Tavily client is not configured."
            )

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=TAVILY_RESULTS_PER_QUERY,
            include_raw_content=False,
            include_images=False,
        )

        raw_results = response.get(
            "results",
            []
        )

        compressed = []

        for offset, result in enumerate(
            raw_results
        ):

            candidate_id = (
                candidate_id_start
                + offset
            )

            compressed.append(
                compress_tavily_result(
                    result=result,
                    query=query,
                    candidate_id=candidate_id,
                )
            )

        return compressed

    except Exception:

        return []


def search_tavily_parallel(
    queries: List[str]
):

    all_results = []

    if not queries:
        return all_results

    with ThreadPoolExecutor(
        max_workers=min(
            5,
            len(queries)
        )
    ) as executor:

        futures = {}

        for index, query in enumerate(
            queries
        ):

            candidate_id_start = (
                index
                * TAVILY_RESULTS_PER_QUERY
                + 1
            )

            future = executor.submit(
                tavily_search,
                query,
                candidate_id_start,
            )

            futures[future] = query

        for future in as_completed(
            futures
        ):

            try:

                results = future.result()

                all_results.extend(
                    results
                )

            except Exception:

                pass

    all_results.sort(
        key=lambda x:
            x["candidate_id"]
    )

    return all_results


# ============================================================
# DEDUPLICATION
# ============================================================
#
# Byte-identical to the CLI's version -- imported from
# core.web_search instead of being redefined here.

from core.web_search import deduplicate_results


# ============================================================
# SOURCE QUALITY
# ============================================================

def source_quality_score(
    candidate: Dict[str, Any]
):

    title = (
        candidate.get("title", "")
        or ""
    ).lower()

    domain = (
        candidate.get("domain", "")
        or ""
    ).lower()

    content = (
        candidate.get("content", "")
        or ""
    ).lower()

    text = (
        title
        + " "
        + domain
        + " "
        + content
    )

    score = 0

    if "sci.gov.in" in domain:
        score += 60

    elif ".hc." in domain:
        score += 50

    elif "highcourt" in domain:
        score += 50

    elif domain.endswith(".gov.in"):
        score += 35

    for legal_domain in [
        "indiankanoon.org",
        "sci.gov.in",
        "hcservices.ecourts.gov.in",
        "main.sci.gov.in",
        "livelaw.in",
        "barandbench.com",
        "casemine.com",
    ]:

        if legal_domain in domain:
            score += 20
            break

    for term in [
        "judgment",
        "judgement",
        "order",
        "petition",
        "appeal",
        "criminal appeal",
        "writ petition",
        "supreme court",
        "high court",
    ]:

        if term in text:
            score += 5

    for term in [
        "united states",
        "u.s.",
        "us law",
        "american law",
        "new york",
        "california",
        "england and wales",
        "uk law",
    ]:

        if term in text:
            score -= 30

    for term in [
        "understanding",
        "what is",
        "explained",
        "guide",
        "legal rights",
        "legal protection",
        "blog",
    ]:

        if term in title:
            score -= 8

    tavily_score = candidate.get(
        "tavily_score"
    )

    if isinstance(
        tavily_score,
        (int, float)
    ):

        score += int(
            max(
                0,
                min(
                    20,
                    tavily_score * 20
                )
            )
        )

    return max(
        0,
        min(
            100,
            score
        )
    )


def infer_source_type_fallback(
    candidate: Dict[str, Any]
):

    domain = (
        candidate.get("domain", "")
        or ""
    ).lower()

    title = (
        candidate.get("title", "")
        or ""
    ).lower()

    if (
        "sci.gov.in" in domain
        or "supreme court" in title
    ):
        return "supreme_court_judgment"

    if (
        ".hc." in domain
        or "highcourt" in domain
        or "high court" in title
    ):
        return "high_court_judgment"

    if "indiankanoon.org" in domain:
        return "judgment_repository"

    if any(
        social in domain
        for social in [
            "instagram.com",
            "youtube.com",
            "facebook.com",
            "twitter.com",
            "x.com",
            "tiktok.com",
        ]
    ):
        return "social_media"

    if "youtube.com" in domain:
        return "video"

    if any(
        legal_site in domain
        for legal_site in [
            "livelaw.in",
            "barandbench.com",
            "casemine.com",
        ]
    ):
        return "legal_commentary"

    if (
        "blog" in domain
        or "blog" in title
    ):
        return "legal_blog"

    return "unknown"


def apply_authority_cap(
    source_type: str,
    judicial_authority_score: int,
):

    cap = SOURCE_TYPE_AUTHORITY_CAPS.get(
        source_type,
        SOURCE_TYPE_AUTHORITY_CAPS["unknown"],
    )

    return min(
        judicial_authority_score,
        cap
    )


# ============================================================
# RANKER CONTEXT
# ============================================================

def build_ranker_candidates(
    candidates: List[Dict[str, Any]]
):

    blocks = []

    for candidate in candidates:

        blocks.append(
            f"""
CANDIDATE {candidate['candidate_id']}

Originating search query:
{candidate['search_query']}

Title:
{candidate['title']}

URL:
{candidate['url']}

Domain:
{candidate['domain']}

Tavily score:
{candidate['tavily_score']}

Content:
{candidate['content']}
""".strip()
        )

    return "\n\n".join(blocks)


# ============================================================
# STEP 4 — PRECEDENT RANKER
# ============================================================

def rank_precedents(
    offence_text: str,
    allegation_text: str,
    research_plan: ResearchPlan,
    candidates: List[Dict[str, Any]],
):

    if not candidates:
        raise RuntimeError(
            "No judicial research candidates available."
        )

    BATCH_SIZE = 5

    all_assessments = []

    for i in range(
        0,
        len(candidates),
        BATCH_SIZE,
    ):

        batch = candidates[
            i:i + BATCH_SIZE
        ]

        candidate_text = build_ranker_candidates(
            batch
        )

        system_prompt = """
You are the PRECEDENT CANDIDATE SCORER AND SOURCE CLASSIFIER.

This is a recall-first stage.

Classify every candidate into exactly one source type.

A legal blog is not judicial authority.

A news article is not judicial authority.

A judgment repository is not necessarily the issuing court.

For every candidate provide:

- source_type
- relevance_score
- source_quality_score
- judicial_authority_score
- factual_similarity_score
- keep_for_reasoner
- redundant_with
- reason

Do not invent information.

Return one assessment for every candidate.
"""

        historical_context = [
            item.model_dump()
            for item in research_plan.historical_equivalents
        ]

        user_prompt = f"""
CASE

OFFENCES:
{offence_text}

ALLEGATION:
{allegation_text}


RESEARCH PLAN

Actual alleged offences:
{json.dumps(
    research_plan.actual_alleged_offences,
    indent=2,
    ensure_ascii=False,
)}

Relevant retrieved sections:
{json.dumps(
    research_plan.relevant_retrieved_sections,
    indent=2,
    ensure_ascii=False,
)}

Historical equivalents:
{json.dumps(
    historical_context,
    indent=2,
    ensure_ascii=False,
)}


CANDIDATES

{candidate_text}


TASK

Assess every supplied candidate.

Use high recall.

Do not omit candidates.

Do not invent judicial authority.
"""

        ranking = groq_structured(
            model=RANKER_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_class=PrecedentRanking,
            max_retries=2,
        )

        all_assessments.extend(
            ranking.assessments
        )

    merged = PrecedentRanking(
        assessments=all_assessments
    )

    expected_ids = {
        c["candidate_id"]
        for c in candidates
    }

    returned_ids = {
        a.candidate_id
        for a in merged.assessments
    }

    missing_ids = (
        expected_ids
        - returned_ids
    )

    if missing_ids:
        raise RuntimeError(
            "Ranker failed to assess candidates: "
            + str(sorted(missing_ids))
        )

    for assessment in merged.assessments:

        assessment.judicial_authority_score = (
            apply_authority_cap(
                assessment.source_type,
                assessment.judicial_authority_score,
            )
        )

    return merged


# ============================================================
# FINAL CANDIDATE SELECTION
# ============================================================

def select_final_candidates(
    candidates: List[Dict[str, Any]],
    ranking: Optional[PrecedentRanking],
):

    candidate_map = {
        c["candidate_id"]: c
        for c in candidates
    }

    scored = []

    if ranking:

        redundant_ids = {
            a.candidate_id
            for a in ranking.assessments
            if a.redundant_with is not None
        }

        for assessment in ranking.assessments:

            candidate = candidate_map.get(
                assessment.candidate_id
            )

            if not candidate:
                continue

            if (
                assessment.candidate_id
                in redundant_ids
                and assessment.redundant_with
                in candidate_map
            ):
                continue

            combined_score = (
                assessment.judicial_authority_score
                * 0.35
                +
                assessment.relevance_score
                * 0.35
                +
                assessment.source_quality_score
                * 0.15
                +
                assessment.factual_similarity_score
                * 0.15
            )

            scored.append(
                {
                    "candidate": candidate,
                    "combined_score": combined_score,
                    "ranker_assessment": assessment,
                }
            )

    if not scored:

        for candidate in candidates:

            scored.append(
                {
                    "candidate": candidate,
                    "combined_score":
                        source_quality_score(
                            candidate
                        ),
                    "ranker_assessment": None,
                    "fallback_source_type":
                        infer_source_type_fallback(
                            candidate
                        ),
                }
            )

    scored.sort(
        key=lambda item:
            item["combined_score"],
        reverse=True,
    )

    judicial_like = []

    for item in scored:

        assessment = item[
            "ranker_assessment"
        ]

        if assessment is None:
            continue

        if (
            assessment.judicial_authority_score
            >= 40
            and assessment.relevance_score
            >= 40
        ):

            judicial_like.append(item)

    if len(judicial_like) >= MIN_SELECTED_PRECEDENTS:

        selected = judicial_like[
            :MAX_SELECTED_PRECEDENTS
        ]

    else:

        selected = scored[
            :MAX_SELECTED_PRECEDENTS
        ]

    return selected


# ============================================================
# FINAL PRECEDENT CONTEXT
# ============================================================

def build_final_precedent_context(
    selected_items
):

    if not selected_items:

        return (
            "NO PRECEDENT CANDIDATES WERE RETAINED.\n\n"
            "The final reasoner MUST NOT invent precedent."
        )

    blocks = []

    for position, item in enumerate(
        selected_items,
        start=1,
    ):

        candidate = item["candidate"]

        assessment = item[
            "ranker_assessment"
        ]

        trimmed_content = candidate[
            "content"
        ][:MAX_REASONER_CANDIDATE_CHARS]

        if assessment:

            score_block = f"""
Source type:
{assessment.source_type}

Relevance:
{assessment.relevance_score}

Judicial authority:
{assessment.judicial_authority_score}

Keep recommendation:
{assessment.keep_for_reasoner}

Reason:
{assessment.reason}
""".strip()

        else:

            fallback_type = item.get(
                "fallback_source_type",
                "unknown"
            )

            score_block = f"""
Ranker assessment:
Unavailable.

Inferred source type:
{fallback_type}
""".strip()

        blocks.append(
            f"""
PRECEDENT CANDIDATE {position}

Original candidate ID:
{candidate['candidate_id']}

Search query:
{candidate['search_query']}

Title:
{candidate['title']}

URL:
{candidate['url']}

Domain:
{candidate['domain']}

Retrieved content:
{trimmed_content}

{score_block}
""".strip()
        )

    return "\n\n".join(blocks)


# ============================================================
# STEP 5 — FINAL LEGAL REASONING
# ============================================================

def final_legal_reasoning(
    offence_text: str,
    allegation_text: str,
    statute_context: str,
    research_plan: ResearchPlan,
    selected_items,
    original_user_query: str,
):

    precedent_context = (
        build_final_precedent_context(
            selected_items
        )
    )

    system_prompt = """
You are the FINAL LEGAL REASONER in an Indian legal research system.

You are a neutral legal research officer.

You do not determine guilt, innocence, liability or punishment.

You must analyze the law relevant to the facts presented.

SOURCE HIERARCHY:

1. Current statutory text.
2. Actual Supreme Court / High Court decisions.
3. Secondary legal material.

Never fabricate:

- case names
- citations
- courts
- dates
- holdings
- quotations
- factual similarities
- statutory provisions

Historical IPC/CrPC provisions may be used to locate older case law, but must not be confused with current BNS/BNSS law.

Treat factual allegations as allegations.

Do not treat disputed facts as established.

For every legal proposition distinguish:

- statutory text
- judicial authority
- secondary source
- unverified proposition

Evidence completeness should normally be:

limited_search_may_have_missed_authority

unless the research genuinely establishes otherwise.

Return only the requested schema.
"""

    historical_context = [
        item.model_dump()
        for item in research_plan.historical_equivalents
    ]

    user_prompt = f"""
ORIGINAL USER QUERY
===================

{original_user_query}


EXTRACTED FACTS
===============

OFFENCES:
{offence_text}

ALLEGATION:
{allegation_text}


RETRIEVED STATUTORY MATERIAL
============================

{statute_context}


RELEVANT SECTIONS
=================

{json.dumps(
    research_plan.relevant_retrieved_sections,
    indent=2,
    ensure_ascii=False,
)}


HISTORICAL EQUIVALENTS
======================

{json.dumps(
    historical_context,
    indent=2,
    ensure_ascii=False,
)}


JUDICIAL RESEARCH MATERIAL
==========================

{precedent_context}


TASK
====

Produce the final legal research analysis.

Address:

1. Legal issues.
2. Applicable law.
3. Relevant precedent propositions.
4. Application of law to the alleged facts.
5. Evidentiary limitations and uncertainties.
6. Research limitations.

Do not invent authority.

Do not treat allegations as established facts.
"""

    return groq_structured(
        model=REASONING_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_class=LegalResearchOutput,
        max_retries=2,
    )


# ============================================================
# UI HELPERS
# ============================================================

def render_score(
    label: str,
    score: int,
):

    st.write(
        f"**{label}: {score}/100**"
    )

    st.progress(
        max(
            0,
            min(
                100,
                score,
            )
        )
        / 100
    )


def source_type_label(
    source_type: str
):

    return SOURCE_TYPE_LABELS.get(
        source_type,
        source_type.replace(
            "_",
            " "
        ).title(),
    )


def confidence_label(
    confidence: str
):

    mapping = {
        "high": "High confidence",
        "moderate": "Moderate confidence",
        "low": "Low confidence",
    }

    return mapping.get(
        confidence,
        confidence.title(),
    )


def render_bullet_list(
    items: List[str]
):

    if not items:

        st.caption(
            "No information was returned for this section."
        )

        return

    for item in items:

        st.markdown(
            f"- {item}"
        )


def safe_float(value):

    if isinstance(
        value,
        (float, int)
    ):
        return float(value)

    return 0.0


# ============================================================
# REPORT: CASE OVERVIEW
# ============================================================

def render_case_overview(
    extraction: CaseFactExtraction
):

    st.markdown(
        "## Case Overview"
    )

    st.markdown(
        f"""
        <div class="notice">
        <strong>Research summary</strong><br><br>
        {extraction.summary}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### Parties"
        )

        render_bullet_list(
            extraction.parties_involved
        )

    with col2:

        st.markdown(
            "#### Procedural context"
        )

        if extraction.procedural_context:

            st.write(
                extraction.procedural_context
            )

        else:

            st.caption(
                "No procedural context was identified in the query."
            )

    st.markdown(
        "#### Factual allegations"
    )

    st.write(
        extraction.factual_allegations
    )

    st.markdown(
        "#### Alleged offences / legal wrongs"
    )

    st.write(
        extraction.alleged_offences
    )

    if extraction.disputed_facts:

        st.markdown(
            "#### Disputed or contested facts"
        )

        render_bullet_list(
            extraction.disputed_facts
        )


# ============================================================
# REPORT: ISSUES
# ============================================================

def render_issues(
    extraction: CaseFactExtraction,
    legal_research: FinalLegalResearch,
):

    st.markdown(
        "## Issues for Determination"
    )

    issues = (
        legal_research.issues
        or extraction.legal_issues
    )

    if not issues:

        st.info(
            "No discrete legal issues were identified."
        )

        return

    for index, issue in enumerate(
        issues,
        start=1,
    ):

        with st.expander(
            f"Issue {index}: {issue}",
            expanded=False,
        ):

            st.markdown(
                issue
            )

            if index <= len(
                legal_research.application
            ):

                st.markdown(
                    "**Relevant analysis**"
                )

                st.write(
                    legal_research.application[
                        index - 1
                    ]
                )


# ============================================================
# REPORT: STATUTORY LAW
# ============================================================

def render_statutory_law(
    statute_results,
    legal_research: FinalLegalResearch,
):

    st.markdown(
        "## Applicable Law"
    )

    if legal_research.applicable_law:

        st.markdown(
            "### Legal provisions identified by the final analysis"
        )

        render_bullet_list(
            legal_research.applicable_law
        )

    st.markdown(
        "### Retrieved statutory material"
    )

    if not statute_results:

        st.info(
            "No statutory retrieval results were available."
        )

        return

    for index, result in enumerate(
        statute_results,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {}
        )

        act = metadata.get(
            "act",
            "Unknown Act"
        )

        section = metadata.get(
            "section_number",
            "Unknown"
        )

        chapter = metadata.get(
            "chapter_title",
            ""
        )

        title = (
            f"{act} §{section}"
        )

        if chapter:

            title += (
                f" — {chapter}"
            )

        with st.expander(
            title,
            expanded=False,
        ):

            col1, col2 = st.columns(
                [1, 3]
            )

            with col1:

                st.markdown(
                    "**Act**"
                )

                st.write(
                    act
                )

                st.markdown(
                    "**Section**"
                )

                st.write(
                    f"§{section}"
                )

            with col2:

                st.markdown(
                    "**Why it was retrieved**"
                )

                if index <= len(
                    legal_research.applicable_law
                ):

                    st.write(
                        legal_research.applicable_law[
                            index - 1
                        ]
                    )

                else:

                    st.write(
                        "This provision was retrieved during statutory research."
                    )

            st.markdown(
                "**Retrieved statutory text**"
            )

            st.code(
                result.get(
                    "document",
                    result.get(
                        "snippet",
                        ""
                    )
                ),
                language=None,
            )


# ============================================================
# REPORT: PRECEDENT RESEARCH
# ============================================================

def render_precedent_research(
    candidates,
    ranking,
    selected_items,
):

    st.markdown(
        "## Judicial Research"
    )

    st.markdown(
        "The judicial research stage searches for potentially relevant Indian authorities, classifies the retrieved material, and then selects the strongest candidates for final legal reasoning."
    )

    if not candidates:

        st.warning(
            "No judicial research candidates were retrieved."
        )

        return

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Retrieved",
            len(candidates),
        )

    with col2:

        if ranking:

            st.metric(
                "Assessed",
                len(
                    ranking.assessments
                ),
            )

        else:

            st.metric(
                "Assessed",
                0,
            )

    with col3:

        st.metric(
            "Selected",
            len(
                selected_items
            ),
        )

    st.markdown(
        "### Selected authorities"
    )

    if selected_items:

        for index, item in enumerate(
            selected_items,
            start=1,
        ):

            candidate = item[
                "candidate"
            ]

            assessment = item.get(
                "ranker_assessment"
            )

            title = (
                candidate.get(
                    "title"
                )
                or "Untitled result"
            )

            with st.expander(
                f"{index}. {title}",
                expanded=True,
            ):

                col1, col2 = st.columns(
                    [2, 1]
                )

                with col1:

                    st.markdown(
                        "**Source**"
                    )

                    st.write(
                        candidate.get(
                            "domain",
                            "Unknown"
                        )
                    )

                    if candidate.get(
                        "url"
                    ):

                        st.markdown(
                            candidate[
                                "url"
                            ]
                        )

                    st.markdown(
                        "**Search query that retrieved it**"
                    )

                    st.write(
                        candidate.get(
                            "search_query",
                            ""
                        )
                    )

                with col2:

                    if assessment:

                        st.markdown(
                            "**Source classification**"
                        )

                        st.write(
                            source_type_label(
                                assessment.source_type
                            )
                        )

                        st.markdown(
                            "**Combined research score**"
                        )

                        st.write(
                            f"{item['combined_score']:.1f}/100"
                        )

                st.markdown(
                    "**Retrieved material**"
                )

                st.write(
                    candidate.get(
                        "content",
                        ""
                    )
                )

                if assessment:

                    st.markdown("---")
                    st.markdown("**Research assessment**")

                    with st.container():

                        render_score(
                            "Relevance",
                            assessment.relevance_score,
                        )

                        render_score(
                            "Source quality",
                            assessment.source_quality_score,
                        )

                        render_score(
                            "Judicial authority",
                            assessment.judicial_authority_score,
                        )

                        render_score(
                            "Factual similarity",
                            assessment.factual_similarity_score,
                        )

                        st.markdown(
                            "**Why the candidate was assessed this way**"
                        )

                        st.write(
                            assessment.reason
                        )

                        if assessment.redundant_with:

                            st.warning(
                                "This candidate was marked as redundant with candidate "
                                f"#{assessment.redundant_with}."
                            )

    else:

        st.info(
            "No final authorities were selected."
        )

    # --------------------------------------------------------
    # All retrieved candidates
    # --------------------------------------------------------

    st.markdown(
        "### All retrieved research candidates"
    )

    st.caption(
        "This section preserves the broader search trail rather than showing only the authorities selected for final reasoning."
    )

    assessment_map = {}

    if ranking:

        assessment_map = {
            a.candidate_id: a
            for a in ranking.assessments
        }

    rows = []

    for candidate in candidates:

        assessment = assessment_map.get(
            candidate["candidate_id"]
        )

        if assessment:

            rows.append(
                {
                    "ID":
                        candidate[
                            "candidate_id"
                        ],
                    "Source":
                        source_type_label(
                            assessment.source_type
                        ),
                    "Relevance":
                        assessment.relevance_score,
                    "Authority":
                        assessment.judicial_authority_score,
                    "Quality":
                        assessment.source_quality_score,
                    "Similarity":
                        assessment.factual_similarity_score,
                    "Selected":
                        any(
                            item["candidate"][
                                "candidate_id"
                            ]
                            == candidate[
                                "candidate_id"
                            ]
                            for item
                            in selected_items
                        ),
                    "Title":
                        candidate[
                            "title"
                        ],
                }
            )

        else:

            rows.append(
                {
                    "ID":
                        candidate[
                            "candidate_id"
                        ],
                    "Source":
                        "Unassessed",
                    "Relevance": 0,
                    "Authority": 0,
                    "Quality":
                        source_quality_score(
                            candidate
                        ),
                    "Similarity": 0,
                    "Selected": False,
                    "Title":
                        candidate[
                            "title"
                        ],
                }
            )

    if rows:

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# REPORT: FINAL PRECEDENT ANALYSIS
# ============================================================

def render_precedent_analysis(
    legal_research: FinalLegalResearch
):

    st.markdown(
        "## Precedent Analysis"
    )

    st.caption(
        "These are the legal propositions generated by the final reasoner and the type of support attributed to each proposition."
    )

    if not legal_research.precedent_analysis:

        st.info(
            "No precedent-specific propositions were returned."
        )

        return

    for index, item in enumerate(
        legal_research.precedent_analysis,
        start=1,
    ):

        support_label = item.support.replace(
            "_",
            " "
        ).title()

        with st.expander(
            f"Proposition {index}: {item.claim}",
            expanded=False,
        ):

            st.markdown(
                "**Legal proposition**"
            )

            st.write(
                item.claim
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "**Support type**"
                )

                st.write(
                    support_label
                )

            with col2:

                st.markdown(
                    "**Confidence**"
                )

                st.write(
                    confidence_label(
                        item.confidence
                    )
                )


# ============================================================
# REPORT: APPLICATION
# ============================================================

def render_application(
    legal_research: FinalLegalResearch
):

    st.markdown(
        "## Application of Law to the Reported Facts"
    )

    st.caption(
        "This section explains how the identified legal provisions relate to the factual allegations. It does not treat disputed allegations as established facts."
    )

    if not legal_research.application:

        st.info(
            "The final analysis did not return a separate application section."
        )

        return

    for index, application in enumerate(
        legal_research.application,
        start=1,
    ):

        with st.expander(
            f"Analysis {index}",
            expanded=True,
        ):

            st.write(
                application
            )


# ============================================================
# REPORT: EVIDENCE
# ============================================================

def render_evidence(
    legal_research: FinalLegalResearch
):

    st.markdown(
        "## Evidentiary Considerations"
    )

    evidence_items = []

    for item in legal_research.uncertainties:

        lower = item.lower()

        if any(
            keyword in lower
            for keyword in [
                "evidence",
                "proof",
                "witness",
                "document",
                "recording",
                "cctv",
                "admissib",
                "disputed",
                "unproven",
            ]
        ):

            evidence_items.append(
                item
            )

    if evidence_items:

        render_bullet_list(
            evidence_items
        )

    else:

        st.info(
            "No dedicated evidentiary findings were returned by the final research stage."
        )


# ============================================================
# REPORT: UNCERTAINTIES
# ============================================================

def render_uncertainties(
    legal_research: FinalLegalResearch
):

    st.markdown(
        "## Uncertainties and Limitations"
    )

    if not legal_research.uncertainties:

        st.info(
            "No additional uncertainties were returned."
        )

        return

    render_bullet_list(
        legal_research.uncertainties
    )

    st.markdown(
        "### Research completeness"
    )

    if (
        legal_research.evidence_completeness
        == "exhaustive_search_no_authority_found"
    ):

        st.success(
            "The final reasoner assessed the search as sufficiently comprehensive for the stated research question."
        )

    else:

        st.warning(
            "The research was based on a limited search and may have missed relevant authority. The absence of a retrieved authority should not be treated as proof that no such authority exists."
        )


# ============================================================
# REPORT: RESEARCH PLAN
# ============================================================

def render_research_methodology(
    research_plan: ResearchPlan
):

    st.markdown(
        "## Research Methodology"
    )

    with st.expander(
        "Research questions",
        expanded=False,
    ):

        render_bullet_list(
            research_plan.research_questions
        )

    with st.expander(
        "Legal concepts investigated",
        expanded=False,
    ):

        render_bullet_list(
            research_plan.legal_concepts
        )

    with st.expander(
        "Actual alleged offences identified",
        expanded=False,
    ):

        render_bullet_list(
            research_plan.actual_alleged_offences
        )

    with st.expander(
        "Retrieved sections considered relevant",
        expanded=False,
    ):

        render_bullet_list(
            research_plan.relevant_retrieved_sections
        )

    with st.expander(
        "Retrieved sections considered irrelevant",
        expanded=False,
    ):

        render_bullet_list(
            research_plan.irrelevant_retrieved_sections
        )

    with st.expander(
        "Historical equivalents",
        expanded=False,
    ):

        if not research_plan.historical_equivalents:

            st.caption(
                "No historical equivalents were identified."
            )

        for item in research_plan.historical_equivalents:

            st.markdown(
                f"**{item.current_concept}**"
            )

            st.write(
                f"{item.historical_provision} "
                f"({item.statute})"
            )

            st.write(
                item.explanation
            )

            st.divider()

    with st.expander(
        "Search queries used",
        expanded=False,
    ):

        for index, query in enumerate(
            research_plan.search_queries,
            start=1,
        ):

            st.write(
                f"{index}. {query}"
            )


# ============================================================
# REPORT: RAW OUTPUT
# ============================================================

def render_raw_output(
    final_result: LegalResearchOutput
):

    st.markdown(
        "## Structured Output"
    )

    st.caption(
        "Machine-readable representation of the final legal research result."
    )

    payload = final_result.model_dump()

    json_str = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        label="Download research report as JSON",
        data=json_str,
        file_name="legal_research_output.json",
        mime="application/json",
        use_container_width=False,
    )

    with st.expander(
        "View raw JSON",
        expanded=False,
    ):

        st.code(
            json_str,
            language="json",
        )


# ============================================================
# REPORT HEADER
# ============================================================

def render_report_header(
    extraction: CaseFactExtraction,
    final_result: LegalResearchOutput,
    selected_items,
):

    st.markdown(
        '<div class="report-title">Indian Legal Research Report</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="report-subtitle">Structured statutory and judicial research based on the facts provided</div>',
        unsafe_allow_html=True,
    )

    lr = final_result.legal_research

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Issues",
            len(lr.issues),
        )

    with col2:

        st.metric(
            "Legal provisions",
            len(lr.applicable_law),
        )

    with col3:

        st.metric(
            "Precedent propositions",
            len(
                lr.precedent_analysis
            ),
        )

    with col4:

        st.metric(
            "Selected authorities",
            len(selected_items),
        )

    with st.expander(
        "Research scope and case summary",
        expanded=True,
    ):

        st.write(
            extraction.summary
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Application header
    # --------------------------------------------------------

    st.markdown(
        '<div class="report-title">Indian Legal Research</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="report-subtitle">Research statutory provisions, judicial authorities and their relevance to a legal problem.</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # API configuration check
    # --------------------------------------------------------

    missing_keys = []

    if not GROQ_API_KEY:
        missing_keys.append(
            "GROQ_API_KEY"
        )

    if not TAVILY_API_KEY:
        missing_keys.append(
            "TAVILY_API_KEY"
        )

    if missing_keys:

        st.error(
            "The application is not configured correctly. "
            "Required environment variables are missing: "
            + ", ".join(missing_keys)
        )

        st.code(
            "export GROQ_API_KEY='your-key'\n"
            "export TAVILY_API_KEY='your-key'",
            language="bash",
        )

        st.stop()

    # --------------------------------------------------------
    # Query input
    # --------------------------------------------------------

    st.markdown(
        "### Legal query"
    )

    raw_query = st.text_area(
        "Describe the legal problem",
        height=220,
        placeholder=(
            "Describe the facts, parties involved, "
            "alleged conduct, procedural history and "
            "the legal question you want researched."
        ),
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(
        [1, 5]
    )

    with col1:

        run_button = st.button(
            "Run research",
            type="primary",
            use_container_width=True,
        )

    with col2:

        st.caption(
            "The system will extract the facts, retrieve statutory material, research judicial authorities, assess the sources and prepare a structured legal research report."
        )

    if not raw_query.strip():

        st.info(
            "Enter a legal problem above to begin."
        )

        return

    if not run_button:

        return

    # --------------------------------------------------------
    # Clear previous results
    # --------------------------------------------------------

    st.session_state.pop(
        "pipeline_results",
        None,
    )

    # --------------------------------------------------------
    # Pipeline progress
    # --------------------------------------------------------

    progress_bar = st.progress(
        0,
        text="Preparing research...",
    )

    try:

        # ====================================================
        # STEP 0
        # ====================================================

        progress_bar.progress(
            5,
            text="Extracting case facts...",
        )

        extraction = extract_case_facts(
            raw_query
        )

        # ====================================================
        # STEP 1
        # ====================================================

        progress_bar.progress(
            20,
            text="Retrieving applicable statutory material...",
        )

        OFFENCES = extraction.alleged_offences
        ALLEGATION = extraction.factual_allegations

        statute_results = (
            retrieve_statutes(
                OFFENCES,
                ALLEGATION,
            )
        )

        statute_context = (
            build_statute_context(
                statute_results
            )
        )

        # ====================================================
        # STEP 2
        # ====================================================

        progress_bar.progress(
            35,
            text="Building legal research plan...",
        )

        research_plan = (
            generate_research_plan(
                OFFENCES,
                ALLEGATION,
                statute_context,
            )
        )

        # ====================================================
        # STEP 3
        # ====================================================

        progress_bar.progress(
            50,
            text="Searching judicial authorities...",
        )

        tavily_results = (
            search_tavily_parallel(
                research_plan.search_queries
            )
        )

        tavily_results = (
            deduplicate_results(
                tavily_results
            )
        )

        candidates = tavily_results[
            :MAX_CANDIDATES_TO_RANK
        ]

        # ====================================================
        # STEP 4
        # ====================================================

        progress_bar.progress(
            70,
            text="Assessing judicial research candidates...",
        )

        if candidates:

            ranking = rank_precedents(
                OFFENCES,
                ALLEGATION,
                research_plan,
                candidates,
            )

            selected_items = (
                select_final_candidates(
                    candidates,
                    ranking,
                )
            )

        else:

            ranking = None
            selected_items = []

        # ====================================================
        # STEP 5
        # ====================================================

        progress_bar.progress(
            90,
            text="Preparing final legal analysis...",
        )

        final_result = (
            final_legal_reasoning(
                OFFENCES,
                ALLEGATION,
                statute_context,
                research_plan,
                selected_items,
                raw_query,
            )
        )

        progress_bar.progress(
            100,
            text="Research complete.",
        )

        # ----------------------------------------------------
        # Store everything
        # ----------------------------------------------------

        st.session_state.pipeline_results = {
            "raw_query": raw_query,
            "extraction": extraction,
            "statute_results": statute_results,
            "research_plan": research_plan,
            "candidates": candidates,
            "ranking": ranking,
            "selected_items": selected_items,
            "final_result": final_result,
        }

        time.sleep(0.2)

        progress_bar.empty()

    except Exception as exc:

        progress_bar.empty()

        st.error(
            "The research pipeline encountered an error."
        )

        with st.expander(
            "Technical error details",
            expanded=False,
        ):

            st.exception(exc)

        return

    # ========================================================
    # FINAL REPORT
    # ========================================================

    results = (
        st.session_state.pipeline_results
    )

    extraction = results[
        "extraction"
    ]

    final_result = results[
        "final_result"
    ]

    research_plan = results[
        "research_plan"
    ]

    statute_results = results[
        "statute_results"
    ]

    candidates = results[
        "candidates"
    ]

    ranking = results[
        "ranking"
    ]

    selected_items = results[
        "selected_items"
    ]

    legal_research = (
        final_result.legal_research
    )

    st.divider()

    # --------------------------------------------------------
    # Report header
    # --------------------------------------------------------

    render_report_header(
        extraction,
        final_result,
        selected_items,
    )

    # --------------------------------------------------------
    # Main report navigation
    # --------------------------------------------------------

    tabs = st.tabs(
        [
            "Overview",
            "Issues",
            "Applicable Law",
            "Judicial Research",
            "Legal Analysis",
            "Evidence & Limitations",
            "Research Method",
            "Structured Output",
        ]
    )

    # ========================================================
    # TAB 1 — OVERVIEW
    # ========================================================

    with tabs[0]:

        render_case_overview(
            extraction
        )

    # ========================================================
    # TAB 2 — ISSUES
    # ========================================================

    with tabs[1]:

        render_issues(
            extraction,
            legal_research,
        )

    # ========================================================
    # TAB 3 — APPLICABLE LAW
    # ========================================================

    with tabs[2]:

        render_statutory_law(
            statute_results,
            legal_research,
        )

    # ========================================================
    # TAB 4 — JUDICIAL RESEARCH
    # ========================================================

    with tabs[3]:

        render_precedent_research(
            candidates,
            ranking,
            selected_items,
        )

    # ========================================================
    # TAB 5 — LEGAL ANALYSIS
    # ========================================================

    with tabs[4]:

        render_precedent_analysis(
            legal_research
        )

        st.divider()

        render_application(
            legal_research
        )

    # ========================================================
    # TAB 6 — EVIDENCE & LIMITATIONS
    # ========================================================

    with tabs[5]:

        render_evidence(
            legal_research
        )

        st.divider()

        render_uncertainties(
            legal_research
        )

    # ========================================================
    # TAB 7 — RESEARCH METHOD
    # ========================================================

    with tabs[6]:

        render_research_methodology(
            research_plan
        )

    # ========================================================
    # TAB 8 — STRUCTURED OUTPUT
    # ========================================================

    with tabs[7]:

        render_raw_output(
            final_result
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()