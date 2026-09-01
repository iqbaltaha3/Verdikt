"""
core/ranking.py

Step 4 of the pipeline: deterministic fallback scoring, the Groq
20B recall-first ranker + source classifier, and Python's final
weighted candidate selection. Identical logic in agent.py and
research-pipeline.py (app.py has its own divergent version left
in ui/app.py).
"""

from typing import Any, Dict, List, Optional

from core.config import (
    RANKER_MODEL,
    SOURCE_TYPE_AUTHORITY_CAPS,
    MAX_REASONER_CANDIDATE_CHARS,
    MIN_SELECTED_PRECEDENTS,
    MAX_SELECTED_PRECEDENTS,
)
from core.llm import groq_structured
from core.models import PrecedentRanking, ResearchPlan


def source_quality_score(candidate: Dict[str, Any]) -> int:
    """
    Deterministic fallback score.

    Used only if the 20B ranking fails.

    This is NOT a semantic legal judgment.

    It estimates source quality from URL/title/content
    signals.
    """

    title = (candidate.get("title", "") or "").lower()
    domain = (candidate.get("domain", "") or "").lower()
    content = (candidate.get("content", "") or "").lower()
    text = title + " " + domain + " " + content

    score = 0

    # Indian judiciary
    if "sci.gov.in" in domain:
        score += 60
    elif ".hc." in domain:
        score += 50
    elif "highcourt" in domain:
        score += 50
    elif domain.endswith(".gov.in"):
        score += 35

    # Indian legal repositories
    Indian_legal_domains = [
        "indiankanoon.org",
        "sci.gov.in",
        "hcservices.ecourts.gov.in",
        "main.sci.gov.in",
        "livelaw.in",
        "barandbench.com",
        "casemine.com",
    ]

    for legal_domain in Indian_legal_domains:
        if legal_domain in domain:
            score += 20
            break

    # Judicial indicators
    judicial_terms = [
        "judgment",
        "judgement",
        "order",
        "petition",
        "appeal",
        "criminal appeal",
        "writ petition",
        "supreme court",
        "high court",
    ]

    for term in judicial_terms:
        if term in text:
            score += 5

    # Foreign-law signals
    foreign_terms = [
        "united states",
        "u.s.",
        "us law",
        "american law",
        "new york",
        "california",
        "england and wales",
        "uk law",
    ]

    for term in foreign_terms:
        if term in text:
            score -= 30

    # Generic article indicators
    generic_terms = [
        "understanding",
        "what is",
        "explained",
        "guide",
        "legal rights",
        "legal protection",
        "blog",
    ]

    for term in generic_terms:
        if term in title:
            score -= 8

    # Tavily retrieval score
    tavily_score = candidate.get("tavily_score")
    if isinstance(tavily_score, (int, float)):
        score += int(max(0, min(20, tavily_score * 20)))

    return max(0, min(100, score))



def infer_source_type_fallback(candidate: Dict[str, Any]) -> str:
    """
    Used only when the 20B ranker is unavailable and Python
    must fall back to deterministic scoring. Provides a rough
    source_type so the same authority-cap logic still applies
    even without an LLM assessment.
    """

    domain = (candidate.get("domain", "") or "").lower()
    title = (candidate.get("title", "") or "").lower()

    if "sci.gov.in" in domain or "supreme court" in title:
        return "supreme_court_judgment"

    if ".hc." in domain or "highcourt" in domain or "high court" in title:
        return "high_court_judgment"

    if "indiankanoon.org" in domain:
        return "judgment_repository"

    if any(
        social in domain
        for social in (
            "instagram.com",
            "youtube.com",
            "facebook.com",
            "twitter.com",
            "x.com",
            "tiktok.com",
        )
    ):
        return "social_media"

    if "youtube.com" in domain:
        return "video"

    if any(
        legal_site in domain
        for legal_site in (
            "livelaw.in",
            "barandbench.com",
            "casemine.com",
        )
    ):
        return "legal_commentary"

    if "blog" in domain or "blog" in title:
        return "legal_blog"

    return "unknown"



def apply_authority_cap(
    source_type: str,
    judicial_authority_score: int,
) -> int:
    """
    Hard ceiling enforced in Python. A legal_blog or
    social_media hit cannot out-rank an actual judgment simply
    because an LLM assigned it an inflated numerical score.
    """

    cap = SOURCE_TYPE_AUTHORITY_CAPS.get(
        source_type,
        SOURCE_TYPE_AUTHORITY_CAPS["unknown"],
    )

    return min(judicial_authority_score, cap)



def build_ranker_candidates(candidates: List[Dict[str, Any]]):
    """
    Build compact candidate context for 20B.
    """

    blocks = []

    for candidate in candidates:
        blocks.append(
            f"""
============================================================
CANDIDATE {candidate['candidate_id']}
============================================================

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



def rank_precedents(
    offence_text: str,
    allegation_text: str,
    research_plan: ResearchPlan,
    candidates: List[Dict[str, Any]],
):
    """
    Ask 20B to assess every candidate AND classify its source
    type.

    20B does NOT make final selection.

    Batched into chunks of 5 to avoid output token truncation.
    """

    if not candidates:
        raise RuntimeError(
            "No Tavily candidates available for ranking."
        )

    BATCH_SIZE = 5
    all_assessments = []

    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i : i + BATCH_SIZE]
        candidate_text = build_ranker_candidates(batch)

        system_prompt = """
You are the PRECEDENT CANDIDATE SCORER AND SOURCE CLASSIFIER in an
Indian legal research pipeline.

You are NOT the final legal reasoner.

You are NOT the final selector.

Your job is to assess every search candidate and classify its
source type.

============================================================
CORE PRINCIPLE
============================================================

This is a RECALL-FIRST stage.

Do NOT aggressively reject candidates.

It is preferable to send a few weak candidates to the final reasoner
than to accidentally discard a potentially important judicial
authority.

============================================================
SOURCE TYPE (REQUIRED FIELD)
============================================================

Classify every candidate into EXACTLY ONE of:

supreme_court_judgment
high_court_judgment
tribunal_decision
judgment_repository
legal_commentary
legal_blog
news_report
video
social_media
generic_information
unknown

A legal blog is NOT itself a judicial precedent.

A news article is NOT itself a judicial precedent.

A judgment hosted by a legal repository MAY contain a judicial
precedent -- classify it as judgment_repository, not as the
judgment itself, unless the candidate content clearly reproduces
the actual judgment text.

This field is used downstream to CAP how much authority Python
will assign the candidate, regardless of your numerical scores.
Classify honestly -- inflating source_type to force a candidate
through will not help the reasoner and will be caught by the cap.

============================================================
REDUNDANCY (REQUIRED FIELD: redundant_with)
============================================================

If a candidate makes essentially the same point as another
candidate already assessed (e.g. two different blogs both
explaining the same section in similar terms), set
`redundant_with` to the candidate_id of the OTHER, STRONGER
candidate making that same point. Leave it null if the
candidate is not redundant with anything else in this batch.

============================================================
SCORING
============================================================

For EVERY candidate return:

candidate_id
source_type
relevance_score        0-100
source_quality_score   0-100
judicial_authority_score   0-100
factual_similarity_score   0-100
keep_for_reasoner
redundant_with (nullable)
reason

============================================================
KEEP RULE
============================================================

Keep plausible Indian judicial material.

Do not require perfect factual similarity.

A case can be legally important even if the facts are not identical.

============================================================
IMPORTANT
============================================================

You MUST return one assessment for EVERY supplied candidate.

Do not omit candidates.

Do not invent candidate IDs.

Return ONLY the schema-defined JSON.
"""

        # FIX: Removed legal_concepts and research_questions from ranker
        # context to reduce prompt size and avoid 429 rate limits.
        historical_context = [
            item.model_dump() for item in research_plan.historical_equivalents
        ]

        user_prompt = f"""
============================================================
CASE
============================================================

OFFENCES:

{offence_text}

ALLEGATION:

{allegation_text}

============================================================
RESEARCH PLAN (TRIMMED FOR RANKER)
============================================================

Actual alleged offences:
{json.dumps(research_plan.actual_alleged_offences, indent=2, ensure_ascii=False)}

Relevant retrieved sections:
{json.dumps(research_plan.relevant_retrieved_sections, indent=2, ensure_ascii=False)}

Historical equivalents:
{json.dumps(historical_context, indent=2, ensure_ascii=False)}

============================================================
CANDIDATES (BATCH {i // BATCH_SIZE + 1} of {(len(candidates) - 1) // BATCH_SIZE + 1})
============================================================

{candidate_text}

============================================================
TASK
============================================================

Assess EVERY candidate in this batch.

Classify each candidate's source_type.

Flag redundant candidates via redundant_with.

Use high recall.

Do not invent information not contained in the candidate.

Return exactly one CandidateAssessment per candidate in this batch.
"""

        ranking = groq_structured(
            model=RANKER_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_class=PrecedentRanking,
            max_retries=2,
        )

        all_assessments.extend(ranking.assessments)
        print(
            f"  [Ranker] Processed batch "
            f"{i // BATCH_SIZE + 1}/{(len(candidates) - 1) // BATCH_SIZE + 1}"
        )

    merged = PrecedentRanking(assessments=all_assessments)

    expected_ids = {c["candidate_id"] for c in candidates}
    returned_ids = {a.candidate_id for a in merged.assessments}
    missing_ids = expected_ids - returned_ids

    if missing_ids:
        raise RuntimeError(
            f"20B failed to assess every candidate. "
            f"Missing IDs: {sorted(missing_ids)}"
        )

    for assessment in merged.assessments:
        assessment.judicial_authority_score = apply_authority_cap(
            assessment.source_type,
            assessment.judicial_authority_score,
        )

    return merged



def select_final_candidates(
    candidates: List[Dict[str, Any]],
    ranking: Optional[PrecedentRanking],
):
    """
    Python makes the final selection.

    Weighted score:

        Judicial authority    35%   (already capped by source_type)
        Relevance              35%
        Source quality         15%
        Factual similarity     15%

    Redundant candidates (per the ranker's redundant_with field)
    are dropped in favour of the stronger candidate they duplicate,
    so the final reasoner isn't handed four sources all making the
    same point.

    If ranking is unavailable, deterministic source-quality
    scoring plus a rough source_type inference is used instead.
    """

    candidate_map = {c["candidate_id"]: c for c in candidates}
    scored = []

    if ranking:
        redundant_ids = {
            assessment.candidate_id
            for assessment in ranking.assessments
            if assessment.redundant_with is not None
        }

        for assessment in ranking.assessments:
            candidate = candidate_map.get(assessment.candidate_id)
            if not candidate:
                continue

            if (
                assessment.candidate_id in redundant_ids
                and assessment.redundant_with in candidate_map
            ):
                continue

            combined_score = (
                assessment.judicial_authority_score * 0.35
                + assessment.relevance_score * 0.35
                + assessment.source_quality_score * 0.15
                + assessment.factual_similarity_score * 0.15
            )

            scored.append(
                {
                    "candidate": candidate,
                    "combined_score": combined_score,
                    "ranker_assessment": assessment,
                }
            )

    if not scored:
        print()
        print("[Python fallback]")
        print("20B produced no usable ranking.")
        print("Using deterministic source-quality scoring.")

        for candidate in candidates:
            scored.append(
                {
                    "candidate": candidate,
                    "combined_score": source_quality_score(candidate),
                    "ranker_assessment": None,
                    "fallback_source_type": infer_source_type_fallback(
                        candidate
                    ),
                }
            )

    scored.sort(key=lambda item: item["combined_score"], reverse=True)

    judicial_like = []

    for item in scored:
        assessment = item["ranker_assessment"]
        if assessment is None:
            continue

        if (
            assessment.judicial_authority_score >= 40
            and assessment.relevance_score >= 40
        ):
            judicial_like.append(item)

    if len(judicial_like) >= MIN_SELECTED_PRECEDENTS:
        selected = judicial_like[:MAX_SELECTED_PRECEDENTS]
    else:
        selected = scored[:MAX_SELECTED_PRECEDENTS]

    return selected

