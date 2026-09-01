"""
core/reasoning.py

Step 5 of the pipeline: rebuilding full precedent context and the
final Groq 120B legal-reasoning call. This is research-pipeline.py's
version of final_legal_reasoning (no original_user_query kwarg).

cli/agent.py needs a variant that also takes the raw user query so
the reasoner can tailor its analysis to the exact question asked --
that variant is NOT duplicated logic (it's a genuinely different
prompt/signature), so it stays local to cli/agent.py rather than
living here.
"""

from typing import Any, Dict, List

from core.config import REASONING_MODEL, MAX_REASONER_CANDIDATE_CHARS
from core.llm import groq_structured
from core.models import LegalResearchOutput, ResearchPlan


def build_final_precedent_context(selected_items):
    """
    Reconstruct original Tavily candidate information.

    The final 120B receives:

        original candidate
        +
        20B scores + source_type

    It does NOT receive only the 20B interpretation, and it does
    NOT receive candidates that were dropped as redundant --
    that trimming is what keeps this prompt under the TPM limit.
    """

    if not selected_items:
        return """
NO PRECEDENT CANDIDATES WERE RETAINED.

The final reasoner MUST NOT invent precedent.
"""

    blocks = []

    for position, item in enumerate(selected_items, start=1):
        candidate = item["candidate"]
        assessment = item["ranker_assessment"]

        trimmed_content = candidate["content"][:MAX_REASONER_CANDIDATE_CHARS]

        if assessment:
            score_block = f"""
Source type (20B classified, Python-capped):
{assessment.source_type}

20B relevance score:
{assessment.relevance_score}

20B judicial authority score (post-cap):
{assessment.judicial_authority_score}

20B keep recommendation:
{assessment.keep_for_reasoner}

20B reason:
{assessment.reason}
""".strip()

        else:
            fallback_type = item.get("fallback_source_type", "unknown")
            score_block = f"""
20B assessment: UNAVAILABLE.

This candidate was selected using Python's deterministic
source-quality fallback. Inferred source type (heuristic,
not model-verified): {fallback_type}
""".strip()

        blocks.append(
            f"""
============================================================
PRECEDENT CANDIDATE {position}
============================================================

Original candidate ID:
{candidate['candidate_id']}

Found through search query:
{candidate['search_query']}

Title:
{candidate['title']}

URL:
{candidate['url']}

Domain:
{candidate['domain']}

Original compressed Tavily content (trimmed):
{trimmed_content}

{score_block}
""".strip()
        )

    return "\n\n".join(blocks)



def final_legal_reasoning(
    offence_text: str,
    allegation_text: str,
    statute_context: str,
    research_plan: ResearchPlan,
    selected_items,
):
    """
    Final legal reasoning stage.

    NOTE ON PROMPT SIZE: this function deliberately does NOT
    resend `irrelevant_retrieved_sections`, `research_questions`,
    or `legal_concepts` from the research plan. Those fields did
    their job during planning/ranking; re-sending them here was
    the main driver of TPM overruns and adds no new evidence for
    the reasoner. Only relevant sections + historical equivalents
    are carried forward.
    """

    precedent_context = build_final_precedent_context(selected_items)

    system_prompt = """
You are the FINAL LEGAL REASONER in an Indian legal research system.

You receive:

1. Factual allegations.
2. Alleged offences.
3. ORIGINAL BNS / BNSS / BSA statute text.
4. The relevant sections and historical equivalents identified
   by the research plan.
5. Python-reconstructed judicial research candidates (top 3-5,
   already deduplicated and capped by source type).
6. 20B candidate assessments, including source_type.

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
CASE
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

Do not invent authorities simply to make the answer appear complete.
"""

    return groq_structured(
        model=REASONING_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_class=LegalResearchOutput,
        max_retries=2,
    )

