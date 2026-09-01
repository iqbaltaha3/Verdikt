"""
core/models.py

All Pydantic schemas shared across the CLI pipeline, the CLI
agent, and the Streamlit UI (verified byte-identical in logic
across agent.py, research-pipeline.py and app.py before being
centralized here -- no behavior was changed).

CaseFactExtraction is the one exception: it is only used by the
case-fact-extraction stage (core/extraction.py, used by
cli/agent.py), since app.py and research-pipeline.py never had
that stage.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CaseFactExtraction(BaseModel):
    """
    Structured output from the case fact extractor.
    """

    summary: str = Field(
        description="A concise 2-3 sentence summary of the legal problem."
    )
    parties_involved: List[str] = Field(
        description="List of parties (e.g. complainant, accused, police officer, magistrate)."
    )
    factual_allegations: str = Field(
        description="The factual narrative of what allegedly happened, in plain language."
    )
    alleged_offences: str = Field(
        description="The offences or legal wrongs that appear to be involved, based on the facts."
    )
    legal_issues: List[str] = Field(
        description="Key legal questions or issues raised by the facts."
    )
    disputed_facts: List[str] = Field(
        description="Facts that are contested or denied by any party."
    )
    procedural_context: Optional[str] = Field(
        default=None,
        description="Any procedural information (court, stage of case, bail, etc.) if mentioned.",
    )


class HistoricalEquivalent(BaseModel):
    """
    Historical provision relevant to the current legal concept.
    """

    current_concept: str
    historical_provision: str
    statute: str
    explanation: str


class ResearchPlan(BaseModel):
    """
    Exact output expected from the 120B planner.
    """

    actual_alleged_offences: List[str]
    relevant_retrieved_sections: List[str]
    irrelevant_retrieved_sections: List[str]
    historical_equivalents: List[HistoricalEquivalent]
    legal_concepts: List[str]
    research_questions: List[str]
    search_queries: List[str]


# Categorical source classification. This is the key addition:
# instead of trusting only a numerical judicial_authority_score,
# the ranker must bucket every candidate into one of these types,
# and Python enforces a hard authority cap per bucket.
SourceType = Literal[
    "supreme_court_judgment",
    "high_court_judgment",
    "tribunal_decision",
    "judgment_repository",
    "legal_commentary",
    "legal_blog",
    "news_report",
    "video",
    "social_media",
    "generic_information",
    "unknown",
]


class CandidateAssessment(BaseModel):
    """
    20B assessment of one search candidate.
    """

    candidate_id: int
    source_type: SourceType
    relevance_score: int = Field(ge=0, le=100)
    source_quality_score: int = Field(ge=0, le=100)
    judicial_authority_score: int = Field(ge=0, le=100)
    factual_similarity_score: int = Field(ge=0, le=100)
    keep_for_reasoner: bool
    redundant_with: Optional[int] = None
    reason: str


class PrecedentRanking(BaseModel):
    """
    20B returns one assessment for every candidate.
    """

    assessments: List[CandidateAssessment]


class PrecedentAnalysisItem(BaseModel):
    """
    One precedent-related claim, with its own confidence and
    evidentiary basis.
    """

    claim: str
    support: Literal[
        "statutory_text",
        "judicial_authority",
        "secondary_source",
        "unverified",
    ]
    confidence: Literal[
        "high",
        "moderate",
        "low",
    ]


class FinalLegalResearch(BaseModel):
    """
    Structured legal research section.
    """

    issues: List[str]
    applicable_law: List[str]
    precedent_analysis: List[PrecedentAnalysisItem]
    application: List[str]
    uncertainties: List[str]
    research_queries: List[str]
    evidence_completeness: Literal[
        "exhaustive_search_no_authority_found",
        "limited_search_may_have_missed_authority",
    ]


class LegalResearchOutput(BaseModel):
    """
    Final structured legal research response.
    """

    laws: str
    sections_applied: str
    precedents: str
    legal_research: FinalLegalResearch

