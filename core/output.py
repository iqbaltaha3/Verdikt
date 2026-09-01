"""
core/output.py

Console pretty-printers for each pipeline stage, plus saving the
final result to JSON. Byte-identical in agent.py and
research-pipeline.py (print_extraction is agent-only, used by
cli/agent.py after its extra extraction stage).
"""

import json

from core.models import CaseFactExtraction, LegalResearchOutput, ResearchPlan, PrecedentRanking


def print_statutes(results):
    print()
    print("Retrieved statutes:")
    for result in results:
        metadata = result["metadata"]
        print(
            f"  - {metadata.get('act', '')} "
            f"§{metadata.get('section_number', '')} "
            f"({metadata.get('chapter_title', '')})"
        )



def print_plan(plan: ResearchPlan):
    print()
    print("Actual alleged offences:")
    for item in plan.actual_alleged_offences:
        print(f"  - {item}")

    print()
    print("Relevant retrieved sections:")
    for item in plan.relevant_retrieved_sections:
        print(f"  - {item}")

    print()
    print("Irrelevant retrieved sections:")
    for item in plan.irrelevant_retrieved_sections:
        print(f"  - {item}")

    print()
    print("Historical equivalents:")
    for item in plan.historical_equivalents:
        print(f"  - {item.current_concept}")
        print(f"    Historical: {item.historical_provision}")
        print(f"    Statute: {item.statute}")
        print(f"    Reason: {item.explanation}")

    print()
    print("Legal concepts:")
    for item in plan.legal_concepts:
        print(f"  - {item}")

    print()
    print("Research questions:")
    for item in plan.research_questions:
        print(f"  - {item}")

    print()
    print("Generated search queries:")
    for index, query in enumerate(plan.search_queries, start=1):
        print(f"  {index}. {query}")



def print_candidates(candidates):
    print()
    print("Candidates available to 20B:")
    for candidate in candidates:
        print()
        print(f"  [{candidate['candidate_id']}] {candidate['title']}")
        print(f"      Domain: {candidate['domain']}")
        print(f"      URL: {candidate['url']}")
        print(f"      Query: {candidate['search_query']}")
        print(f"      Tavily score: {candidate['tavily_score']}")



def print_ranking(ranking):
    print()
    print("20B candidate assessments:")
    if ranking is None:
        print("  No 20B ranking available.")
        return

    for assessment in ranking.assessments:
        print()
        print(f"  Candidate {assessment.candidate_id}")
        print(f"      Source type: {assessment.source_type}")
        print(f"      Relevance: {assessment.relevance_score}")
        print(f"      Source quality: {assessment.source_quality_score}")
        print(
            f"      Judicial authority (post-cap): "
            f"{assessment.judicial_authority_score}"
        )
        print(f"      Factual similarity: {assessment.factual_similarity_score}")
        print(f"      Keep: {assessment.keep_for_reasoner}")
        print(f"      Redundant with: {assessment.redundant_with}")
        print(f"      Reason: {assessment.reason}")



def print_final_candidates(selected_items):
    print()
    print("Final candidates passed to 120B:")
    if not selected_items:
        print("  NONE")
        return

    for index, item in enumerate(selected_items, start=1):
        candidate = item["candidate"]
        print()
        print(f"  [{index}] {candidate['title']}")
        print(f"      URL: {candidate['url']}")
        print(f"      Domain: {candidate['domain']}")
        print(f"      Python combined score: {item['combined_score']:.2f}")

        assessment = item["ranker_assessment"]
        if assessment:
            print(f"      Source type: {assessment.source_type}")
            print(f"      Relevance: {assessment.relevance_score}")
            print(
                f"      Judicial authority (post-cap): "
                f"{assessment.judicial_authority_score}"
            )



def print_extraction(extraction: CaseFactExtraction):
    print()
    print("=" * 70)
    print("STEP 0 — CASE FACT EXTRACTION")
    print("=" * 70)
    print(f"Summary: {extraction.summary}")
    print()
    print("Parties involved:")
    for party in extraction.parties_involved:
        print(f"  - {party}")
    print()
    print("Factual allegations:")
    print(f"  {extraction.factual_allegations}")
    print()
    print("Alleged offences:")
    print(f"  {extraction.alleged_offences}")
    print()
    print("Legal issues:")
    for issue in extraction.legal_issues:
        print(f"  - {issue}")
    print()
    print("Disputed facts:")
    for fact in extraction.disputed_facts:
        print(f"  - {fact}")
    if extraction.procedural_context:
        print()
        print(f"Procedural context: {extraction.procedural_context}")



def save_final_output(
    final_result: LegalResearchOutput,
    filename: str = "legal_research_output.json",
):
    payload = final_result.model_dump()
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print()
    print(f"Saved final output to: {filename}")

