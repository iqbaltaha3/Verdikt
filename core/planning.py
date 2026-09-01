"""
core/planning.py

Step 2 of the pipeline: the Groq 120B legal research planner.
Identical logic in agent.py and research-pipeline.py.
"""

from core.config import PLANNER_MODEL
from core.llm import groq_structured
from core.models import ResearchPlan


def generate_research_plan(
    offence_text: str,
    allegation_text: str,
    statute_context: str,
):

    system_prompt = """
You are the LEGAL RESEARCH PLANNER of an Indian legal RAG system.

You are NOT the final legal reasoner.

Your task is to construct a research plan from:

1. Alleged offences.
2. Factual allegations.
3. ORIGINAL retrieved BNS / BNSS / BSA statutory text.

============================================================
ABSOLUTE OUTPUT CONTRACT
============================================================

You MUST return ONLY these seven fields:

1. actual_alleged_offences
2. relevant_retrieved_sections
3. irrelevant_retrieved_sections
4. historical_equivalents
5. legal_concepts
6. research_questions
7. search_queries

DO NOT create any other fields.

In particular, DO NOT create:

- current_law_mapping
- offence_or_issue
- current_law_sections
- explanation outside the required structures
- extra metadata
- confidence
- notes

The field:

historical_equivalents

MUST contain objects with EXACTLY these four fields:

- current_concept
- historical_provision
- statute
- explanation

============================================================
TASK
============================================================

Analyze the allegations first.

Then determine:

1. Actual alleged offences.

2. Which retrieved sections are relevant.

3. Which retrieved sections are irrelevant or weakly related.

4. Historical provisions that are genuinely relevant to the
   legal concepts.

5. Legal concepts requiring judicial interpretation.

6. Research questions.

7. EXACTLY FIVE precedent-search queries.

============================================================
CURRENT LAW
============================================================

Use the supplied statutory text as the primary evidence for
current-law identification.

Do not assume that every retrieved section is relevant.

Some retrieved sections may be procedural rather than substantive.

Some may be completely irrelevant.

============================================================
HISTORICAL LAW
============================================================

This is extremely important.

NEVER infer the meaning of an old IPC / CrPC / Indian Evidence Act
section merely from its section number.

The historical equivalent must be based on the actual legal concept.

For example, criminal intimidation involves distinguishing the
historical provision defining the offence from the historical
provision prescribing punishment.

Do not collapse different provisions merely because they concern
the same offence.

If uncertain, explicitly state the uncertainty in the explanation.

Do not invent equivalence.

============================================================
SEARCH QUERIES
============================================================

Return EXACTLY FIVE queries.

The queries should be driven by:

FACTS
+
LEGAL ISSUE
+
OFFENCE
+
JUDICIAL QUESTION

Do NOT simply search section numbers.

Do NOT restrict every query to the modern BNS/BNSS section
number. Older Indian precedent overwhelmingly discusses the
HISTORICAL provision (IPC / CrPC / Indian Evidence Act), not
the new BNS/BNSS numbering. At least two of your five queries
should be framed around the historical provision and legal
concept rather than the new section number, so that precedent
search is not biased toward only recent explanatory pages.

Bad:

"BNSS 223 case law"

Better:

"Supreme Court complaint against public servant magistrate
cognizance procedural safeguards"

Bad:

"IPC 506 judgments"

Better:

"Supreme Court criminal intimidation threat to detain person
police officer Section 506 IPC"

Searches should preferentially target:

1. Supreme Court judgments
2. High Court judgments
3. Indian judicial orders
4. Reliable judgment repositories

============================================================
IMPORTANT
============================================================

A search query is not itself evidence.

The final reasoner must later verify whether search results
actually contain judicial authorities.

Do not fabricate case names.

============================================================
OUTPUT DISCIPLINE
============================================================

Return exactly the schema-defined structure.

Do not return explanations outside the schema.

Do not add fields.
"""

    user_prompt = f"""
============================================================
CASE
============================================================

OFFENCES:

{offence_text}

ALLEGATION:

{allegation_text}

============================================================
ORIGINAL RETRIEVED STATUTES
============================================================

{statute_context}

============================================================
TASK
============================================================

Determine:

- actual alleged offences
- relevant retrieved sections
- irrelevant retrieved sections
- historical equivalents
- legal concepts
- research questions
- exactly five search queries

IMPORTANT:

The output MUST match the supplied schema exactly.

The field historical_equivalents must contain objects with:

current_concept
historical_provision
statute
explanation

Do not create current_law_mapping.

Return EXACTLY FIVE search_queries, and ensure at least two are
framed around the historical provision rather than the new
section number.
"""

    plan = groq_structured(
        model=PLANNER_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_class=ResearchPlan,
        max_retries=2,
    )

    plan.search_queries = [
        q.strip() for q in plan.search_queries if q.strip()
    ]

    if len(plan.search_queries) != N_SEARCH_QUERIES:
        raise RuntimeError(
            f"Planner returned {len(plan.search_queries)} queries. "
            f"Expected exactly {N_SEARCH_QUERIES}."
        )

    return plan

