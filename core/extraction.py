"""
core/extraction.py

Step 0, used only by cli/agent.py: turns a raw, messy free-form
legal query into structured case facts (offences/allegations/etc.)
before the rest of the pipeline runs. This stage does not exist in
research-pipeline.py or app.py.
"""

from core.config import EXTRACTOR_MODEL
from core.llm import groq_structured
from core.models import CaseFactExtraction


def extract_case_facts(raw_query: str) -> CaseFactExtraction:
    """
    Takes any raw legal query and extracts structured factual
    information, offences, and allegations.
    """

    system_prompt = """
You are a CASE FACT EXTRACTOR in an Indian legal research system.

Your job is to read a raw legal query -- which may be messy,
conversational, or incomplete -- and extract structured factual
information from it.

============================================================
TASK
============================================================

1. Summarize the legal problem in 2-3 sentences.
2. Identify the parties involved.
3. Extract the factual allegations (what allegedly happened).
4. Identify the alleged offences or legal wrongs.
5. List the key legal issues raised.
6. Note any facts that are disputed or denied.
7. Capture any procedural context if mentioned (court, bail,
   complaint stage, etc.).

============================================================
RULES
============================================================

- Do NOT invent facts not present in the query.
- If the query is vague, state what is unclear.
- Distinguish between alleged facts and established facts.
- The alleged_offences field should be a short paragraph or
  list of offences that appear to be involved, phrased as
  "Alleged [offence] by [party]" where possible.
- The factual_allegations field should be a plain-language
  narrative of what the complainant/petitioner alleges.
- Keep the output strictly within the schema. No extra fields.

============================================================
OUTPUT
============================================================

Return ONLY the schema-defined JSON.
"""

    user_prompt = f"""
============================================================
RAW LEGAL QUERY
============================================================

{raw_query}

============================================================
TASK
============================================================

Extract structured facts from the query above.

Return exactly the schema-defined structure.
"""

    return groq_structured(
        model=EXTRACTOR_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_class=CaseFactExtraction,
        max_retries=2,
    )

