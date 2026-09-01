"""
core/retrieval.py

Real hybrid BM25 + Chroma statute retrieval. Identical logic in
agent.py and research-pipeline.py. This is the "real" retrieval
that ui/app.py's old retrieve_statutes_stub() has been replaced
with -- app.py no longer fakes statute lookups with keyword
matching, it uses this module against the real Chroma DB just
like the CLI tools do.
"""

from vectorstore.query_vector_db import hybrid_search

from core.config import N_STATUTE_RESULTS, MAX_STATUTE_TEXT_CHARS


def retrieve_statutes(
    offence_text: str,
    allegation_text: str,
    collection,
    bm25,
    corpus_ids,
):
    """
    Hybrid BM25 + Chroma retrieval.
    """

    query = f"""
OFFENCES:

{offence_text}

ALLEGATION:

{allegation_text}
""".strip()

    results = hybrid_search(
        query,
        collection,
        bm25,
        corpus_ids,
        n_results=N_STATUTE_RESULTS,
    )
    return results



def build_statute_context(results, collection):
    """
    Retrieve the original full statute text from Chroma.

    The planner and final reasoner receive this original text,
    not merely the hybrid search snippet.
    """

    ids = [result["id"] for result in results]
    fetched = collection.get(ids=ids)

    full_text_by_id = dict(zip(fetched["ids"], fetched["documents"]))

    blocks = []

    for result in results:
        metadata = result["metadata"]
        full_text = full_text_by_id.get(
            result["id"], result.get("snippet", "")
        )

        full_text = full_text[:MAX_STATUTE_TEXT_CHARS]

        block = f"""
ACT:
{metadata.get("act", "")}

SECTION:
§{metadata.get("section_number", "")}

CHAPTER:
{metadata.get("chapter_title", "")}

STATUTORY TEXT:
{full_text}
""".strip()

        blocks.append(block)

    return "\n\n" + (
        "\n\n" + "=" * 70 + "\n\n"
    ).join(blocks)

