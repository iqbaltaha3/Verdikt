"""
query_vector_db.py

Combines BM25 keyword search with Chroma vector search using Reciprocal
Rank Fusion (RRF). Fixes cases where the embedding model has no semantic
bridge between a query term and the statute's actual wording -- e.g. "FIR"
vs. "Information in cognizable cases" -- since BM25 will still catch exact
keyword overlap even when the vector model misses it.

WHY RRF AND NOT A WEIGHTED SCORE BLEND
---------------------------------------
BM25 scores and vector cosine/L2 distances live on completely different,
unbounded scales, so combining them directly (e.g. 0.5*bm25 + 0.5*vector)
requires fragile manual score normalization that breaks as your corpus
changes. RRF sidesteps this entirely -- it only looks at each result's
*rank position* in each list, not its raw score, so no tuning is needed:

    RRF_score(doc) = sum( 1 / (k + rank_in_list) )  for each list it appears in

k=60 is the standard constant from the original RRF paper; it just
softens the impact of rank 1 vs rank 2 slightly.

USAGE
-----
    pip install chromadb rank_bm25
    python hybrid_query_vector_db.py
"""

import csv
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

csv.field_size_limit(sys.maxsize)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "indian_criminal_law"
CSV_FILES = [
    str(BASE_DIR.parent / "data" / "BNS_sections.csv"),
    str(BASE_DIR.parent / "data" / "BNSS_sections.csv"),
    str(BASE_DIR.parent / "data" / "BSA_sections.csv"),
]

RRF_K = 60
CANDIDATES_PER_METHOD = 20  # how many results each method contributes before fusion
TOKEN_RE = re.compile(r"[a-zA-Z]+")

# Common Indian criminal-law shorthand that never appears verbatim in the
# statute text itself (the Acts use formal phrasing instead), so neither
# BM25 nor vector search can bridge the gap on their own. Expanding the
# query with the statute's actual phrasing before search closes that gap.
# Extend this list as you find more real query failures.
LEGAL_ALIASES = {
    "fir": "information cognizable offence police station",
    "chargesheet": "police report completion investigation forwarded magistrate",
    "charge sheet": "police report completion investigation forwarded magistrate",
    "bail": "release on bail bond surety",
    "anticipatory bail": "direction for grant of bail to person apprehending arrest",
    "remand": "custody magistrate detention",
    "cognizance": "magistrate taking cognizance offence",
    "pil": "public interest",
    "plea bargaining": "application for plea bargaining accused",
    "cross examination": "cross-examination witness",
    "hearsay": "statements persons who cannot be called as witnesses",
    "search warrant": "search-warrant issued by court",
    "acquittal": "finding of acquittal accused",
    "cognizable offence": "cognizable case police station",
}


def expand_query(query: str) -> str:
    """Append statutory phrasing for any known colloquial/legal shorthand
    found in the query, so search isn't limited to exact statute wording."""
    lower_query = query.lower()
    additions = []
    for alias, expansion in LEGAL_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", lower_query):
            additions.append(expansion)
    if additions:
        return query + " " + " ".join(additions)
    return query


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def load_corpus():
    """Load the same rows used to build the Chroma DB, so BM25 stays in sync."""
    ids, docs = [], []
    for fname in CSV_FILES:
        path = Path(fname)
        if not path.exists():
            print(f"WARNING: {fname} not found, skipping.")
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ids.append(f"{row['act']}_{row['section_number']}")
                docs.append(row["text"])
    return ids, docs


def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    """
    ranked_lists: list of lists of doc_ids, each already ordered best-first.
    Returns dict {doc_id: fused_score}.
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(query, collection, bm25, corpus_ids, n_results=5, where=None):
    expanded_query = expand_query(query)

    # --- Vector search ---
    vector_results = collection.query(
        query_texts=[expanded_query], n_results=CANDIDATES_PER_METHOD, where=where
    )
    vector_ranked_ids = vector_results["ids"][0]

    # --- BM25 keyword search ---
    tokenized_query = tokenize(expanded_query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(
        zip(corpus_ids, bm25_scores), key=lambda x: x[1], reverse=True
    )
    bm25_ranked_ids = [doc_id for doc_id, score in bm25_ranked if score > 0][:CANDIDATES_PER_METHOD]

    # Apply the same metadata filter to BM25 results if one was given, so a
    # BM25 hit outside the filtered act doesn't leak into fused results.
    if where and "act" in where:
        allowed_prefix = where["act"] + "_"
        bm25_ranked_ids = [i for i in bm25_ranked_ids if i.startswith(allowed_prefix)]

    # --- Fuse ---
    fused_scores = reciprocal_rank_fusion([vector_ranked_ids, bm25_ranked_ids])
    fused_sorted = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    top_ids = [doc_id for doc_id, _ in fused_sorted[:n_results]]
    if not top_ids:
        return []

    fetched = collection.get(ids=top_ids)
    id_to_record = {
        doc_id: (doc, meta)
        for doc_id, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
    }

    results = []
    for doc_id, score in fused_sorted[:n_results]:
        doc, meta = id_to_record[doc_id]
        results.append({
            "id": doc_id,
            "score": score,
            "in_vector_top": doc_id in vector_ranked_ids,
            "in_bm25_top": doc_id in bm25_ranked_ids,
            "metadata": meta,
            "snippet": doc[:120].replace("\n", " "),
        })
    return results


def main():
    corpus_ids, corpus_docs = load_corpus()
    print(f"Loaded {len(corpus_ids)} sections for BM25 index.")
    tokenized_corpus = [tokenize(doc) for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
    print(f"Chroma collection has {collection.count()} chunks.\n")

    queries = [
        "punishment for murder",
        "what counts as theft",
        "rules for admitting electronic evidence",
        "procedure for filing an FIR",
    ]

    for q in queries:
        print(f"=== Query: {q!r} ===")
        results = hybrid_search(q, collection, bm25, corpus_ids, n_results=3)
        for r in results:
            m = r["metadata"]
            tags = []
            if r["in_vector_top"]:
                tags.append("vector")
            if r["in_bm25_top"]:
                tags.append("bm25")
            print(f"  [{m['act']} §{m['section_number']}] (rrf={r['score']:.4f}, hit_by={'+'.join(tags)}) {r['snippet']}...")
        print()

    print("=== Filtered query: 'confession' within BSA only ===")
    results = hybrid_search("confession to police", collection, bm25, corpus_ids, n_results=3, where={"act": "BSA"})
    for r in results:
        m = r["metadata"]
        print(f"  [{m['act']} §{m['section_number']}] {r['snippet']}...")


if __name__ == "__main__":
    main()