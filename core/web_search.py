"""
core/web_search.py

Tavily search + Python-side compression/deduplication utilities.
Identical logic in agent.py and research-pipeline.py. Several of
these (normalize_whitespace, infer_domain, compress_tavily_result,
deduplicate_results) are also identical to app.py's copies and are
imported by ui/app.py directly; tavily_search/search_tavily_parallel
differ slightly in app.py (it has its own retry/timeout behavior)
so app.py keeps its own version of those two.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from core.config import MAX_RESULT_CONTENT_CHARS, TAVILY_RESULTS_PER_QUERY, tavily_client


def normalize_whitespace(text: str) -> str:
    """
    Formatting-only transformation.

    NO semantic filtering.
    """
    return " ".join(
        (text or "").replace("\x00", " ").split()
    ).strip()



def infer_domain(url: str) -> str:
    match = re.match(
        r"https?://([^/]+)",
        url or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).lower()



def compress_tavily_result(
    result: Dict[str, Any],
    query: str,
    candidate_id: int,
):
    """
    LOSSY FORMATTING ONLY.

    This function DOES NOT determine relevance.

    It only:

    - normalizes whitespace
    - truncates long content
    - stores metadata
    """

    title = normalize_whitespace(result.get("title", ""))
    url = (result.get("url", "") or "").strip()
    content = result.get("content") or result.get("snippet") or ""

    content = normalize_whitespace(content)
    content = content[:MAX_RESULT_CONTENT_CHARS]

    return {
        "candidate_id": candidate_id,
        "search_query": query,
        "title": title,
        "url": url,
        "domain": infer_domain(url),
        "tavily_score": result.get("score"),
        "content": content,
    }



def tavily_search(query: str, candidate_id_start: int):
    """
    Execute one Tavily search.
    """

    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=TAVILY_RESULTS_PER_QUERY,
            include_raw_content=False,
            include_images=False,
        )

        raw_results = response.get("results", [])
        compressed = []

        for offset, result in enumerate(raw_results):
            candidate_id = candidate_id_start + offset
            compressed.append(
                compress_tavily_result(
                    result=result,
                    query=query,
                    candidate_id=candidate_id,
                )
            )

        return compressed

    except Exception as exc:
        print()
        print("[Tavily ERROR]")
        print(f"Query: {query}")
        print(f"Error: {exc}")
        return []



def search_tavily_parallel(queries: List[str]):
    """
    Execute the five Tavily searches concurrently.

    Five queries still mean five API calls.
    """

    all_results = []

    if not queries:
        return all_results

    with ThreadPoolExecutor(
        max_workers=min(5, len(queries))
    ) as executor:
        futures = {}

        for index, query in enumerate(queries):
            candidate_id_start = (
                index * TAVILY_RESULTS_PER_QUERY + 1
            )
            future = executor.submit(
                tavily_search,
                query,
                candidate_id_start,
            )
            futures[future] = query

        for future in as_completed(futures):
            query = futures[future]
            print()
            print("[Tavily] Completed:")
            print(f"  {query}")
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as exc:
                print(f"  Failed: {exc}")

    # Sort by candidate ID so output remains deterministic.
    all_results.sort(key=lambda x: x["candidate_id"])
    return all_results



def deduplicate_results(results: List[Dict[str, Any]]):
    """
    Deduplicate candidates by URL.

    First occurrence wins.

    NOTE: this only catches exact/near-identical URLs. It does
    NOT catch two different domains making the same point (e.g.
    two separate legal-blog explainers of the same section).
    That kind of CONTENT redundancy is handled later by the 20B
    ranker's `redundant_with` field plus Python's selection
    logic in select_final_candidates().
    """

    seen_urls = set()
    unique = []

    for result in results:
        url = result.get("url", "").strip()
        if not url:
            continue

        normalized_url = url.rstrip("/").lower()
        if normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)
        unique.append(result)

    return unique

