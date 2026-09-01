# Verdikt 

## 1. What it is

Verdikt is an AI research agent for Indian criminal law. Given a case — either
structured offences/allegations or a raw, messy free-form description — it produces a
structured legal research memo: the applicable statutory provisions, relevant judicial
precedent pulled from the live web, and a final reasoned legal analysis, with every
claim traceable back to real statute text or a real search result.

It is built around three ideas:

1. **Retrieval should be hybrid.** Statutory language and everyday legal language
   don't always match. A client says "FIR"; the statute says "information in
   cognizable cases." Keyword search alone misses semantic matches; vector search
   alone misses exact statutory phrasing. Verdikt runs both and fuses the results.
2. **Precedent must be found live, not recalled.** An LLM's training data is a stale
   snapshot. Judicial research is done through real-time web search, every run.
3. **Authority is not the LLM's call.** A model can *assess* how relevant or credible
   a source looks, but whether that source is allowed to carry weight — Supreme Court
   judgment vs. legal blog vs. news report — is enforced by deterministic rules, not
   left to the model's judgment.

## surfaces

| Surface | Entry point | Use case |
|---|---|---|
| CLI — fixed case | `python -m cli.pipeline` | Run the pipeline against a hardcoded sample case (development, testing, demos) |
| CLI — free-form agent | `python -m cli.agent` | Paste in a raw client narrative; the agent extracts the legal facts itself first |
| Web UI | `streamlit run ui/app.py` | Interactive research tool — enter a case, watch each stage run, browse ranked precedent and the final memo in tabs |

## Pipeline stages

```
User input (structured or raw free-form query)
        │
        ▼
┌───────────────────────┐
│ Stage 0 — Case Fact    │  (free-form queries only)
│ Extraction             │  raw text → structured offences + allegations
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 1 — Statute      │  BM25 (keyword) + Chroma (vector) search
│ Retrieval              │  over BNS / BNSS / BSA, fused via
│                        │  Reciprocal Rank Fusion (RRF)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 2 — Research     │  LLM reads statutes + case facts,
│ Planning               │  produces a structured research plan
│                        │  + search queries
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 3 — Judicial     │  Live web search (parallel) against the
│ Research               │  planned queries; results deduplicated
│                        │  and compressed
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 4 — Precedent     │  LLM classifies each result by source type
│ Ranking                │  and scores relevance; Python enforces a
│                        │  hard authority CAP per source type
│                        │  (judgment > tribunal > commentary > blog)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 5 — Final Legal   │  LLM synthesizes statutes + plan + ranked
│ Reasoning              │  precedent into the final structured
│                        │  research output
└───────────┬───────────┘
            ▼
   Structured legal research memo (JSON)
   applicable_law · precedent_analysis ·
   sections_applied · final reasoning
```

### Why Stage 4 is the core design decision

Every other stage hands off judgment to an LLM and moves on. Stage 4 doesn't. The
ranker LLM assigns a `judicial_authority_score` and a `source_type` to each candidate —
but that score is then passed through a fixed ceiling table before it's allowed to
influence the final selection:

```python
SOURCE_TYPE_AUTHORITY_CAPS = {
    "supreme_court_judgment": 100,
    "high_court_judgment":     90,
    "tribunal_decision":       75,
    "judgment_repository":     65,
    "legal_commentary":        35,
    "legal_blog":              20,
    "news_report":             15,
    "generic_information":     10,
    "video":                    0,
    "social_media":             0,
    "unknown":                 25,
}
```

No matter how highly the model scores a blog post, it cannot cross the ceiling
assigned to its source type. This is what keeps the final memo's precedent list
actually authoritative rather than just "whatever ranked highest by vibes."

## System architecture

```
                         ┌─────────────────────────────┐
                         │        User-facing layer      │
                         │                              │
                         │  cli/pipeline.py              │
                         │  cli/agent.py                 │
                         │  ui/app.py  (Streamlit)        │
                         └───────────────┬──────────────┘
                                         │  orchestrates stages,
                                         │  handles I/O
                                         ▼
                         ┌─────────────────────────────┐
                         │            core/               │
                         │                              │
                         │  config.py    — env, models,   │
                         │                 tunables, caps  │
                         │  models.py    — Pydantic schemas│
                         │  llm.py       — structured LLM  │
                         │                 output helpers   │
                         │  extraction.py — Stage 0        │
                         │  retrieval.py  — Stage 1        │
                         │  planning.py   — Stage 2        │
                         │  web_search.py — Stage 3        │
                         │  ranking.py    — Stage 4        │
                         │  reasoning.py  — Stage 5        │
                         │  output.py     — printing/saving│
                         └───┬────────────────┬──────────┘
                             │                │
                 ┌───────────▼──────┐   ┌─────▼─────────┐
                 │   vectorstore/     │   │  External APIs  │
                 │                    │   │                │
                 │  build_vector_db.py│   │  Groq — LLM     │
                 │  query_vector_db.py│   │  inference      │
                 │  chroma_db/  (DB)   │   │                │
                 └───────────┬──────┘   │  Tavily — live  │
                             │           │  web search     │
                       ┌─────▼─────┐    └────────────────┘
                       │   data/     │
                       │  BNS/BNSS/  │
                       │  BSA CSVs   │
                       └────────────┘
```

**Data flow at retrieval time:** `data/*.csv` is ingested once by
`vectorstore/build_vector_db.py` into a persistent Chroma collection. At query time,
`core/retrieval.py` calls `vectorstore/query_vector_db.py`'s `hybrid_search()`, which
runs BM25 over the raw corpus and a Chroma vector query in parallel, then fuses the two
ranked lists with Reciprocal Rank Fusion before returning the top statute sections.

## Repository structure

```
legal_research_agent/
├── core/                    # shared pipeline logic — one implementation per stage
│   ├── config.py               # env keys, model names, tunables, authority caps, API clients
│   ├── models.py                # Pydantic schemas for every stage's structured output
│   ├── llm.py                    # strict_schema() + groq_structured()
│   ├── extraction.py             # Stage 0
│   ├── retrieval.py              # Stage 1
│   ├── planning.py               # Stage 2
│   ├── web_search.py             # Stage 3
│   ├── ranking.py                # Stage 4
│   ├── reasoning.py              # Stage 5
│   └── output.py                  # console printers + save_final_output()
│
├── cli/
│   ├── pipeline.py               # entry point: fixed sample case
│   └── agent.py                  # entry point: free-form query
│
├── ui/
│   └── app.py                     # Streamlit web app
│
├── vectorstore/
│   ├── build_vector_db.py        # CSV → Chroma ingestion
│   ├── query_vector_db.py        # hybrid_search() (BM25 + Chroma + RRF)
│   └── chroma_db/                  # persisted vector database
│
├── data/
│   ├── BNS_sections.csv
│   ├── BNSS_sections.csv
│   └── BSA_sections.csv
│
├── requirements.txt
├── .env
└── legal_research_output.json    # example output
```

## Tech stack

| Layer | Tool | Role |
|---|---|---|
| LLM inference | **Groq** | Case extraction, planning, ranking, final reasoning |
| Web search | **Tavily** | Live judicial precedent lookup |
| Vector store | **ChromaDB** | Semantic statute retrieval |
| Keyword search | **rank_bm25** | Lexical statute retrieval, fused with vector search |
| Schema validation | **Pydantic** | Structured, validated output at every stage |
| UI | **Streamlit** | Interactive web app |