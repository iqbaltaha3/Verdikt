# Legal Research Agent — modular structure

## Layout

```
legal_research_agent/
├── core/            # shared pipeline logic (single source of truth)
│   ├── config.py       # env keys, model names, tunables, SOURCE_TYPE_AUTHORITY_CAPS, groq/tavily clients
│   ├── models.py       # all Pydantic schemas (ResearchPlan, CandidateAssessment, LegalResearchOutput, ...)
│   ├── llm.py           # strict_schema() + groq_structured() (CLI version)
│   ├── retrieval.py     # STEP 1 — real hybrid BM25 + Chroma statute retrieval
│   ├── planning.py      # STEP 2 — Groq 120B research planner
│   ├── web_search.py    # STEP 3 — Tavily search, compression, dedup (CLI version)
│   ├── ranking.py       # STEP 4 — Groq 20B ranker + source classifier + Python selection (CLI version)
│   ├── reasoning.py     # STEP 5 — final Groq 120B legal reasoning (CLI version, no raw-query param)
│   ├── extraction.py    # STEP 0 — case-fact extraction (used only by cli/agent.py)
│   └── output.py        # console pretty-printers + save_final_output()
│
├── cli/             # thin CLI entry points, no business logic of their own
│   ├── pipeline.py     # `python -m cli.pipeline`  — fixed OFFENCES/ALLEGATION sample case
│   └── agent.py        # `python -m cli.agent`     — free-form query -> case-fact extraction -> pipeline
│                          (keeps its own final_legal_reasoning_with_query(), the one function that
│                           genuinely differs from core.reasoning.final_legal_reasoning())
│
├── ui/
│   └── app.py           # Streamlit app (`streamlit run ui/app.py`)
│                          Imports the pieces from core/ that were byte-identical to the CLI's
│                          (models, strict_schema, normalize_whitespace, infer_domain,
│                          compress_tavily_result, deduplicate_results) and now uses the REAL
│                          Chroma+BM25 retrieval from core.retrieval instead of the old
│                          retrieve_statutes_stub() keyword-matching fake.
│                          Its own planner/ranker/reasoner/tavily-search prompts and logic are
│                          genuinely different from the CLI's and were left in place unchanged.
│
├── vectorstore/
│   ├── build_vector_db.py   # one-time ingestion: data/*.csv -> chroma_db/
│   ├── query_vector_db.py   # hybrid_search() (BM25 + Chroma via RRF), used by core.retrieval
│   └── chroma_db/           # the real, persisted Chroma DB (untouched, just relocated)
│
├── data/
│   ├── BNS_sections.csv
│   ├── BNSS_sections.csv
│   └── BSA_sections.csv
│
├── requirements.txt
├── .env
└── legal_research_output.json   # sample saved output
```

## What changed vs. the original flat layout

- **No behavior changes to any function that had real, distinct logic.** Before touching
  anything, every function/class in `agent.py`, `research-pipeline.py`, and `app.py` was
  compared at the AST level (not just text diffed) to confirm which were truly identical
  vs. genuinely different. Only truly-identical code was merged into `core/`.
- `agent.py` and `research-pipeline.py` were ~95% byte-identical already; that shared code
  now lives once in `core/`, and each CLI script is a thin `main()` that calls it.
- `app.py`'s planner/ranker/selection/reasoning/Tavily-search prompts genuinely differ from
  the CLI's (different prompt wording, different weighting) — those were **left untouched**
  in `ui/app.py` rather than force-merged, per your instruction that other functions remain
  as-is.
- `ui/app.py`'s `retrieve_statutes_stub()` — a hardcoded keyword-matcher with a handful of
  fake sections, completely disconnected from the vector DB — was replaced with the real
  hybrid BM25 + Chroma retrieval (`core.retrieval`), against the same `chroma_db/` the CLI
  tools use. This was the one explicit behavior change you asked for.

## Two things worth flagging (found during the refactor, not introduced by it)

1. **`app.py` had live Groq and Tavily API keys hardcoded in plaintext** (and committed to
   `.git`). They've been switched to `os.getenv("GROQ_API_KEY")` / `os.getenv("TAVILY_API_KEY")`,
   matching the `.env`-based approach the rest of the project already uses. **Rotate both
   keys in their provider dashboards** — being in source control (even briefly) means they
   should be treated as compromised.
2. `requirements.txt` was missing `chromadb` and `rank_bm25`, even though `build_vector_db.py`
   / `query_vector_db.py` import them directly. Added both.

## Running it

```bash
pip install -r requirements.txt

# one-time: build the vector DB from data/*.csv (already built and included, but if you
# need to rebuild it after editing the CSVs):
python -m vectorstore.build_vector_db

# CLI, fixed sample case:
python -m cli.pipeline

# CLI, free-form query with case-fact extraction:
python -m cli.agent

# Streamlit UI:
streamlit run ui/app.py
```
