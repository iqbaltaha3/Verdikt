"""
core/config.py

Central configuration shared by the CLI pipeline, the CLI agent,
and (partially) the Streamlit UI: environment/API keys, model
names, retrieval/ranking tunables, and the Groq/Tavily clients.

Moved here verbatim from research-pipeline.py / agent.py, which
defined identical copies of all of this at module level.
"""

import os

from groq import Groq
from tavily import TavilyClient

# ============================================================
# ENVIRONMENT
# ============================================================

# Prevent HuggingFace tokenizer fork warnings/deadlocks.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# CONFIGURATION
# ============================================================
#
# API keys are NEVER hardcoded. Set them in your shell / .env:
#
#   export GROQ_API_KEY='your_key'
#   export TAVILY_API_KEY='your_key'
#
# If a key has ever been committed to source control or pasted
# into a chat/log, rotate it immediately in the provider
# dashboard -- treat it as compromised even after removal.

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ============================================================
# MODELS
# ============================================================

EXTRACTOR_MODEL = "openai/gpt-oss-120b"
PLANNER_MODEL = "openai/gpt-oss-120b"
RANKER_MODEL = "openai/gpt-oss-20b"
REASONING_MODEL = "openai/gpt-oss-120b"

# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

N_STATUTE_RESULTS = 8
N_SEARCH_QUERIES = 5
TAVILY_RESULTS_PER_QUERY = 3

MAX_RESULT_CONTENT_CHARS = 700
MAX_CANDIDATES_TO_RANK = 15
MIN_SELECTED_PRECEDENTS = 3
MAX_SELECTED_PRECEDENTS = 5

MAX_STATUTE_TEXT_CHARS = 1200
MAX_REASONER_CANDIDATE_CHARS = 500

# ============================================================
# SOURCE TYPE AUTHORITY CAPS
# ============================================================
#
# These caps are enforced in PYTHON, not left to the LLM's
# numerical judicial_authority_score. A legal_blog or a
# social_media hit should never be able to out-rank an actual
# judgment purely because a model assigned it a high score.
#
# The cap is an upper bound applied to judicial_authority_score
# AFTER the 20B assessment, before Python's weighted selection.

SOURCE_TYPE_AUTHORITY_CAPS = {
    "supreme_court_judgment": 100,
    "high_court_judgment": 90,
    "tribunal_decision": 75,
    "judgment_repository": 65,
    "legal_commentary": 35,
    "legal_blog": 20,
    "news_report": 15,
    "generic_information": 10,
    "video": 0,
    "social_media": 0,
    "unknown": 25,
}

# ============================================================
# SAMPLE CASE (used by cli/pipeline.py's default run)
# ============================================================

OFFENCES = """
Alleged criminal intimidation and misconduct by a public servant
"""

ALLEGATION = """
The complainant alleges that the police officer abused,
intimidated, threatened to detain him, and pushed a document toward him
during a visit to the police post, which the officer denies.
"""

# Raw free-form query used by cli/agent.py's default run (goes
# through the case-fact extractor instead of the OFFENCES/ALLEGATION
# constants above).
RAW_USER_QUERY = """
My client visited the police station to file a complaint.
The SHO abused him, threatened to put him in lock-up if he didn't
withdraw the complaint, and pushed a paper at him. The officer
now denies everything. We want to know what sections apply and
whether we can file a complaint directly before the magistrate.
"""

# ============================================================
# CLIENTS
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
