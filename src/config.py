import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
LATEX_DIR = BASE_DIR / "latex"
PROMPTS_DIR = BASE_DIR / "prompts"

load_dotenv(BASE_DIR / ".env")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "grok"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.5")
# Writer: strongest model for section writing. Judge: fast model for JSON
# analysis/scoring where the strict rubric does the heavy lifting.
GROK_WRITER_MODEL = os.getenv("GROK_WRITER_MODEL", GROK_MODEL)
GROK_JUDGE_MODEL = os.getenv("GROK_JUDGE_MODEL", "grok-4.20-0309-non-reasoning")
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")

# Fix loop: keep applying reviewer suggestions until ATS score hits threshold
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "90"))
MAX_REMAKE_ATTEMPTS = int(os.getenv("MAX_REMAKE_ATTEMPTS", "4"))  # targeted fix rounds
# 92, not 95: the reviewer rubric caps differentiation at 2/5 and
# keyword_optimization at 5/10 whenever its anti-AI penalty fires, so the
# reachable ceiling is 35+20+15+10+5+2+5 = 92. Targeting 95 made the exit
# condition unsatisfiable, so every build ground through all MAX_REMAKE_ATTEMPTS
# rounds for nothing (the best-version-kept guarantee means extra rounds can
# never improve the result — only cost time).
TARGET_SCORE = int(os.getenv("TARGET_SCORE", "92"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))  # reasoning models are slow

# Quality-over-cost knobs (user opted into higher API spend)
BEST_OF_N = int(os.getenv("BEST_OF_N", "2"))  # parallel candidate resumes per build
SCORE_SAMPLES = int(os.getenv("SCORE_SAMPLES", "2"))  # reviewer samples averaged per version
