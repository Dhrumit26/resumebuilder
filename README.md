# Tech Resume Builder

Tailors your real experience to a job posting. Every bullet traces back to a fact
you wrote down; the tool chooses which of them to show, in what order, and how to
phrase them for the posting.

---

## The fact-grounded pipeline (current — `/api/v2/build`)

**Read this first if you are changing anything.**

### The one rule

`data/facts.yaml` is the only source of truth for what may appear on the resume —
**except** when a role sets `fabricated: true`.

A writer agent may **select**, **order**, and **rephrase** facts. It may never
invent a system, a tool, or a number. That is enforced in code, not by asking the
model nicely: [`src/verify.py`](src/verify.py) rejects any bullet containing a
number or technology its own fact does not license, and a bullet that cannot pass
falls back to the plain fact text.

**`flexible: true`** (Clerxi) — aggressive *truthful* reframing of confirmed facts
toward the JD's vocabulary. Still fact-grounded.

**`fabricated: true`** (Clerxi, when you want invent mode) — ignore the fact bank
for that role's bullets and invent a coherent mid-level story natively in the JD's
engineering domain (v1-style). Intuit and projects stay fact-grounded. Craft
checks still apply; fact-license checks do not. Set `fabricated: false` to return
to select/order/rephrase-only for Clerxi.

### How tailoring happens

The LaTeX templates in `latex/` own the layout **and the bullet count** (Clerxi 5,
Intuit 5, each project 2, 4 skills lines). Nothing the model returns can change
that — [`src/skeleton.py`](src/skeleton.py) parses the templates and substitutes
only bullet *text*. So a posting changes the resume in three ways:

1. **Selection** — which facts fill the fixed slots
2. **Order** — the fact that best matches the posting goes first
3. **Framing** — the same work described in the posting's vocabulary

### Adding facts — the single highest-value thing you can do

Each role currently has exactly as many facts as it has bullet slots, so every
fact is always used and only order and framing change. **More real facts = more
genuine tailoring.** If you did frontend work at Clerxi, add it, and a frontend
posting will pull real frontend bullets out of your current role instead of
backend ones reworded.

```yaml
- id: short-slug
  core: >-
    What you actually did, one or two lines.
  tools: [React, TypeScript]     # only what you really used for THIS work
  themes: [frontend-ui, react]   # vocabulary: THEMES in src/facts.py
  metrics: ["35%", "2 hours to 10 minutes"]   # a number not listed here is rejected
  angles:                        # optional: honest reframings for other domains
    - Front-end feature work measured by user outcome
```

Small things count: a UI you touched, a script, a dashboard, a migration, a bug
class you fixed, an on-call save. Facts with no metric are fine — that bullet
simply gets written without a number.

### Scoring

The score is **measured, not judged**: keyword coverage (45), metric density (20),
domain match (20), skills coverage (15). Same resume in, same number out. It says
how well the resume fits the posting — not your odds of getting hired. Missing
keywords are reported as genuine gaps rather than papered over.

### Layout

| File | Role |
|------|------|
| `data/facts.yaml` | the fact bank — what may be said |
| `src/facts.py` | loads and validates it; `THEMES` vocabulary |
| `src/skeleton.py` | parses `latex/` templates, fills bullet slots, escapes LaTeX |
| `src/matching.py` | JD → themes → fact selection and ordering (no LLM) |
| `src/verify.py` | groundedness, craft, and coverage checks (no LLM) |
| `src/pipeline_v2.py` | orchestration |
| `prompts/v2/` | the two writer prompts |
| `tests/facts_pipeline.py` | offline suite, no API key needed |

```bash
python3 tests/facts_pipeline.py     # v2
python3 tests/edge_cases.py         # v1
```

---

## v1 pipeline (legacy — `/api/build`, `/api/build/stream`)

Still present and still passing its tests. It asked the model to invent a
plausible current role and then policed the invention with symptom-specific
gates. Kept for reference; the sections below describe it.

## What Makes This Gold Standard

| Typical resume tools | This builder |
|---------------------|--------------|
| Generic bullet rewriting | **Google XYZ formula** (metric + tech + outcome) |
| One-pass generation | **10-pass pipeline** with review at every stage |
| Basic keyword matching | **Deep JD analysis** (role, seniority, must-haves, ATS keywords) |
| Generic scoring | **Calibrated vs typical applicants** (top 5% target: 80+) |
| Any industry | **Tech-only** — optimized for SWE, backend, frontend, full-stack, DevOps |

## Pipeline (10 AI Passes)

```
Job Description
      │
      ▼
① JD Analyzer — extract role, seniority, must-have skills, ATS keywords
      │
      ▼
② Generate Summary — 2-line role-aligned hook with metric
③ Generate Experience — XYZ bullets, 70%+ with tech stack
④ Bullet Reviewer — kill weak bullets, enforce quality gate
⑤ Competitive Edge — beat typical "responsible for..." applicants
⑥ Generate Projects — production-grade framing, max 2
⑦ Bullet Reviewer — projects pass
⑧ Generate Skills — mirror exact JD terminology
      │
      ▼
⑨ Final Polish — one-page discipline, keyword coverage check
      │
      ▼
⑩ ATS Scorer + ATS Reviewer + Hiring Manager Review
      │
      ▼
LaTeX + Scores + Competitive Advantages
```

## Project Structure

```
resumebuilder/
├── latex/
│   ├── full/           # Complete resume template
│   │   ├── empty.tex   # Skeleton with all sections
│   │   └── original.tex
│   ├── summary/        # Summary section only
│   ├── experience/     # Work experience section only
│   ├── projects/       # Projects section only
│   └── skills/         # Skills section only
│       ├── empty.tex   # Format/structure template
│       └── original.tex # YOUR actual content (edit this!)
├── prompts/
│   ├── tech_standards.txt      # Gold standard rules (XYZ, metrics, tech)
│   ├── jd_analyzer.txt         # Deep JD intelligence extraction
│   ├── summary_generate.txt
│   ├── job_generate.txt
│   ├── project_generate.txt
│   ├── skills_generate.txt
│   ├── bullets_reviewer.txt    # Ruthless quality gate
│   ├── competitive_edge.txt    # Beat typical applicants
│   ├── final_polish.txt        # Last pass before submission
│   ├── ats_scorer.txt
│   ├── ats_reviewer.txt
│   └── human_reviewer.txt      # FAANG hiring manager lens
├── src/
│   ├── main.py           # FastAPI app
│   ├── resume_builder.py # Orchestration pipeline
│   ├── llm.py            # OpenAI / Grok client
│   └── config.py
├── static/               # Web UI
├── requirements.txt
├── Dockerfile
└── run.py
```

## Setup

### 1. Add Your Resume Content

Replace the placeholder content in these files with your real resume:

- `latex/summary/original.tex`
- `latex/experience/original.tex`
- `latex/projects/original.tex`
- `latex/skills/original.tex`
- `latex/full/original.tex` (optional reference copy)

Also update the header (name, email, links) in `latex/full/empty.tex`.

### 2. Configure API Key

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Use OpenAI — gpt-4o recommended for gold-standard quality
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o

# OR use Grok (xAI)
LLM_PROVIDER=grok
GROK_API_KEY=xai-your-key-here
GROK_MODEL=grok-2-1212
```

### 3. Run Locally

```bash
pip install -r requirements.txt
python run.py
```

Open http://localhost:8000

## Deploy Free

### Render (recommended)

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect your repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables: `LLM_PROVIDER`, `OPENAI_API_KEY` (or `GROK_API_KEY`)

### Docker

```bash
docker build -t resumebuilder .
docker run -p 8000:8000 --env-file .env resumebuilder
```

## API

```bash
curl -X POST http://localhost:8000/api/build \
  -H "Content-Type: application/json" \
  -d '{"job_description": "We are looking for a Senior Software Engineer with Python, React, and AWS experience..."}'
```

Response:

```json
{
  "latex": "\\documentclass...",
  "sections": { "summary": "...", "experience": "...", "projects": "...", "skills": "..." },
  "scores": {
    "ats_scorer": { "overall_score": 82, "breakdown": {...}, "verdict": "good_match" },
    "ats_reviewer": { "ats_score": 78, "matched_keywords": [...], "missing_keywords": [...] },
    "human_reviewer": { "human_score": 75, "interview_recommendation": "yes" }
  }
}
```

## Cost Notes

Each generation makes ~**10 LLM calls** (JD analysis + 4 generate + 2 bullet review + competitive edge + final polish + 3 scoring). With `gpt-4o`, roughly **$0.15–0.30 per resume**. Use `gpt-4o-mini` to cut cost (~$0.03) but quality drops.

## Score Calibration

| Score | Meaning |
|-------|---------|
| 80+ | Top 5% — strong interview candidate |
| 65-79 | Good match — competitive |
| 50-64 | Average — typical applicant level |
| <50 | Weak — needs more relevant experience or better framing |
