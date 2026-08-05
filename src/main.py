import json
import queue
import threading
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline import build_tailored_resume
from .pipeline_v2 import build_resume_v2, refine_resume_v2

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Resume Builder", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class BuildRequest(BaseModel):
    job_description: str = Field(..., min_length=20, description="Target job description")


class RefineRequest(BaseModel):
    job_description: str = Field(..., min_length=20, description="Target job description")
    suggestion: str = Field(..., min_length=3, description="What to change in the resume")
    sections: dict = Field(..., description="sections object from the last /api/v2/build")
    jd_analysis: Optional[dict] = Field(
        default=None,
        description="Optional JD analysis from the last build (skips re-analyzing)",
    )


class BuildResponse(BaseModel):
    latex: str
    sections: dict
    jd_analysis: dict
    scores: dict
    meta: dict = {}


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Resume Builder API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# Sync endpoint on purpose: FastAPI runs it in a worker thread, so the
# 40s+ pipeline doesn't block the event loop (health checks, static files).
@app.post("/api/build", response_model=BuildResponse)
def build_resume(request: BuildRequest):
    try:
        result = build_tailored_resume(request.job_description)
        return result
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model returned invalid JSON. Try again. ({exc})",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Resume generation failed: {exc}") from exc


# Fact-grounded build. Every bullet traces to data/facts.yaml, the LaTeX
# templates own the layout, and quality is measured in code rather than scored
# by an LLM. See src/pipeline_v2.py.
@app.post("/api/v2/build")
def build_resume_facts(request: BuildRequest):
    try:
        return build_resume_v2(request.job_description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Resume generation failed: {exc}") from exc


@app.post("/api/v2/refine")
def refine_resume_facts(request: RefineRequest):
    try:
        return refine_resume_v2(
            request.job_description,
            request.sections,
            request.suggestion,
            request.jd_analysis,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Resume refine failed: {exc}") from exc


_STREAM_DONE = object()


# Same pipeline, streamed as Server-Sent Events. The fix loop is serial and can't
# be shortened, so this makes the wait usable instead: a complete resume arrives
# at roughly a third of total wall-clock, then better versions replace it.
# Events: jd -> draft -> pass -> (draft -> pass)* -> final | error
@app.post("/api/build/stream")
def build_resume_stream(request: BuildRequest):
    events: queue.Queue = queue.Queue()

    def run():
        try:
            result = build_tailored_resume(
                request.job_description,
                on_progress=lambda event, payload: events.put((event, payload)),
            )
            events.put(("final", result))
        except ValueError as exc:
            events.put(("error", {"detail": str(exc)}))
        except Exception as exc:
            events.put(("error", {"detail": f"Resume generation failed: {exc}"}))
        finally:
            events.put(_STREAM_DONE)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            item = events.get()
            if item is _STREAM_DONE:
                break
            event, payload = item
            yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop proxy buffering from defeating the point
        },
    )
