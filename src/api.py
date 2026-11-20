"""Phase 5: REST API for the RAG pipeline.

Run:  uvicorn src.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.generator import MIN_RELEVANCE, answer_question
from src.retriever import load_bm25

log = logging.getLogger("uvicorn.error")


# ---------- Request / response models (the API's contract) ----------

class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["How much paternity leave do I get in Ireland?"])


class Source(BaseModel):
    n: int                      # the [n] used in the answer's citations
    title: str | None
    heading: str | None
    source: str | None          # file path in the handbook
    score: float                # reranker relevance, 0–1
    text: str


class Confidence(BaseModel):
    label: str                  # High / Medium / Low / None
    score: float | None
    reasons: list[str]


class CitationCheck(BaseModel):
    supported: int
    total: int


class AskResponse(BaseModel):
    question: str
    answer: str
    gated: bool                 # True = blocked by the "I don't know" gate (no LLM call)
    confidence: Confidence
    citations: CitationCheck | None   # None when gated or the answer has no citations
    sources: list[Source]
    latency_s: float
    input_tokens: int
    output_tokens: int
    truncated: bool
    timings: dict[str, float]   # seconds per stage: search / answer / check
    gate_threshold: float


# ---------- App ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-up: build the BM25 index at startup so the first request isn't slow."""
    load_bm25("header")
    log.info("BM25 index loaded, API ready")
    yield


app = FastAPI(
    title="GitLab HR Handbook RAG API",
    description="Hybrid search + reranking + Claude answers with verified citations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # Plain `def` (not async): the pipeline makes blocking API calls,
    # so FastAPI runs it in a worker thread and the server stays responsive.
    start = time.perf_counter()
    try:
        result = answer_question(req.question)
    except Exception:
        log.exception("Pipeline failed for question: %s", req.question)
        raise HTTPException(status_code=502, detail="The AI service failed. Please try again.")

    check = result.get("check") or {}
    citations = (CitationCheck(supported=check["supported"], total=check["total"])
                 if check and not check.get("skipped") else None)
    conf = result["confidence"]

    return AskResponse(
        question=req.question,
        answer=result["answer"],
        gated=result["gated"],
        confidence=Confidence(label=conf["label"], score=conf.get("score"),
                              reasons=conf.get("reasons", [])),
        citations=citations,
        sources=[
            Source(n=i, title=c.get("title"), heading=c.get("heading"),
                   source=c.get("source"), score=round(c["score"], 3), text=c.get("text", ""))
            for i, c in enumerate(result["chunks"], 1)
        ],
        latency_s=round(time.perf_counter() - start, 2),
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        truncated=result["truncated"],
        timings=result.get("timings", {}),
        gate_threshold=MIN_RELEVANCE,
    )