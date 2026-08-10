"""
backend/main.py
------------------
FastAPI application entry point. Thin HTTP layer only: routes, request/
response wiring via backend/schemas.py, and error translation. All actual
inference logic lives in backend/inference_engine.py -- this file never
touches the model directly.

Model loading happens once, at startup (via `lifespan`), and the resulting
InferenceEngine is held in `app.state` for the life of the process --
requests never reload the model.

Endpoints:
    GET  /health         -- liveness/readiness check
    POST /correct         -- raw sequence submitted as JSON string
    POST /correct/file     -- single-record .fasta/.fa file upload
"""

from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from config import DEFAULT_MODEL_CHECKPOINT_PATH
from backend.schemas import (
    ErrorResponse,
    HealthCheckResponse,
    InferenceRequest,
    InferenceResponse,
)
from backend.inference_engine import InferenceEngine, parse_fasta_upload
from model.sequence_translation_model import SequenceTranslationConfig, SequenceTranslationModel

ALLOWED_UPLOAD_EXTENSIONS = {".fasta", ".fa"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if Path(DEFAULT_MODEL_CHECKPOINT_PATH).exists():
        print(f"Loading model checkpoint from {DEFAULT_MODEL_CHECKPOINT_PATH} ...")
        engine = InferenceEngine.load_from_checkpoint(DEFAULT_MODEL_CHECKPOINT_PATH, device=device)
        app.state.model_loaded_from_checkpoint = True
    else:
        # No checkpoint yet -- training/train.py hasn't been built/run in this
        # project at this point. Serve a freshly-initialized (random-weight)
        # model so the API is fully exercisable end-to-end during development;
        # outputs are meaningless until a real checkpoint exists, which is
        # exactly why this is logged loudly rather than silently proceeding.
        print(
            f"WARNING: no checkpoint found at '{DEFAULT_MODEL_CHECKPOINT_PATH}'. Serving an "
            f"UNTRAINED model (random weights) -- corrections will be meaningless until "
            f"training/train.py produces a real checkpoint at that path."
        )
        model = SequenceTranslationModel(SequenceTranslationConfig())
        engine = InferenceEngine(model, device=device)
        app.state.model_loaded_from_checkpoint = False

    app.state.engine = engine
    app.state.device = str(device)
    print(f"Model ready on device: {device}")

    yield

    app.state.engine = None


app = FastAPI(
    title="Context-Driven Neural Sequence Translation API",
    description="Neural genome error correction for ONT long-read sequencing data.",
    version="0.1.0",
    lifespan=lifespan,
)

# NOTE: permissive CORS is fine for local development against the React
# frontend, but should be narrowed to the actual deployed frontend origin(s)
# before this is exposed anywhere beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_engine() -> InferenceEngine:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try again shortly.")
    return engine


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    engine = getattr(app.state, "engine", None)
    return HealthCheckResponse(
        status="ok" if engine is not None else "loading",
        model_loaded=engine is not None,
        device=getattr(app.state, "device", "unknown"),
    )


@app.post(
    "/correct",
    response_model=InferenceResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def correct_sequence(request: InferenceRequest) -> InferenceResponse:
    """
    Corrects a raw nucleotide sequence submitted as a JSON string.
    request.sequence has already been validated and normalized (alphabet,
    length, FASTA-header stripping) by InferenceRequest -- this handler
    never sees malformed input.

    The actual model call runs in a threadpool (run_in_threadpool) so this
    async handler's event loop is never blocked by the model's synchronous,
    CPU/GPU-bound forward pass -- required so long-read inference doesn't
    stall other concurrent requests (Part 4 spec).
    """
    engine = _get_engine()
    try:
        result = await run_in_threadpool(engine.correct_sequence, request.sequence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    return InferenceResponse(**result)


@app.post(
    "/correct/file",
    response_model=InferenceResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def correct_sequence_file(file: UploadFile = File(...)) -> InferenceResponse:
    """
    Corrects a sequence submitted as a single-record .fasta/.fa file
    upload. Reuses InferenceRequest's validation (alphabet, length) after
    parsing, rather than duplicating those checks here.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{suffix}'. Expected one of {sorted(ALLOWED_UPLOAD_EXTENSIONS)}.",
        )

    raw_bytes = await file.read()
    try:
        raw_sequence = parse_fasta_upload(raw_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        validated = InferenceRequest(sequence=raw_sequence)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    engine = _get_engine()
    try:
        result = await run_in_threadpool(engine.correct_sequence, validated.sequence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e
    return InferenceResponse(**result)


# ---------------------------------------------------------------------------
# Sanity checks (FastAPI TestClient -- exercises the real app, real lifespan,
# real (untrained) model, real routes; no mocking).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from fastapi.testclient import TestClient

    with TestClient(app) as client:  # __enter__/__exit__ trigger the lifespan startup/shutdown
        # -- 1. Health check reports the (untrained, no-checkpoint) model loaded --
        resp = client.get("/health")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["model_loaded"] is True
        assert body["status"] == "ok"
        print(f"[1/6] GET /health passed: {body}")

        # -- 2. POST /correct with a short, valid sequence -----------------------
        short_seq = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        resp = client.post("/correct", json={"sequence": short_seq})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "corrected_sequence" in body
        assert body["metrics"]["input_length"] == len(short_seq)
        assert len(body["attention_chunks"]) == 1
        print(
            f"[2/6] POST /correct passed: corrected_length={body['metrics']['corrected_length']}, "
            f"edit_distance={body['metrics']['edit_distance']} (untrained model, so this only "
            f"proves the endpoint runs end to end, not correction quality)."
        )

        # -- 3. POST /correct with an invalid sequence -> 422 --------------------
        resp = client.post("/correct", json={"sequence": "ACGTXYZ"})
        assert resp.status_code == 422, resp.text
        print("[3/6] POST /correct with invalid alphabet correctly returned 422.")

        # -- 4. POST /correct/file with a valid single-record FASTA upload -------
        #       using a real excerpt of the real E. coli genome ------------------
        with open("data/reference/ecoli_k12_mg1655.fasta") as f:
            lines = f.readlines()
        real_excerpt = "".join(l.strip() for l in lines[1:])[300_000:300_100]  # real 100bp excerpt
        fasta_bytes = (">test_excerpt\n" + real_excerpt + "\n").encode("utf-8")

        resp = client.post(
            "/correct/file",
            files={"file": ("excerpt.fasta", fasta_bytes, "text/plain")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["metrics"]["input_length"] == 100
        print(f"[4/6] POST /correct/file passed on a real 100bp E. coli excerpt: {body['metrics']}")

        # -- 5. POST /correct/file with wrong extension -> 400 --------------------
        resp = client.post(
            "/correct/file",
            files={"file": ("excerpt.txt", fasta_bytes, "text/plain")},
        )
        assert resp.status_code == 400, resp.text
        print("[5/6] POST /correct/file with disallowed extension correctly returned 400.")

        # -- 6. POST /correct/file with a multi-record FASTA -> 400 ---------------
        multi_record = b">r1\nACGTACGT\n>r2\nGGCCGGCC\n"
        resp = client.post(
            "/correct/file",
            files={"file": ("multi.fasta", multi_record, "text/plain")},
        )
        assert resp.status_code == 400, resp.text
        assert "more than one record" in resp.json()["detail"]
        print("[6/6] POST /correct/file with a multi-record FASTA correctly returned 400.")

    print("\nAll backend/main.py sanity checks passed.")
