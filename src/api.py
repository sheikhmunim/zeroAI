"""FastAPI service wrapping the detector.

A minimal FastAPI app has three moving parts:

  * `app = FastAPI()` -- an ASGI application object. It is not a server; it is
    a callable that a server (uvicorn) invokes per request. It also collects
    your route definitions and derives an OpenAPI schema from their type hints,
    which is where the free /docs page comes from.
  * a route decorator (`@app.post("/predict")`) -- registers a function to a
    method + path. FastAPI inspects the signature and turns each parameter into
    a piece of request parsing: `UploadFile` means multipart, a pydantic model
    means a JSON body, a plain scalar means a query parameter.
  * pydantic models -- declare the shape of data. On input they validate and
    coerce, returning a 422 with field-level detail on failure. On output they
    filter and document the response. Neither behaviour is decoration: it is
    the actual parsing and serialisation layer.

Run:
    uvicorn src.api:app --reload --port 8000
    open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.config import ARTIFACT_DIR, PROJECT_ROOT
from src.inference import Detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("detector")

# Uploads are read fully into memory before decoding, so this cap is what stops
# a single request from exhausting the container's RAM. Note this is a
# defence-in-depth measure: a reverse proxy should also cap body size, because
# by the time we check here we have already buffered the bytes.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

HEAD_PATH = Path(os.getenv("HEAD_PATH", ARTIFACT_DIR / "head.pt"))
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class Prediction(BaseModel):
    """Response schema. Also what renders in the /docs example."""

    label: str = Field(description="'real' or 'ai'")
    confidence: float = Field(ge=0, le=1, description="confidence in the label given")
    p_ai: float = Field(ge=0, le=1, description="raw probability the image is AI-generated")
    threshold: float = Field(description="cutoff applied to p_ai")
    inference_ms: float
    filename: str | None = None


class Health(BaseModel):
    status: str
    model_loaded: bool
    model: dict | None = None


# `state` holds the single Detector instance for the process lifetime.
state: dict[str, Detector | None] = {"detector": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, at startup -- never per request.

    This is the production gotcha worth internalising. Constructing a Detector
    means reading ~600 MB of CLIP weights off disk and materialising 151M
    parameters. That takes seconds and hundreds of MB of RAM.

    Doing it inside the request handler would mean:
      * every request pays the full load cost, so a ~40 ms inference becomes a
        multi-second request;
      * N concurrent requests hold N copies of the model in memory, and the
        container is OOM-killed under trivial load;
      * the disk read dominates any optimisation you make to the model itself.

    Loading here instead means the cost is paid once, before the first request,
    and every handler shares one read-only instance. Handlers must therefore
    treat the model as shared state -- fine here because inference under
    `torch.inference_mode()` mutates nothing.

    Code above `yield` runs at startup, code below at shutdown. This replaces
    the older `@app.on_event("startup")`, which is deprecated -- lifespan also
    guarantees teardown runs, which on_event did not reliably do.
    """
    log.info("loading model from %s", HEAD_PATH)
    detector = Detector(HEAD_PATH)
    warmup_ms = detector.warmup()
    state["detector"] = detector
    log.info("model ready (warmup %.0f ms) %s", warmup_ms, detector.info())

    yield

    state["detector"] = None
    log.info("shutdown")


app = FastAPI(
    title="AI Image Detector",
    description="Frozen CLIP backbone + trained MLP head. Returns real vs AI-generated.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS. Explained in docs/03-frontend.md -- in short, a browser refuses to let
# JS read a response from a different origin unless the server opts in, and
# "different origin" includes a different port on localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_detector() -> Detector:
    detector = state["detector"]
    if detector is None:
        # 503, not 500: the service is temporarily unable to serve, which tells
        # a load balancer to retry elsewhere rather than treating it as a bug.
        raise HTTPException(status_code=503, detail="model not loaded")
    return detector


@app.get("/health", response_model=Health)
def health() -> Health:
    """Liveness/readiness probe.

    Deployment platforms poll this to decide whether the container should
    receive traffic, and CI uses it as the gate before running a smoke test.

    The important design point: it reports whether the *model* is loaded, not
    merely whether the web server answered. A health check that returns 200 as
    soon as the HTTP layer is up will mark the container ready during the
    several seconds CLIP is still loading -- traffic arrives, every request
    500s, and the deploy looks successful. Health must mean "can actually serve
    a request", which is why it is wired to the same state the handlers use.

    It deliberately does NOT run an inference. A probe hit every few seconds
    that burns 40 ms of CPU competes with real traffic, and would make the
    health check itself a source of load.
    """
    detector = state["detector"]
    if detector is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return Health(status="ok", model_loaded=True, model=detector.info())


@app.post("/predict", response_model=Prediction)
def predict(
    file: UploadFile = File(..., description="image file (jpeg/png/webp)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
) -> Prediction:
    """Classify one uploaded image.

    Note this is `def`, not `async def`, and that is deliberate. FastAPI runs a
    plain `def` handler in a worker thread from its threadpool; an `async def`
    handler runs directly on the single event loop. Model inference is blocking
    CPU work with no await points, so putting it in an `async def` would stall
    the event loop for its whole duration -- every other in-flight request,
    including /health, would freeze behind it. Writing `def` lets the event
    loop keep serving while the blocking work happens off to the side.

    The counterintuitive rule: for blocking work, the *non*-async handler is
    the concurrent one.
    """
    detector = get_detector()

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"expected an image, got {file.content_type}")

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file is {len(raw) / 1e6:.1f} MB, limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB",
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()  # force decode now so a corrupt file fails here, not later
    except (UnidentifiedImageError, OSError) as exc:
        # A malformed upload is the client's error, so 400. Letting PIL's
        # exception escape would surface as a 500 and page whoever is on call
        # for what is really just a bad file.
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc

    result = detector.predict([image], threshold=threshold)[0]
    return Prediction(**result, filename=file.filename)


# Mounted last so the API routes above take precedence. html=True serves
# index.html at "/". Serving the frontend from the same origin as the API is
# what makes CORS unnecessary in production -- it only matters in local dev.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
