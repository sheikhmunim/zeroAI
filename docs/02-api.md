# Stage 2 — FastAPI Backend

Files: `src/inference.py`, `src/api.py`, `tests/test_api.py`,
`requirements-serve.txt`

---

## 1. The shape of a minimal FastAPI app

Three moving parts:

**`app = FastAPI()`** — an ASGI application object. It is *not* a server; it is
a callable that a server (uvicorn) invokes once per request. It also collects
your route definitions and derives an OpenAPI schema from their type hints,
which is where the free `/docs` page comes from.

**A route decorator** (`@app.post("/predict")`) — registers a function against a
method + path. FastAPI inspects the function signature and turns each parameter
into a piece of request parsing:

| parameter type | parsed from |
|---|---|
| `UploadFile` | multipart form body |
| a pydantic model | JSON body |
| a plain scalar | query string |
| `Path`-annotated | URL path segment |

**Pydantic models** — declare the shape of data. On input they validate and
coerce, returning `422` with field-level detail on failure. On output they
filter and document the response. This is not decoration: it is the actual
parsing and serialisation layer.

---

## 2. Request/response schema — why multipart file upload

Three options for getting an image to the server:

| approach | for | against |
|---|---|---|
| **multipart file upload** | native to browsers (`FormData`), no size inflation, streams, `curl -F` just works | slightly more ceremony than JSON |
| base64 in JSON | trivially loggable, one content type | **+33% payload size**, whole image buffered as a string, awkward from a browser |
| image URL | tiny request, no upload | server-side fetch is an **SSRF hole** — a caller can make your server request `http://169.254.169.254/` and read cloud credentials. Also adds latency and a failure mode you don't control |

**Chosen: multipart.** The URL option is not merely worse, it is a security
liability that would need an allowlist and egress controls to be safe.

Response is a pydantic `Prediction`:

```json
{"label": "ai", "confidence": 0.9996, "p_ai": 0.9996,
 "threshold": 0.5, "inference_ms": 90.6, "filename": "x.png"}
```

`confidence` is deliberately the confidence *in the answer given*, not `p_ai`. A
confident "real" should read 0.98, not 0.02. Both are returned so the caller can
have the raw number when it wants it.

---

## 3. Why the model loads once at startup

This is the production gotcha. Constructing a `Detector` reads ~350 MB of CLIP
weights off disk and materialises 88M parameters. It takes seconds.

Doing it inside the request handler would mean:

- every request pays the full load cost — a 110 ms inference becomes a
  multi-second request;
- **N concurrent requests hold N copies of the model**, and the container is
  OOM-killed under trivial load;
- the disk read dominates every optimisation you could make to the model.

Loading in the lifespan handler means the cost is paid once, before the first
request, and every handler shares one read-only instance. Handlers must
therefore treat the model as shared state — which is fine here because
inference under `torch.inference_mode()` mutates nothing.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    state["detector"] = Detector(HEAD_PATH)   # startup
    yield
    state["detector"] = None                  # shutdown
```

Code above `yield` runs at startup, below at shutdown. This replaces the older
`@app.on_event("startup")`, which is deprecated — lifespan also guarantees
teardown actually runs, which `on_event` did not reliably do.

### Warmup

`Detector.warmup()` runs one throwaway prediction during startup. The first
forward pass in a fresh torch process pays for lazy kernel selection,
thread-pool spin-up and allocator warm-up — often several times steady-state
latency. Paying it at startup is free and removes an ugly outlier from your p99.

---

## 4. `GET /health` — what it is for

Deployment platforms poll it to decide whether the container should receive
traffic. CI uses it as the gate before running a smoke test.

> **It reports whether the *model* is loaded, not whether the web server
> answered.**

A health check that returns 200 as soon as the HTTP layer is up will mark the
container ready during the ~24 s CLIP is still loading. Traffic arrives, every
request 500s, and the deploy reports success. Health must mean "can actually
serve a request", which is why it reads the same `state` the handlers do.

It deliberately does **not** run an inference. A probe hit every 30 s that burns
110 ms of CPU competes with real traffic and makes the health check itself a
source of load.

It also returns model provenance (backbone, head, training size, best epoch).
When production behaves oddly, the first question is always "which model is
actually running?" and this answers it without shell access.

---

## 5. `def` vs `async def` — the counterintuitive one

`/predict` is a plain `def`. This is deliberate and it is a common interview
question.

- FastAPI runs a plain **`def`** handler in a worker thread from its threadpool.
- An **`async def`** handler runs directly on the single event loop.

Model inference is blocking CPU work with no `await` points. In an `async def`,
it would stall the event loop for its entire duration — every other in-flight
request, including `/health`, freezes behind it. One slow prediction takes down
the readiness probe and the platform restarts your container.

> **For blocking work, the non-async handler is the concurrent one.**

Use `async def` only when the body is genuinely awaiting I/O (a database driver,
an HTTP client). Use `def` for CPU-bound or blocking-library work.

---

## 6. Error handling: status codes are an API contract

| condition | status | why |
|---|---|---|
| model not loaded | **503** | not 500 — tells a load balancer to retry elsewhere rather than flagging a bug |
| non-image content type | **415** | unsupported media type |
| undecodable bytes | **400** | client's fault. Letting PIL's `UnidentifiedImageError` escape gives a **500** and pages someone for a bad upload |
| > 10 MB | **413** | the memory guard |
| missing field / bad threshold | **422** | pydantic, automatic, with field detail |

The 400-vs-500 distinction is the one that matters operationally: 5xx should
mean *you* are broken, because that is what alerts on.

---

## 7. `src/inference.py` — the anti-drift module

Everything that must match between training and serving lives in one place:

- the **backbone identity is read from the checkpoint's own metadata**, not from
  `config.py`. Edit `config.py` to a different backbone and this still loads the
  one the head was trained against;
- the **preprocessing transform comes from the model object**, so resize, crop
  and normalisation constants cannot drift;
- **L2 normalisation is applied if and only if** the checkpoint says it was
  applied during training.

> Feature-pipeline drift between training and serving is the single most common
> way an ML system produces confidently wrong answers — and it never raises an
> exception.

### Dropping the text tower

CLIP is *two* encoders trained jointly. We classify images and never embed a
single string, so the ~63M-parameter text transformer plus its 49k-token
embedding table is dead weight.

```python
backbone = clip_model.visual
del clip_model
```

Verified exact, not approximate: `encode_image()` is literally
`self.visual(image)` when `normalize=False`, and a direct comparison gave
`max abs diff = 0.0`, `torch.equal → True`.

**Measured effect: container RSS fell from 1,696 MB to 729 MB** — far more than
the ~250 MB of weights predicted, because deleting the parent model also
releases the loader's transient buffers and lets the allocator return pages.
Median latency also improved from ~180 ms to ~112 ms.

### `TORCH_THREADS`

torch sizes its thread pool from the *host's* visible cores. In a 1-vCPU
container it will still spawn ~8 threads to fight over one core — slower than
single-threaded, and more memory. The env var lets deployment match the CPU it
actually paid for.

---

## 8. Testing

`tests/test_api.py` — 11 cases via `TestClient`, which drives the ASGI app
directly without binding a socket.

**The gotcha:** `TestClient` must be used as a context manager
(`with TestClient(app) as c:`) or the lifespan handler never runs, the model is
never loaded, and every request returns 503. Genuinely confusing the first time.

Covered: health reports a loaded model; both fixtures classify correctly;
threshold changes the label but *not* `p_ai` (catches the threshold leaking into
inference); corrupt bytes → 400; wrong content type → 415; missing field → 422;
oversized → 413; greyscale/RGBA/palette images all work; frontend served at `/`.

They take ~13 s because loading CLIP is slow. That is the honest cost of testing
what you ship instead of a mock.

---

## 9. Split requirements files

`requirements-serve.txt` excludes `datasets`, `pandas`, `pyarrow` and
`scikit-learn` — hundreds of MB the server never imports. `python-multipart` is
a hard dependency despite never being imported by name: FastAPI raises at import
time without it when any route uses `UploadFile`.
