# AI-Generated Image Detector

**Live: https://zeroai.fly.dev**

Upload an image, get `real` or `ai` plus a confidence score. A frozen CLIP
backbone with a small trained classification head, served by FastAPI, packaged
in Docker, deployed to Fly.io, gated by GitHub Actions.

**95.4% accuracy on held-out CIFAKE.** Please read
[Known limitations](#known-limitations) before believing that number means what
it looks like it means — it is an upper bound on a narrow benchmark, not an
estimate of real-world performance.

---

## Architecture

```
                    browser  ──  frontend/index.html
                       │         file input → FormData → fetch()
                       │
                       │  POST /predict   multipart/form-data
                       ▼
   ┌─────────────────────────────────────────────────────────┐
   │  FastAPI + uvicorn                          src/api.py  │
   │                                                          │
   │    GET  /health    readiness + model provenance          │
   │    POST /predict   image → label, confidence, p_ai       │
   │    GET  /          static frontend (same origin)         │
   │                                                          │
   │  model loaded ONCE in the lifespan handler, not per-req  │
   └────────────────────────┬─────────────────────────────────┘
                            │  Detector.predict()
                            ▼        src/inference.py
   ┌─────────────────────────────────────────────────────────┐
   │  preprocess    resize 224 · center crop · normalise      │
   │       │        (taken from open_clip, never hand-rolled) │
   │       ▼                                                  │
   │  CLIP ViT-B/32 visual tower      FROZEN     88M params   │
   │       │                          text tower discarded    │
   │       ▼        512-d embedding, L2-normalised            │
   │  MLP head      512 → 256 → ReLU → Dropout → 1            │
   │       │                          TRAINED  131,585 params │
   │       ▼        raw logit                                 │
   │  sigmoid  →  p(ai)  →  threshold  →  label               │
   └─────────────────────────────────────────────────────────┘
```

The training pipeline is offline and separate. Because the backbone is frozen,
its output for an image never changes, so embeddings are computed **once** and
cached — turning head training into a 60-second tabular problem:

```
  CIFAKE (HuggingFace)
        │  src/download_data.py      balanced subset → PNGs on disk
        ▼
  data/cifake/{train,test}/{real,ai}/
        │  src/extract_embeddings.py  frozen CLIP, one pass, ~25 img/s CPU
        ▼
  data/embeddings/{train,test}.npz    (N, 512) float32 + labels + provenance
        │  src/train_head.py          15% stratified val split, best-checkpoint
        ▼
  artifacts/head.pt                   weights + architecture + serving contract
        │  src/eval.py                sealed test set, touched exactly once
        ▼
  artifacts/eval-head.json
```

---

## Measured numbers

| | |
|---|---|
| Test accuracy (4,000 held-out images) | 95.40% |
| ROC-AUC | 0.9908 |
| Expected calibration error | 0.0141 |
| False positives / false negatives | 75 / 109 |
| Docker image | 2.35 GB |
| Peak memory at startup | 1,507 MB |
| Steady-state memory | 1,283 MB |
| Cold start to healthy | ~20 s |
| Inference latency (median, local container) | 112 ms |

Peak is the number that sizes a machine — the first deploy was OOM-killed on a
1 GB VM despite `docker stats` reporting 706 MB, because `docker stats`
subtracts page cache and the OOM killer does not. See
[`docs/05-deploy-ci.md`](docs/05-deploy-ci.md) §0.

---

## Run it locally

Requires Python 3.12+ and about 4 GB of disk (CLIP weights + dataset).

### Serve the trained model

`artifacts/head.pt` is committed, so you can skip training entirely.

```bash
python -m venv .venv && .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-serve.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

uvicorn src.api:app --reload --port 8000
```

- UI — <http://127.0.0.1:8000/>
- Interactive API docs — <http://127.0.0.1:8000/docs>
- Health — <http://127.0.0.1:8000/health>

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@tests/fixtures/ai_sample.png"
# {"label":"ai","confidence":0.9996,"p_ai":0.9996,"threshold":0.5,"inference_ms":90.6,...}

# The threshold is a product decision, not a model property:
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.99" -F "file=@image.png"
```

### Retrain from scratch

```bash
pip install -r requirements-train.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

python -m src.download_data          # ~1 min   14,000 PNGs
python -m src.extract_embeddings     # ~10 min  frozen CLIP, one pass
python -m src.train_head             # ~60 s
python -m src.eval                   # sealed test set

python -m src.train_head --hidden 0 --lr 1e-2 --out linear.pt   # linear baseline
```

### Docker

```bash
docker build -t ai-image-detector .
docker run -p 8080:8000 ai-image-detector
python scripts/smoke_test.py http://127.0.0.1:8080
```

### Tests

```bash
pytest tests/ -v                                  # 11 in-process API tests
python scripts/smoke_test.py http://127.0.0.1:8080  # against a live container
ruff check . && ruff format --check .
```

---

## How deployment works

**Target: Fly.io**, 1 GB `shared-cpu-1x`, config in [`fly.toml`](fly.toml).
Chosen against the measured 729 MB footprint — Render's free and Starter tiers
are 512 MB and would be OOM-killed mid-load. The full comparison, including
Modal and HuggingFace Spaces, is in the `fly.toml` header comment.

`.github/workflows/ci.yml` runs four jobs on push to `main`:

```
  lint  ─┐
  test  ─┼─→  deploy      (only on push to main, never on pull_request)
  smoke ─┘
```

- **lint** — `ruff check` + `ruff format --check`
- **test** — 11 pytest cases in-process, with the HuggingFace weight cache keyed
  on the checkpoint name
- **smoke** — builds the image, runs it under `--memory=1g` (matching the Fly VM
  so an OOM shows up in CI rather than production), and hits the real HTTP
  surface via `scripts/smoke_test.py`
- **deploy** — `flyctl deploy --remote-only`, then re-runs the smoke test
  against the live URL

The deploy is an **explicit CLI step**, not Render-style git integration or a
webhook. Git integration builds in parallel with CI, so a failing test does not
stop it. A webhook returns as soon as the build is *queued*. `flyctl deploy`
blocks until health checks pass, so a green pipeline means the new version is
actually serving.

**Live at https://zeroai.fly.dev** — Fly app `zeroai`, region `syd`,
shared-cpu-1x / 2 GB, scale-to-zero. The first request after an idle period
wakes a suspended machine and takes a few seconds; subsequent ones are fast.

---

## Known limitations

**Be skeptical of the 95.4%.** It is a real number, honestly measured on a
sealed test split that was touched exactly once — and it still overstates
real-world performance, for reasons baked into the dataset rather than fixable
by better training.

**The generalization gap.** CIFAKE is 32×32 pixels. CLIP expects 224×224, so
every image is bicubically upscaled 7× before the model sees it. The
high-frequency artifacts that real AI-detectors depend on — upsampling residue,
frequency-domain fingerprints, compression traces — are *physically destroyed*
before inference begins. The model cannot be using them, so whatever it learned
is something else, and that something else is unlikely to transfer to a 1024px
image from a modern generator.

**One generator, one source.** Every "fake" came from Stable Diffusion 1.4;
every "real" from CIFAR-10. The model has plausibly learned "looks like SD 1.4
at low resolution" and "looks like CIFAR-10", not "is synthetic". Expect
materially worse results on Midjourney, DALL·E, Flux, SDXL, or any 2024+ model.

**A balanced test set flatters precision.** Test is 50/50 by construction. If
real traffic is 5% AI-generated, the same error profile produces far more false
accusations than true catches, and precision collapses while accuracy is
unchanged. Accuracy on a balanced set does not transfer to an unbalanced
deployment.

**The threshold has not been chosen for you.** The default is 0.5, which
optimizes nothing in particular. At 0.99 precision reaches 0.9986 but recall
falls to 0.7295. Pick against the actual cost of a false accusation in your
application; the sweep is in `artifacts/eval-head.json`.

**Adversarially trivial.** No robustness work was done. Mild JPEG
recompression, resizing, or noise would likely move predictions substantially,
and nothing here resists someone actively trying to evade it.

**What would actually improve it:** train on a high-resolution multi-generator
dataset, keep native resolution instead of upscaling, add JPEG/resize
augmentation, and evaluate on a *held-out generator* the model never saw during
training — that last one is the only honest test of whether it detects
"synthetic" rather than "SD 1.4".

---

## Learning log

This was built as a learn-the-stack project, so the reasoning behind each
decision is written up rather than left in commit messages:

- [`docs/00-overview.md`](docs/00-overview.md) — working rules, stack table, stage checklist
- [`docs/01-model.md`](docs/01-model.md) — frozen backbones, loss curves, the MLP-vs-linear experiment
- [`docs/02-api.md`](docs/02-api.md) — FastAPI, sync vs async, startup loading
- [`docs/03-frontend.md`](docs/03-frontend.md) — CORS, multipart uploads
- [`docs/04-docker.md`](docs/04-docker.md) — base images, layer caching, the 1.6 GB mistake
- [`docs/05-deploy-ci.md`](docs/05-deploy-ci.md) — platform tradeoffs, workflow structure
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — **deployment runbook**: setup, deploy, operate, troubleshoot, tear down

## Layout

```
src/            download_data · extract_embeddings · train_head · eval
                model · inference · api
frontend/       index.html   single page, no framework
tests/          test_api.py + fixtures
scripts/        smoke_test.py
artifacts/      head.pt (committed) · eval + history JSON (generated)
data/           dataset + cached embeddings (generated, gitignored)
```

## License

MIT. CIFAKE is CC BY 4.0 (Bird & Lotfi, 2023). CLIP weights are LAION's
`ViT-B-32 / laion2b_s34b_b79k`.
