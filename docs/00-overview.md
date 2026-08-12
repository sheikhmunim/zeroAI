# AI-Generated Image Detector — Learning Log

A build-to-learn project: a binary classifier that takes an image and answers
"real photograph" or "AI-generated", wrapped in a real production stack.

The goal is **not** a working demo. The goal is understanding every layer well
enough to defend it in an interview or debug it at 2am.

---

## Working rules for this project

These are deliberate constraints on how the work proceeds:

1. Before any new piece is written (model, API route, Docker config, CI
   workflow), explain in plain terms what it does and why it's needed.
2. After writing it, call out the 2–3 things worth understanding — what breaks
   if misconfigured, what's a common production/interview question, what was a
   judgment call rather than a fixed rule.
3. Never silently pick between meaningfully different approaches. State the
   options and the reasoning.
4. **Stop at the end of each stage.** No chaining model → API → frontend →
   deploy → CI in one uninterrupted run. Each stage gets tested by hand before
   the next begins.
5. "Why?" about something already built is a real question, and gets a real
   answer — not the code restated in English.

This is slower than shipping it. That's the point.

---

## Target stack

| Layer | Choice | Rationale |
|---|---|---|
| Model | Frozen CLIP ViT-B/32 + trainable MLP head | Cheap to train, no GPU needed for the head |
| Serving | FastAPI | A real API with a separate frontend, not a bundled Gradio demo |
| Frontend | Plain HTML/JS, single page | No framework until the backend is solid |
| Container | Docker | — |
| Hosting | Render / Fly.io / Modal (decided in Stage 5) | CPU inference, free-or-cheap tier |
| CI/CD | GitHub Actions | lint + smoke test on push, auto-deploy on merge to main |

---

## Stage progress

- [x] **Stage 1a** — environment, dataset download script
- [x] **Stage 1b** — embedding extraction (frozen backbone)
- [x] **Stage 1c** — MLP head training  (`src/model.py`, `src/train_head.py`)
- [x] **Stage 1d** — evaluation  (`src/eval.py`) — 95.40% test accuracy
- [x] **Stage 2** — FastAPI backend (`POST /predict`, `GET /health`)
- [x] **Stage 3** — HTML/JS frontend + CORS
- [x] **Stage 4** — Dockerfile — 2.35 GB image, 729 MB RSS
- [x] **Stage 5** — deploy config (`fly.toml`) — **written, not deployed**
- [x] **Stage 6** — GitHub Actions CI/CD — **written, never run**
- [x] README with architecture diagram + honest limitations

Detailed notes per stage:
- [`01-model.md`](01-model.md) — Stage 1: frozen backbones, loss curves, MLP vs linear
- [`02-api.md`](02-api.md) — Stage 2: FastAPI, sync vs async, startup loading
- [`03-frontend.md`](03-frontend.md) — Stage 3: CORS, multipart uploads
- [`04-docker.md`](04-docker.md) — Stage 4: base images, layers, the 1.6 GB mistake
- [`05-deploy-ci.md`](05-deploy-ci.md) — Stages 5–6: platform tradeoffs, workflow structure

## Deployment status

Repo is live at **https://github.com/sheikhmunim/zeroAI** (branch `main` —
renamed from `master`, because `ci.yml` triggers on `main` and would otherwise
sit silently doing nothing).

CI has run twice:

| run | lint | test | smoke | deploy | |
|---|---|---|---|---|---|
| 1 | ✅ | ❌ | ✅ | skipped | `ModuleNotFoundError` — see below |
| 2 | ✅ | ✅ | ✅ | ❌ | expected: Fly not configured yet |

**Run 1's failure is the most useful thing in this whole file.** `python -m
pytest` inserts the current directory into `sys.path`; the bare `pytest`
console script does not. Every local run had used the `-m` form, so
`from src.api import app` resolved and the suite passed — for a reason that had
nothing to do with the code. The tests were never portable. Fixed with
`pythonpath = ["."]` in `pyproject.toml` rather than by changing the CI command,
so both invocations behave identically.

**Run 2's deploy failure corrects a wrong assumption:** a missing secret does
*not* skip a job, it fails it. `secrets` is not available in a job-level `if`
at all. Deploy was skipped in run 1 only because the `needs:` gate blocked it
behind the failing test job.

### Remaining to go live

1. `flyctl auth login` (Fly requires a payment method on file)
2. `flyctl launch --no-deploy` — **say yes to the existing `fly.toml`**; it
   carries the 1 GB sizing, the 90 s health-check grace period and
   `TORCH_THREADS=1`, all of which came from measurement. App names are
   globally unique, so pick one and commit it back to `fly.toml` (CI reads the
   verify URL from there).
3. `flyctl deploy --remote-only` — by hand, watched, before CI ever does it
4. `flyctl auth token` → GitHub repository secret `FLY_API_TOKEN`
5. Push a small change and watch the full pipeline deploy unattended
6. **`flyctl apps destroy <app>` when finished.** Fly bills hourly; a few days
   costs cents, leaving it up indefinitely is ~$5.70/mo.

Nothing is currently billing. Nothing is time-sensitive.

## Note on how stages 1d–6 were built

Stages 1d through 6 were completed autonomously in one pass at the user's
request, rather than with the stop-and-test rhythm above. The explanations were
written into these docs as the work happened, so the reasoning is preserved —
but the hands-on debugging that makes it stick was skipped by design.

---

## Environment

- Windows 11, Python 3.14.6, venv at `.venv/`
- **CPU-only PyTorch** (2.13.0+cpu) — deliberate; see `01-model.md`
- A GTX 1660 Ti (6 GB) is present but unused. All code is device-agnostic.

```powershell
# one-time setup
uv venv .venv --python 3.14
uv pip install -r requirements-train.txt `
    --extra-index-url https://download.pytorch.org/whl/cpu `
    --index-strategy unsafe-best-match
```

Run everything as a module from the project root so `src.` imports resolve:

```powershell
.\.venv\Scripts\python.exe -m src.download_data
.\.venv\Scripts\python.exe -m src.extract_embeddings
```
