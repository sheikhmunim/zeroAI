# Stage 5 & 6 — Deployment and CI/CD

Files: `fly.toml`, `.github/workflows/ci.yml`, `scripts/smoke_test.py`

> **Status: configured and locally verified, not deployed.** Everything through
> the container is built and tested. Going live needs `flyctl launch` once and a
> `FLY_API_TOKEN` repository secret.

---

## 1. Platform tradeoff — decided against measured numbers

The measurements that constrain the choice:

```
image size    2.35 GB
steady RSS      729 MB      <- the binding constraint
cold start     ~24 s to healthy
inference     ~112 ms median (multi-threaded)
```

| | fits 729 MB? | cost | notes |
|---|---|---|---|
| **Render** free / Starter | **No** — 512 MB | $0 / $7 | Container is OOM-killed mid-load. Not "slow", *dead* |
| Render Standard | Yes — 2 GB | ~$25/mo | Nicest git integration of the three. Wrong price for a demo |
| **Modal** | Yes | free credits | Serverless, scale-to-zero, best fit for bursty traffic. But Python-native: you define the image in a Python DSL, so **Stage 4's Dockerfile stops being what gets deployed** |
| **Fly.io** | Yes — 1 GB | ~$5.70/mo | Deploys this exact Dockerfile unmodified. Scale-to-zero. `flyctl deploy` is an explicit, legible step |
| HuggingFace Spaces (Docker) | Yes — 16 GB | **$9/mo** | The *hardware* tier (CPU Basic, 2 vCPU / 16 GB) is free, but **creating a Docker Space requires a PRO account**. More expensive than Fly |
| HuggingFace Spaces (Gradio/ZeroGPU) | Yes | free, 2 max | Free only for **Gradio** Spaces — means discarding the FastAPI backend, which this project's stack spec explicitly rejected |
| HuggingFace Spaces (Static) | n/a | free | Static files only. Cannot run a Python backend |
| Google Cloud Run | Yes | ~free at low traffic | Scales to zero, deploys a Dockerfile, generous free tier. Cold start is the catch: pulling a 2.35 GB image from cold is far slower than Fly's `suspend` |

**Chosen: Fly.io** — the cheapest option that keeps the architecture intact. It
runs the artifact we actually built, the memory fits with headroom, and the
deploy is a command you can read and run yourself rather than dashboard magic.

> **Correction.** An earlier version of this file (and the `fly.toml` header)
> claimed HF Spaces was "genuinely free and would fit comfortably." That was
> wrong: free personal accounts can no longer create Docker Spaces at all. The
> free-tier *hardware* is real; the free-tier *account* cannot use it for
> Docker. Verified against `huggingface.co/docs/hub/spaces-overview`,
> August 2026 — worth re-checking, since this is exactly the kind of policy
> that moves.

The 512 MB finding is the whole reason Stage 1 measured CPU throughput instead
of using the GPU. Without a real footprint number, the platform choice would
have been a guess — and the guess (Render free tier) would have failed in a way
that looks like a crash loop rather than a sizing problem.

---

## 2. `fly.toml` details worth knowing

**`TORCH_THREADS = "1"`** — `shared-cpu-1x` is a single vCPU. Left unset, torch
sizes its pool from the visible core count and spawns ~8 threads to fight over
one core: slower than single-threaded, and more memory.

> Expect latency well above the 112 ms measured locally. Single-core inference
> on shared CPU is realistically 300–500 ms.

**Scale to zero** (`auto_stop_machines = "suspend"`, `min_machines_running = 0`)
— idle cost drops to ~$0, the first request after idle pays a cold start.
`suspend` snapshots memory rather than killing the machine, so resume is a few
seconds rather than the full ~24 s model load.

**Low concurrency limits** (`soft_limit = 2`, `hard_limit = 4`) — inference is
CPU-bound and single-threaded here, so requests queue rather than overlap
usefully. Low limits make Fly shed or queue at the edge instead of letting 50
requests pile into one process and all time out.

**`grace_period = "90s"`** on the health check. This must exceed model load time
or Fly kills the machine before it finishes starting — and the failure presents
as a crash loop, not a slow boot, which is a genuinely misleading symptom.

Note Fly ignores the Dockerfile's `HEALTHCHECK`; the `[[http_service.checks]]`
block is the one that counts.

---

## 3. What a GitHub Actions workflow is

A YAML file in `.github/workflows/`. Three levels:

- **`on:`** — triggers. Which repository events start a run.
- **`jobs:`** — independent units of work. Each gets a **fresh VM** ("runner").
  Jobs run in **parallel** unless `needs:` forces an order, and share **nothing**
  except what you explicitly upload/download or rebuild.
- **`steps:`** — ordered commands inside one job, sharing that job's filesystem.
  `uses:` runs a prepackaged action; `run:` runs a shell command.

The "fresh VM, nothing shared" part is the one that surprises people: a file
written in `lint` does not exist in `test`.

### The gate

```
  lint  ─┐
  test  ─┼─→  deploy
  smoke ─┘
```

`needs: [lint, test, smoke]` makes deploy run only if all three pass.

```yaml
if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

A `pull_request` targeting main must **never** deploy — otherwise anyone opening
a PR from a fork ships code to production.

### Concurrency

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

Without this, three quick pushes queue three full deploys that can land **out of
order** — the slowest run finishing last would overwrite newer code.

---

## 4. What the smoke test should check, and why not "just run pytest"

The unit suite proves the **code** is correct in a Python process where imports
resolve and the model file happens to be on disk. It cannot catch the things
that actually break deploys:

- the image built, but `CMD` is wrong and the process exits immediately;
- the server binds 8000 while the platform routes to `$PORT`;
- `artifacts/head.pt` was excluded by `.dockerignore`, so startup crashes;
- `HF_HUB_OFFLINE` is set but weights were never baked in, so startup hangs;
- the container needs more memory than the plan allows and is OOM-killed.

**Every one of those passes pytest and fails in production.**

So `scripts/smoke_test.py` exercises the real HTTP surface of the real artifact:

1. poll `/health` until `model_loaded` is true (or time out)
2. `/health` reports the expected backbone
3. `POST /predict` with a known real image → 200, label `real`
4. `POST /predict` with a known AI image → 200, label `ai`
5. response schema is complete
6. `p_ai` is a number in [0, 1]
7. out-of-range threshold → 422

It **polls rather than sleeps**. Startup is dominated by loading weights, which
varies with disk speed, CPU allocation and cache state — a hardcoded `sleep 30`
is simultaneously too slow on a fast machine and flaky on a slow one.

It asserts on **labels, not accuracy**. A smoke test's job is "the wiring is
intact and predictions are sane". Model quality belongs in `src/eval.py` against
the full test set.

It uses **only the standard library**, so it runs on a bare runner or against
production without installing anything.

### `--memory=1g` in CI

```yaml
run: docker run -d --name detector -p 8080:8000 --memory=1g detector:ci
```

Matching the Fly VM. Without it, the runner's 7 GB hides an OOM that would only
appear in production. Also note `-p 8080:8000` deliberately maps a *different*
external port — if the app ever hardcoded 8000 in a URL, this catches it.

---

## 5. How the deploy is triggered — three models

| model | how | problem |
|---|---|---|
| **Platform git integration** (Render default) | connect the repo in a dashboard; platform builds on every push | runs **in parallel with CI**, not after it. A failing test does not stop the deploy — which defeats the entire point of a gate |
| **Deploy hook / webhook** | CI curls a secret URL | ordering is fixed, but the webhook returns as soon as the build is *queued*. Green pipeline means "asked nicely", not "deployed" |
| **Explicit CLI deploy** ← chosen | `flyctl deploy --remote-only --wait-timeout 300` in the workflow | blocks until health checks pass, so green means **actually live and serving**. Also legible: a command you can read and run yourself |

Then re-run the smoke test against the production URL — the difference between
"the deploy command exited 0" and "production works".

**Secrets** go in Settings → Secrets and variables → Actions, referenced as
`${{ secrets.FLY_API_TOKEN }}`. Never a literal in the file: workflow files are
public in a public repo, and committed history is forever.

---

## 6. Caching

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/huggingface
    key: hf-ViT-B-32-laion2b_s34b_b79k
```

The tests load CLIP for real. Caching keyed on the **checkpoint name** turns a
2-minute download into seconds. Keying on the checkpoint rather than a lockfile
hash means the cache survives unrelated dependency changes, and a real backbone
change naturally busts it.

---

## 7. To actually go live

```bash
flyctl auth login
flyctl launch --no-deploy       # creates the app, reconciles fly.toml
flyctl deploy                   # first manual deploy — watch it once by hand

flyctl auth token               # -> add as FLY_API_TOKEN repo secret
git push origin main            # from here CI does it
```

Deploy manually the first time. Watching `flyctl` build, push and release —
then health-check before shifting traffic — is what makes the workflow step
legible rather than a black box.
