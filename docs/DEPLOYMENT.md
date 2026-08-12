# Deployment Runbook — A to Z

Everything needed to deploy this project, operate it, debug it, and shut it
down completely.

This is the **how**. For the **why** — platform comparison, workflow design,
and the postmortem on what broke — see [`05-deploy-ci.md`](05-deploy-ci.md).

| | |
|---|---|
| Live URL | https://zeroai.fly.dev |
| Platform | Fly.io |
| App name | `zeroai` |
| Region | `syd` (Sydney) |
| Machine | shared-cpu-1x, 1 vCPU, **2 GB** |
| Repo | https://github.com/sheikhmunim/zeroAI |
| Auto-deploys | on push to `main`, gated by CI |

> **Windows note.** In the Claude Code session, `!` runs **Bash** (Git Bash), so
> use `/c/Users/Munim/.fly/bin/flyctl.exe`. In a normal PowerShell window
> `flyctl` works directly. All commands below are written as `flyctl`.

---

# Contents

- [1. What you are deploying](#1-what-you-are-deploying)
- [2. Prerequisites](#2-prerequisites)
- [3. One-time setup](#3-one-time-setup)
- [4. fly.toml, line by line](#4-flytoml-line-by-line)
- [5. First manual deploy](#5-first-manual-deploy)
- [6. Verify it works](#6-verify-it-works)
- [7. Automated deploys via GitHub Actions](#7-automated-deploys-via-github-actions)
- [8. Day-2 operations](#8-day-2-operations)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Cost](#10-cost)
- [11. Teardown — full shutdown](#11-teardown--full-shutdown)
- [12. Redeploying from scratch later](#12-redeploying-from-scratch-later)

---

# 1. What you are deploying

A single Docker container that serves both the API and the frontend from one
origin.

```
   https://zeroai.fly.dev
            │
            ├── /              frontend/index.html  (+ /samples/*.png)
            ├── /docs          auto-generated OpenAPI page
            ├── /health        readiness probe (used by Fly and CI)
            └── /predict       POST an image → {label, confidence, p_ai, ...}
                    │
                    └── frozen CLIP ViT-B/32 visual tower + trained MLP head
```

### Resource requirements — measured, not guessed

| | |
|---|---|
| Image size | 2.35 GB local / 779 MB in Fly's registry (compressed) |
| **Peak memory at startup** | **1,507 MB** ← this sizes the machine |
| Steady-state memory | 1,283 MB |
| Startup to healthy | ~20 s |
| Inference | ~112 ms local, 300–500 ms on 1 shared vCPU |

> **Size the machine on PEAK, not steady state.** A 1 GB machine OOM-kills in a
> restart loop. `docker stats` reports 706 MB because it subtracts page cache;
> the kernel's OOM killer does not. Read `/sys/fs/cgroup/memory.peak` instead.

---

# 2. Prerequisites

- A **Fly.io account** — https://fly.io/app/sign-up. A payment method may be
  required; a trial covers initial usage. With no card on file the worst case
  is the app stopping, not a bill.
- A **GitHub account** with the repo pushed (only needed for automated deploys).
- **`artifacts/head.pt` committed to the repo.** The Docker build copies it in.
  It is 545 KB and deliberately tracked in git.
- **Docker** — only for local testing. Fly builds remotely.

---

# 3. One-time setup

## 3.1 Install flyctl

**Windows (PowerShell):**
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

**macOS / Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

Installs to `~/.fly/bin` and adds it to PATH. **Open a new terminal afterwards**
— PATH changes do not apply to already-running shells.

```powershell
flyctl version
```

## 3.2 Sign in

```powershell
flyctl auth login
```

Opens a browser. Sign in or sign up, authorise the CLI. A token is written to
`~/.fly/config.yml` and reused by every later command.

```powershell
flyctl auth whoami        # confirms the signed-in email
```

## 3.3 Create the app

`--no-deploy` creates the app record without building anything, so the deploy
can be watched separately.

```powershell
cd D:\A.U.R.A\AI-Image-detector
flyctl launch --no-deploy
```

**Answer carefully:**

| prompt | answer | why |
|---|---|---|
| Copy existing `fly.toml` configuration? | **Yes** | it carries the memory sizing, grace period and thread count |
| App name | globally unique across all of Fly | becomes `https://<name>.fly.dev` |
| Organization | your personal one | |
| Region | nearest to your users | latency only; `flyctl platform regions` lists them |
| Postgres / Redis / Tigris / Sentry | **No** to all | this app has no database and they cost money |

> ⚠️ **`flyctl launch` rewrites `fly.toml` and deletes every comment**, while
> preserving the functional settings. Recover them with `git diff fly.toml`.
> It also switches double quotes to single quotes — which is what broke the
> CI URL extraction (see §9.5).

Then re-check the config and commit the app name:

```powershell
flyctl config validate
git add fly.toml
git commit -m "Set Fly app name and region"
```

---

# 4. fly.toml, line by line

Every setting that is not a default exists for a measured reason.

```toml
app = 'zeroai'
primary_region = 'syd'          # latency only; free to change before first deploy
```

```toml
[build]
  dockerfile = 'Dockerfile'     # Fly builds this; it does not read your repo
```

```toml
[env]
  PORT = '8000'                 # the Dockerfile CMD binds ${PORT}
  ALLOWED_ORIGINS = '*'         # CORS; irrelevant in prod (same origin)
  TORCH_THREADS = '1'           # 1 vCPU. Unset, torch spawns ~8 threads to
                                # fight over one core: slower AND more memory
```

```toml
[http_service]
  internal_port = 8000
  force_https = true

  auto_stop_machines = 'suspend'   # scale to zero when idle
  auto_start_machines = true       # wake on incoming request
  min_machines_running = 0
```

**Scale-to-zero tradeoff:** idle cost drops to near nothing, but the first
request after a quiet period pays a cold start (~3–4 s observed). `'suspend'`
snapshots memory rather than killing the machine, so resume is seconds instead
of a full ~20 s model load.

```toml
  [http_service.concurrency]
    type = 'requests'
    soft_limit = 2
    hard_limit = 4
```

Inference is CPU-bound and single-threaded, so requests queue rather than
overlap usefully. Low limits make Fly shed or queue at the edge instead of
letting 50 requests pile into one process and all time out.

```toml
  [[http_service.checks]]
    method = 'GET'
    path = '/health'
    interval = '30s'
    timeout = '5s'
    grace_period = '1m'
```

Fly **ignores the Dockerfile's `HEALTHCHECK`** — this block is what decides
whether a deploy succeeded and whether traffic is routed.

> **1 minute is Fly's hard ceiling** for an HTTP service check grace period.
> Anything larger is silently lowered, with a warning. If startup ever exceeds
> 60 s the fix must be a faster startup — there is no longer grace period
> available.

```toml
[[vm]]
  size = 'shared-cpu-1x'
  cpus = 1
  memory = '2gb'
  memory_mb = 2048
```

**2 GB, not 1 GB.** See §1 and §9.1.

---

# 5. First manual deploy

Do this by hand at least once before letting CI do it. Watching it is the point.

```powershell
cd D:\A.U.R.A\AI-Image-detector
flyctl deploy --remote-only
```

**`--remote-only`** builds the image on Fly's builder instead of your machine.
Only the build context is uploaded (a few MB — `.dockerignore` excludes
`data/`), rather than a 2.35 GB image over your home connection.

**Expect 5–10 minutes on the first run** (no layer cache), under a minute after.

### What you should see, in order

```
==> Building image                          Dockerfile steps stream past
--> Pushing image done                      image is in Fly's registry
    image size: 779 MB
Creating machine / Updating existing machines with rolling strategy
Waiting for machine to become healthy       ← the moment that matters
✓ Machine ... is healthy
Visit your newly deployed app at https://zeroai.fly.dev/
```

### Useful flags

| flag | effect |
|---|---|
| `--remote-only` | build on Fly's builder (recommended) |
| `--local-only` | build locally, upload the image |
| `--wait-timeout 300` | seconds to wait for health checks |
| `--strategy immediate` | skip rolling update; faster, brief downtime |
| `--no-cache` | force a clean rebuild |

---

# 6. Verify it works

**Never trust the deploy command's exit code alone.** A green `flyctl deploy`
means "released", not "serving correctly".

```powershell
# 1. HTTP-level check
curl https://zeroai.fly.dev/health

# 2. Full functional check — the same script CI runs
python scripts/smoke_test.py https://zeroai.fly.dev --timeout 120
```

Expected:

```
9 passed, 0 failed
```

That script checks readiness, the backbone identity reported by `/health`, a
known-real image classifying `real`, a known-AI image classifying `ai`, the
response schema, the probability range, and threshold validation.

Then open **https://zeroai.fly.dev** in a browser. The first load may take a
few seconds if the machine was suspended.

---

# 7. Automated deploys via GitHub Actions

## 7.1 Create a scoped deploy token

```powershell
flyctl tokens create deploy -a zeroai
```

> **Use `tokens create deploy`, not `flyctl auth token`.** The latter gives full
> access to your entire Fly account — every app, plus billing. A deploy token is
> scoped to this one app, so a leak is bounded.

Prints a long string beginning `FlyV1 fm2_...`.

> 🔒 **Never paste this into a file, a commit, a chat, or an issue.** It goes
> from your terminal into GitHub's secret field and nowhere else. Anything
> committed to a public repo is public permanently; the only remedy for a leak
> is revoking the token.

## 7.2 Store it as a GitHub secret

1. https://github.com/sheikhmunim/zeroAI/settings/secrets/actions
2. **New repository secret**
3. Name: `FLY_API_TOKEN` — must match `.github/workflows/ci.yml`
4. Value: the full token including the `FlyV1 ` prefix
5. **Add secret**

GitHub will never display it again — only replace it. That is intentional.

## 7.3 How the pipeline works

```
push to main
     │
     ├── lint     ruff check + ruff format --check          ~10 s
     ├── test     11 pytest cases, real model in-process    ~60 s
     └── smoke    docker build + run under 2 GB + 9 checks  ~2-8 min
              │
              └── deploy   (needs: all three · only push to main)
                      ├── flyctl deploy --remote-only --wait-timeout 300
                      └── Verify production → smoke_test.py against live URL
```

Key behaviours:

- The three checks run **in parallel on separate VMs** — nothing is shared.
- `deploy` is gated by `needs: [lint, test, smoke]`.
- `if: github.event_name == 'push' && github.ref == 'refs/heads/main'` prevents
  a pull request from a fork ever deploying.
- **A missing secret does not skip the deploy job — it fails it.** `secrets` is
  not available in a job-level `if` at all.
- The production URL is parsed from `fly.toml` with `tomllib`, so it cannot
  drift from the app name.

## 7.4 Triggering a deploy

```powershell
git add .
git commit -m "your change"
git push origin main
```

Watch at **https://github.com/sheikhmunim/zeroAI/actions**.

Note that *any* push to `main` deploys, including documentation-only changes.
To avoid that, add to `ci.yml`:

```yaml
on:
  push:
    branches: [main]
    paths-ignore: ['**.md', 'docs/**']
```

---

# 8. Day-2 operations

### Status and logs

```powershell
flyctl status -a zeroai                 # app + machine states + health checks
flyctl logs -a zeroai                   # live tail (Ctrl-C to exit)
flyctl checks list -a zeroai            # health check detail
flyctl dashboard -a zeroai              # open the web UI
```

### Machines

```powershell
flyctl machine list -a zeroai
flyctl machine status <machine-id> -a zeroai
flyctl machine restart <machine-id> -a zeroai
flyctl machine stop <machine-id> -a zeroai
flyctl machine start <machine-id> -a zeroai
```

### Shell into the running container

```powershell
flyctl ssh console -a zeroai
```

Useful once inside:

```bash
cat /sys/fs/cgroup/memory.peak      # actual peak memory
cat /sys/fs/cgroup/memory.current   # current, including page cache
ls -la /app/artifacts               # is head.pt actually there?
env | sort                          # what env vars did the app really get?
```

### Change memory or CPU

```powershell
flyctl scale show -a zeroai
flyctl scale memory 4096 -a zeroai        # MB
flyctl scale vm shared-cpu-2x -a zeroai
```

Keep `fly.toml` in sync, or the next `flyctl deploy` reverts it.

### Change region

```powershell
flyctl platform regions                    # list options
```

Edit `primary_region` in `fly.toml`, then redeploy. Existing machines do not
move on their own — destroy and recreate, or use `flyctl machine clone`.

### Environment variables and secrets

Non-sensitive values live in `fly.toml` under `[env]`. Sensitive values:

```powershell
flyctl secrets set SOME_KEY=value -a zeroai     # triggers a restart
flyctl secrets list -a zeroai                   # names only, never values
flyctl secrets unset SOME_KEY -a zeroai
```

### Rollback

```powershell
flyctl releases -a zeroai                  # list versions and image refs
flyctl deploy --image registry.fly.io/zeroai:deployment-<ID>
```

Deploying a previous image is faster and safer than reverting the commit,
because it ships bytes already known to work.

---

# 9. Troubleshooting

**Always start here:**

```powershell
flyctl logs -a zeroai
flyctl status -a zeroai
```

## 9.1 `Out of memory: Killed process` / machine restart loop

**Symptoms** — the deploy reports *"The app is not listening on the expected
address"*, `flyctl status` shows checks `critical`, the dashboard says
"Machines Restarting a Lot — this is an issue with your application code."

**All of that is misleading.** The app is fine; the machine is too small.

Confirm in the logs:

```
Out of memory: Killed process 642 (uvicorn) total-vm:1361188kB, anon-rss:874836kB
INFO Process appears to have been OOM killed!
```

**Fix:**

```powershell
flyctl scale memory 2048 -a zeroai
```

and set `memory = '2gb'` in `fly.toml` so it survives the next deploy.

**Measure peak properly** — do not size off `docker stats`:

```powershell
docker run -d --name m -p 8099:8000 --memory=4g --memory-swap=4g -e TORCH_THREADS=1 <image>
# wait for startup, then:
docker exec m sh -c "cat /sys/fs/cgroup/memory.peak"
docker rm -f m
```

> `docker run --memory=Xg` **without** `--memory-swap` grants swap equal to
> twice the limit, so the container gets 2X and passes a test it should fail.
> Fly's Firecracker VMs have no swap. Always set both.

## 9.2 Health checks never pass, but the app looks fine

Check that the app binds **`0.0.0.0`**, not `127.0.0.1`. A server on localhost
is unreachable from Fly's proxy. `flyctl` tells you outright:

```
WARNING The app is not listening on the expected address
  - 0.0.0.0:8000
```

Also confirm `internal_port` in `fly.toml` matches the port the process binds,
and that the process reads `$PORT`.

## 9.3 Startup exceeds the grace period

Fly caps `grace_period` at **1 minute** for HTTP service checks. If startup is
slower than that, you cannot buy more time — make startup faster (load fewer
weights, use a smaller backbone) or move the check to a `[[services.tcp_checks]]`
with a longer window.

## 9.4 Deploy succeeds but the site 502s or hangs

Usually a suspended machine waking up (`min_machines_running = 0`). The first
request takes a few seconds. If it persists, `flyctl logs` will show whether the
process is actually running.

## 9.5 `URL can't contain control characters`

A parsing bug, not an infrastructure one:

```
verifying https://app = 'zeroai'.fly.dev
```

`cut -d'"' -f2` splits on double quotes; `flyctl launch` rewrote `fly.toml`
with single quotes. Use a real parser:

```bash
APP=$(python -c "import tomllib;print(tomllib.load(open('fly.toml','rb'))['app'])")
```

## 9.6 CI `deploy` job fails with an auth error

- Is the secret named **exactly** `FLY_API_TOKEN`?
- Was the full value pasted, including the `FlyV1 ` prefix?
- Is the token scoped to the right app? `flyctl tokens list`
- Was the token revoked or expired?

## 9.7 `flyctl launch` wiped my config

`git diff fly.toml` shows exactly what changed. It preserves functional settings
but deletes all comments and may reformat quotes.

---

# 10. Cost

Fly bills **hourly**, not monthly. Roughly, for `shared-cpu-1x` with 2 GB:

| | approximate |
|---|---|
| per hour | ~$0.015 |
| a full day | ~$0.36 |
| a full month | ~$11 |

Scale-to-zero (`min_machines_running = 0`) means an idle app costs close to
nothing beyond rootfs storage — you are billed for the machine while it runs.

**Check current usage:**

```powershell
flyctl dashboard -a zeroai        # then Billing
```

With **no payment method on file**, the account runs on trial credit and the
worst case when it is exhausted is the app stopping — not an unexpected bill.

**To reduce cost:** keep `min_machines_running = 0`, use the smallest memory
that clears peak, and destroy the app when you no longer need it (§11).

---

# 11. Teardown — full shutdown

Four levels, from reversible to permanent. **Pick one.**

## Level 1 — Stop the machine (fully reversible)

Keeps the app, config, image and URL. Compute stops; a small rootfs storage
charge remains.

```powershell
flyctl machine list -a zeroai
flyctl machine stop <machine-id> -a zeroai
flyctl status -a zeroai                       # confirm State: stopped
```

Restart with `flyctl machine start <machine-id> -a zeroai`.

> ⚠️ With `auto_start_machines = true`, an incoming HTTP request **wakes a
> stopped machine**. Stopping alone does not reliably stop serving.

To take it out of rotation so traffic cannot wake it, without destroying
anything:

```powershell
flyctl machine cordon <machine-id> -a zeroai     # deactivate its services
flyctl machine stop <machine-id> -a zeroai
```

Reverse with `flyctl machine uncordon <machine-id> -a zeroai`.

## Level 2 — Remove all machines, keep the app

The app name and URL stay reserved; nothing runs and nothing computes.

```powershell
flyctl scale count 0 -a zeroai
flyctl machine list -a zeroai                 # should be empty
```

Bring it back with `flyctl deploy`.

## Level 3 — Destroy the app (recommended when finished)

Removes machines, config, images and the hostname. **Irreversible.** The app
name is released and someone else may claim it.

```powershell
flyctl apps destroy zeroai
```

It prompts for confirmation. Add `-y` to skip (be certain).

Verify:

```powershell
flyctl apps list          # zeroai should be gone
```

`https://zeroai.fly.dev` will stop resolving.

## Level 4 — Clean up everything else

Destroying the app does **not** revoke tokens or remove CI configuration. Finish
the job:

**a) Revoke the deploy token**

```powershell
flyctl tokens list
flyctl tokens revoke <token-id>
```

**b) Delete the GitHub secret**

https://github.com/sheikhmunim/zeroAI/settings/secrets/actions → `FLY_API_TOKEN`
→ **Remove**

**c) Stop CI from attempting to deploy**

Otherwise every push to `main` produces a failing `deploy` job. Either delete
the `deploy:` job from `.github/workflows/ci.yml`, or disable it:

```yaml
  deploy:
    if: false   # app destroyed; see docs/DEPLOYMENT.md §12 to bring it back
```

**d) Sign out locally (optional)**

```powershell
flyctl auth logout
```

**e) Confirm nothing is billing**

```powershell
flyctl apps list          # no apps
flyctl machine list -a zeroai   # should error: app not found
```

Then check the Billing page on the Fly dashboard for anything unexpected.

## Local cleanup (unrelated to Fly)

```powershell
docker rm -f aid                          # any running local containers
docker rmi ai-image-detector:local        # ~2.35 GB reclaimed
docker system prune -a                    # everything unused (careful)
```

Your dataset and embeddings under `data/` are ~200 MB and regenerable with
`python -m src.download_data` + `python -m src.extract_embeddings`.

---

# 12. Redeploying from scratch later

Everything needed is committed. To bring it back after a Level 3 teardown:

```powershell
git clone https://github.com/sheikhmunim/zeroAI.git
cd zeroAI

flyctl auth login
flyctl launch --no-deploy         # yes to existing fly.toml; pick a name
                                  # commit the new name into fly.toml
flyctl deploy --remote-only
python scripts/smoke_test.py https://<new-name>.fly.dev
```

To re-enable automated deploys:

```powershell
flyctl tokens create deploy -a <new-name>
```

→ add as `FLY_API_TOKEN` in GitHub secrets → remove the `if: false` from the
`deploy` job → push.

No retraining is required: `artifacts/head.pt` is in the repo and the Dockerfile
bakes the CLIP weights into the image at build time.

---

# Quick reference

```powershell
# deploy
flyctl deploy --remote-only

# check
flyctl status -a zeroai
flyctl logs -a zeroai
curl https://zeroai.fly.dev/health
python scripts/smoke_test.py https://zeroai.fly.dev

# operate
flyctl ssh console -a zeroai
flyctl scale memory 2048 -a zeroai
flyctl releases -a zeroai

# stop
flyctl machine stop <id> -a zeroai     # reversible
flyctl scale count 0 -a zeroai         # keep app, no machines
flyctl apps destroy zeroai             # permanent
```
