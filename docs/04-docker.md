# Stage 4 — Docker

Files: `Dockerfile`, `.dockerignore`

---

## 1. What each instruction actually does

A Docker image is a stack of **immutable filesystem layers**. Most instructions
add one layer containing the changes it made. A container is that stack plus a
thin writable layer on top.

| instruction | what it does |
|---|---|
| `FROM` | the starting filesystem — someone else's image as your base layer |
| `WORKDIR` | sets the working directory for later instructions (and creates it) |
| `COPY` | copies from the **build context** into the image, as a new layer |
| `RUN` | executes a command *at build time* and commits the resulting filesystem changes as a layer |
| `ENV` | sets an environment variable for **all following instructions and the running container** |
| `USER` | switches the user for later instructions and at runtime |
| `EXPOSE` | pure documentation — publishes nothing |
| `CMD` | the default command run when the container starts. Not executed at build time |

Two consequences that cause most Docker confusion:

**Layers are immutable and cumulative.** Deleting a file in layer 8 does not
reclaim the bytes it occupied in layer 3 — it only hides it. The image still
carries both. Cleanup must happen *in the same `RUN`* that created the mess, or
in an earlier stage that gets discarded.

**`ENV` is positional.** It applies to everything after it, which is a real
ordering hazard — see the mistake in §4.

---

## 2. Base image choice

| option | size | verdict |
|---|---|---|
| `python:3.14` | ~1 GB | Full Debian with compilers and headers. Convenient, and mostly wasted — every dependency here ships a prebuilt manylinux wheel, so nothing compiles |
| **`python:3.14-slim`** | ~150 MB | Debian minus docs, headers, dev tooling. Wheels install fine. **Chosen** |
| `python:3.14-alpine` | ~50 MB | The wrong answer. Alpine uses **musl** libc; PyTorch publishes no musl wheels, so pip falls back to building torch from source — hours, if it succeeds |
| distroless | smallest | More locked down, but **no shell**, so `docker exec` debugging is impossible. Not worth it for an image dominated by 1 GB of torch either way |

The alpine trap is worth remembering: "smaller base image" is the usual advice,
and for anything depending on scientific Python it is actively wrong.

---

## 3. Multi-stage build and layer caching

```dockerfile
FROM python:3.14-slim AS builder
RUN python -m venv /opt/venv
COPY requirements-serve.txt .          # <- requirements BEFORE source
RUN pip install ... -r requirements-serve.txt

FROM python:3.14-slim AS runtime
COPY --from=builder /opt/venv /opt/venv
```

**Why requirements are copied before the source.** Docker caches each
instruction against the files it touched. Editing `src/api.py` invalidates only
the layers after the source `COPY`. If everything were copied first, every
one-line source edit would reinstall a gigabyte of torch.

**Why the CPU index URL matters enormously:**

```dockerfile
--extra-index-url https://download.pytorch.org/whl/cpu
```

Plain `pip install torch` pulls the CUDA build and drags in ~2.5 GB of `nvidia-*`
wheels — completely dead weight on a CPU host. The `+cpu` variant is roughly a
fifth of the size.

---

## 4. The mistake: `ENV` ordering

First build failed:

```
RuntimeError: Failed to download weights for tag 'laion2b_s34b_b79k':
  ... we cannot find the requested files in the local cache.
  Please check your connection
```

Cause: `HF_HUB_OFFLINE=1` was declared in the `ENV` block *above* the `RUN` that
downloads the weights. `ENV` applies to every following instruction, so the
download step ran with networking disabled by its own configuration.

Fix: set it *after* the download.

```dockerfile
RUN python -c "...create_model_and_transforms(...)"   # downloads
ENV HF_HUB_OFFLINE=1                                   # only now
```

Which is also exactly what you want at runtime: any accidental network fetch
fails loudly at startup instead of silently working in dev and hanging in
production.

---

## 5. The 1.6 GB mistake: `chown -R` on a large tree

The first working image was **3.95 GB**. `docker history` showed why:

```
606MB   RUN useradd ... && chown -R appuser:appuser /app /opt/hf-cache
606MB   RUN python -c "...create_model_and_transforms..."
1.13GB  COPY /opt/venv /opt/venv
```

The weights appear **twice**. Changing a file's ownership rewrites it into the
new layer, so a recursive chown over a 606 MB cache stores a second complete
copy — and because layers are immutable, the original stays underneath.

Fix: create the user and switch to it *before* generating the files, so they are
written already owned correctly.

```dockerfile
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app /opt/hf-cache \
    && chown appuser:appuser /app /opt/hf-cache    # empty dirs, not the tree
USER appuser
RUN python -c "...downloads weights as appuser..."
COPY --chown=appuser:appuser src/ ./src/
```

> **Never `chmod`/`chown` a large tree after the fact. Create it correctly.**

Plus trimming bytecode caches and torch's bundled C++ test suite **in the
builder stage**, where the deletions never reach a shipped layer:

**3.95 GB → 2.35 GB.**

---

## 6. Baking the weights in vs downloading at startup

Downloading on first boot makes the image ~600 MB smaller. It is still the wrong
trade for a service:

- every cold start pulls 600 MB before it can answer a request, so the health
  check fails for a minute or more and the platform may kill the container as
  unhealthy **before it ever becomes ready**;
- a HuggingFace outage or rate-limit becomes *your* production outage, in a
  service that otherwise has no runtime network dependency;
- the running image stops being reproducible — the same tag could resolve
  different weights later.

Baked in, with `HF_HUB_OFFLINE=1` as the tripwire.

---

## 7. Other production details

**Non-root user.** If an attacker gets code execution through, say, an
image-parsing bug in Pillow, this is the difference between compromising a
process and compromising the container.

**`$PORT`.** Render, Fly and Cloud Run inject the port to bind and consider the
deploy failed if the process hardcodes 8000. Defaulted so `docker run` stays
simple locally.

**Exec form and signal handling.**

```dockerfile
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
```

Shell form (`CMD uvicorn ...`) wraps the process in `/bin/sh -c`, which does
**not** forward `SIGTERM`. The platform's graceful-shutdown signal is swallowed
and every deploy ends in a 10-second timeout followed by `SIGKILL`. We still
need `sh -c` to expand `$PORT`, so `exec` replaces the shell with uvicorn as
PID 1 — getting variable expansion *and* correct signal delivery.

**`HEALTHCHECK`** with `--start-period=90s`. Below the model load time, Docker
marks the container unhealthy during a perfectly normal startup.

**`.dockerignore` is a correctness control, not just a speed one.** It excludes
`data/` (14,000 files that would be sent to the daemon on every build) — but
more importantly, anything not excluded can be silently baked into a layer by a
broad `COPY . .`, including `.env` files and credentials.

---

## 8. Verification

| | local | container |
|---|---|---|
| `real_sample.png` `p_ai` | 0.000240225418 | 0.000240225418 |
| `ai_sample.png` `p_ai` | 0.999616980553 | 0.999616980553 |

Identical to 12 decimal places.

```
image size    2.35 GB
RSS            729 MB
cold start     ~24 s to healthy
latency        min 94 ms · median 112 ms · max 134 ms
```

`scripts/smoke_test.py` — 9 checks against the live container, all passing.
