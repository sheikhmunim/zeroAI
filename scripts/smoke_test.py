"""Post-deploy smoke test: is this build actually able to serve?

Run against a live URL -- a container in CI, or a deployed environment.

Why this rather than "just run pytest": the unit suite proves the *code* is
correct in a Python process where imports resolve and the model file happens to
sit on disk. It cannot catch the things that actually break deploys:

  * the image built, but CMD is wrong and the process exits immediately;
  * the server binds 8000 while the platform routes to $PORT;
  * artifacts/head.pt was excluded by .dockerignore, so startup crashes;
  * HF_HUB_OFFLINE is set but the weights were never baked in, so the model
    tries to reach the network and hangs;
  * the container needs more memory than the plan allows and is OOM-killed
    partway through loading.

Every one of those passes a unit test suite and fails in production. This
exercises the real HTTP surface of the real artifact, which is the only thing
that distinguishes "the code is fine" from "the deployment works".

Usage:
    python scripts/smoke_test.py http://127.0.0.1:8080
    python scripts/smoke_test.py https://example.fly.dev --timeout 180
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# Cloudflare's default security rules block the stock "Python-urllib/x.y"
# User-Agent with a 403 before the request ever reaches the tunnel -- looks
# identical to the server being down, except every poll fails instantly
# instead of timing out.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; smoke-test/1.0)"}

passed, failed = 0, 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def wait_for_health(base_url: str, timeout: int) -> dict | None:
    """Poll /health until the model reports loaded, or give up.

    Polling rather than a fixed sleep is the point. Startup here is dominated
    by loading 350 MB of weights, and that time varies with disk speed, CPU
    allocation and cache state -- a hardcoded `sleep 30` is simultaneously too
    slow on a fast machine and flaky on a slow one.
    """
    deadline = time.time() + timeout
    last_error = "no attempt made"

    request = urllib.request.Request(f"{base_url}/health", headers=HEADERS)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    body = json.loads(response.read())
                    if body.get("model_loaded"):
                        return body
                    last_error = "responded but model_loaded=false"
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2)

    print(f"  timed out after {timeout}s waiting for /health -- last: {last_error}")
    return None


def post_image(base_url: str, path: Path, query: str = "") -> tuple[int, dict]:
    """Minimal multipart/form-data POST using only the standard library."""
    boundary = uuid.uuid4().hex
    payload = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
            b"Content-Type: image/png\r\n\r\n",
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        f"{base_url}/predict{query}",
        data=payload,
        headers={**HEADERS, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except json.JSONDecodeError:
            return exc.code, {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--timeout", type=int, default=180, help="seconds to wait for readiness")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"smoke testing {base_url}")

    started = time.time()
    health = wait_for_health(base_url, args.timeout)
    if health is None:
        print("\nFAILED: service never became ready")
        return 1
    print(f"  ready in {time.time() - started:.0f}s")

    check("health reports model loaded", health.get("model_loaded") is True)
    check(
        "health reports the expected backbone",
        health.get("model", {}).get("backbone", "").startswith("ViT-B-32"),
        f"got {health.get('model', {}).get('backbone')!r}",
    )

    status, body = post_image(base_url, FIXTURES / "real_sample.png")
    check("predict accepts a real image", status == 200, f"HTTP {status}")
    check("real image classified 'real'", body.get("label") == "real", json.dumps(body))

    status, body = post_image(base_url, FIXTURES / "ai_sample.png")
    check("predict accepts an AI image", status == 200, f"HTTP {status}")
    check("AI image classified 'ai'", body.get("label") == "ai", json.dumps(body))
    check(
        "response schema is complete",
        all(k in body for k in ("label", "confidence", "p_ai", "threshold", "inference_ms")),
        f"got keys {sorted(body)}",
    )
    check(
        "probability is in range",
        isinstance(body.get("p_ai"), (int, float)) and 0.0 <= body["p_ai"] <= 1.0,
        f"p_ai={body.get('p_ai')}",
    )

    status, _ = post_image(base_url, FIXTURES / "ai_sample.png", "?threshold=1.5")
    check("out-of-range threshold rejected", status == 422, f"HTTP {status}")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
