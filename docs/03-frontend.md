# Stage 3 — Frontend

File: `frontend/index.html` — one page, no framework, no build step.

---

## 1. The flow

```
  <input type="file">
        │  change / drop event
        ▼
  File object
        │  URL.createObjectURL(file)  →  <img src>   local preview, no upload
        │
        │  new FormData(); fd.append('file', file)
        ▼
  fetch('/predict?threshold=0.5', { method: 'POST', body: fd })
        │
        ▼
  await res.json()  →  { label, confidence, p_ai, ... }
        │
        ▼
  render into the DOM
```

Two details worth knowing:

**`URL.createObjectURL(file)`** gives the browser a local handle to the file for
the preview. Nothing is uploaded to produce it — the `<img>` reads straight from
disk.

**Never set `Content-Type` manually when sending `FormData`.** This is the
classic multipart bug. A multipart body is delimited by a random boundary token
that must appear in the header:

```
Content-Type: multipart/form-data; boundary=--------a1b2c3d4
```

The browser generates that token and appends it automatically. Writing
`headers: {'Content-Type': 'multipart/form-data'}` yourself strips the boundary,
the server cannot find the field delimiters, and you get a confusing 422 with no
obvious cause. **Set no header and let the browser do it.**

---

## 2. CORS — what the browser is actually blocking

### The rule

An **origin** is the triple `(scheme, host, port)`. `http://localhost:3000` and
`http://localhost:8000` are *different origins* — a different port is enough.
So is `http://` versus `https://` on the same host.

Under the Same-Origin Policy, JavaScript on one origin may **send** a request to
another origin, but **cannot read the response** unless the server explicitly
permits it. The important nuance:

> The request usually still reaches your server and still executes. The browser
> blocks the *reading of the response* by the calling script.

That is why a CORS failure looks so strange in practice — your server logs a
successful 200, and the browser console reports a network error. Nothing is
broken server-side; the browser is refusing to hand the bytes to your JS.

### Why the policy exists

Without it, any page you visit could run `fetch('https://yourbank.com/api/balance')`
with your session cookies attached and read the result. The Same-Origin Policy
is what stops a random tab from silently reading your authenticated responses on
every other site.

### What the fix permits

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

This adds an `Access-Control-Allow-Origin` response header. It is the server
saying *"I consent to scripts from this origin reading my responses."* It grants
nothing else — no new capability, no authentication bypass. It only lifts the
browser's read restriction.

For non-simple requests the browser first sends an `OPTIONS` **preflight**
asking "may I send a POST with this content type from this origin?" The
middleware answers that too; `allow_methods` and `allow_headers` are what the
preflight response advertises.

### `allow_origins=["*"]` — the honest caveat

A wildcard means *any* website's JS can call this API and read the result. For
this project that is fine: the API is unauthenticated, has no user data, and
returns a classification of an image the caller already possessed. There is
nothing to steal.

It would be **wrong** for anything with cookies or auth. Note that browsers
refuse to combine `allow_origins=["*"]` with `allow_credentials=True` for
exactly this reason — the spec prevents the most dangerous combination. In a
real deployment, set `ALLOWED_ORIGINS` to the specific frontend origin.

### The part people miss

**Serving the frontend from the same origin as the API makes CORS irrelevant.**
`src/api.py` mounts `frontend/` at `/`, so in production the page and the API
share an origin and the browser never involves CORS at all.

CORS is configured here for *local development* — opening `index.html` from disk
(`file://`), or running a separate dev server on another port. The frontend
detects this:

```js
const API = location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';
```

Same-origin in production, explicit dev-server URL when opened off disk.

---

## 3. Deliberately minimal

Styling is a handful of CSS rules. `color-scheme: light dark` makes the page
follow the OS theme with no media queries. No framework, no bundler, no
`node_modules` — the page is one file served as a static asset, which is also
why the Docker image needs no JS toolchain.

One feature earns its place: the **threshold slider** re-scores the same image
live. It makes the precision/recall tradeoff from `docs/01-model.md` §8 tangible
— drag it to 0.95 and watch confident predictions flip to "real". That tradeoff
is the most important thing about the model and it is invisible in a static
result.

---

## 4. Error handling

```js
if (!res.ok) {
  const detail = await res.json().catch(() => ({}));
  throw new Error(detail.detail || `HTTP ${res.status}`);
}
```

FastAPI returns errors as `{"detail": "..."}`, so the UI surfaces the server's
actual message (`"could not decode image: ..."`) rather than a generic failure.
The `.catch(() => ({}))` handles the case where the error response is not JSON
at all — a proxy timeout, say — which would otherwise throw inside the error
handler and mask the original problem.

The fallback message names the URL it tried, because in local development the
overwhelmingly likely cause is that the API is not running.
