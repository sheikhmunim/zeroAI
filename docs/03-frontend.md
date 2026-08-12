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

## 3. Colour: why not green/red

The obvious encoding for "real vs AI" is green/red. It is the wrong one — that
is the classic protanopia/deuteranopia confusion pair, so roughly **8% of men**
would see two near-identical bars.

The replacement is a **diverging blue ↔ red pair** with a neutral grey midpoint,
which is exactly the right structure here: two opposed poles with "no signal" in
between. Measured with a CVD validator:

| | light | dark |
|---|---|---|
| CVD separation (protan, OKLab ΔE ×100) | 21.6 | 19.2 |
| normal-vision separation | 32.3 | 29.0 |
| contrast vs surface | ≥ 3:1 | ≥ 3:1 |

Target is ΔE ≥ 8, so these clear it by a wide margin. **Don't eyeball this —
compute it.** Intuition about colour distance under simulated CVD is unreliable.

Colour is never the only channel regardless: the verdict is spelled out in
words, and the exact probability is printed as text.

**Text never wears the data colour.** The verdict sits in primary ink with a
coloured dot beside it. Colouring a 2rem heading red looks fine on a white
background and fails on a dark one; a dot beside ink-coloured text carries the
same identity at any contrast.

---

## 4. The meter is anchored at the threshold, not at zero

A plain 0–100% confidence bar hides the single most important thing about this
model: that the verdict depends on a cutoff someone chose.

So the bar grows **outward from the threshold marker** — left and blue when the
model says real, right and red when it says AI — and its length is the distance
past the boundary. Move the slider and the anchor visibly moves with it.

Mark specs: 4px rounded data-end, square at the anchor; 1px hairline for the
threshold tick; the unfilled track is the neutral diverging midpoint.

**The numbers are all printed as text below the meter** (`p(ai)`, confidence,
latency, filename). That grid is the accessibility table view — nothing is gated
behind reading a colour or estimating a bar length.

---

## 5. Honesty affordances

Three features exist because the model's weaknesses are real and a demo that
hides them is a misleading demo.

**"What the model sees."** A second thumbnail renders the image after CLIP's
actual preprocessing — shortest side resized to 224, then centre-cropped. For a
32×32 CIFAKE sample you see the 7× upscale that destroys high-frequency
artifacts. For a 16:9 photo you see the left and right edges *thrown away*. That
is the honest answer to "why did it get my photo wrong" when the subject sat
outside the crop.

**Out-of-distribution warnings.** The model was trained exclusively on 32×32
images. A 4000px photograph is far outside that, and the confidence score has no
idea — a calibrated 99% means "99% *on data that looks like training data*". The
page checks client-side and says so:

- ≥ 4× larger than 32×32 → "treat the confidence as unreliable"
- aspect ratio ≥ 1.3:1 → "the centre crop discards the edges"
- JPEG → "compression alters exactly the high-frequency detail this keys on"

**Never print "100.0%".** A sigmoid cannot reach 1.0, so the model is never
certain — but `p_ai = 0.9996` rounds to a flat `100.0%` at one decimal place,
claiming exactly the certainty the rest of the page disclaims. Two decimals
(`99.96%`), with a `>99.99` form for float32 saturation.

### A real bug this surfaced

`confidence` from the API is `P(predicted class)`. That is only ≥ 0.5 when the
threshold is 0.5. Drag the slider to 0.45 on an image with `p_ai = 0.499` and
the label becomes `ai` while `confidence` comes back as **0.499** — so the UI
rendered *"49.90% confident"* directly beneath a verdict of *"AI-generated"*,
which states the opposite of itself.

The API value is correct; the English was wrong. When the model's own preference
disagrees with the policy, the page now says so explicitly:

> p(ai) is 0.4990, so the model marginally favours **real** — but your threshold
> of 0.45 classifies it as **AI-generated**.

Which is also the clearest possible demonstration that **the threshold is a
policy, not a model property**.

---

## 6. Input paths, and the borderline sample

Three ways in: click/browse, drag-and-drop, and **paste anywhere on the page**.
Paste matters more than it sounds — screenshots land in the clipboard as image
data with no filename, and that is the single most common way somebody tries an
image detector. Forcing a save-to-disk round trip first is pure friction.

Three built-in samples, served as static files from `frontend/samples/`:

| chip | p(ai) | why |
|---|---|---|
| real photo | 0.0002 | confident correct negative |
| AI-generated | 0.9996 | confident correct positive |
| **borderline** | **0.4990** | sits on the decision boundary |

The borderline one was found by scoring the whole test set and taking the image
closest to 0.5 (it is genuinely a real photo). It exists because the first
version of the threshold slider promised "drag it and watch a verdict flip" and
then **could not deliver** — both other samples are so far from the boundary
that no reachable threshold changes their label. Either fix the copy or supply
an image where the claim is true; supplying the image teaches more.

---

## 7. Deliberately minimal

No framework, no bundler, no `node_modules` — one file served as a static asset,
which is also why the Docker image needs no JS toolchain. Theming is CSS custom
properties swapped under `prefers-color-scheme`, with a `data-theme` override
that wins both ways.

---

## 8. Error handling

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
