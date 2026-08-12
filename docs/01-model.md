# Stage 1 — The Model

Binary classifier: image in → `real` / `ai` + confidence.
Architecture: **frozen CLIP backbone + small trainable MLP head.**

---

## 1. What "frozen backbone + trainable head" actually means

A neural network is a chain of parameterised functions. Training normally runs
in three beats:

1. **Forward** — data flows through, producing a loss.
2. **Backward** — autograd walks the chain in reverse, computing `∂loss/∂param`
   for every parameter.
3. **Step** — the optimizer nudges each parameter against its gradient.

"Freezing" the backbone means setting `requires_grad = False` on all 88M CLIP
parameters. Mechanically this does **two independent things**:

- Autograd stops building the graph nodes needed to compute gradients for
  those tensors — no graph, no gradient buffers, much less memory.
- The optimizer never sees them, because we only hand it the head's
  parameters. Nothing steps them.

So CLIP's weights are **bit-identical before and after training**. It has
become a fixed, deterministic function: `image → 512-dim vector`. Only the
small MLP on top has gradients flowing into it.

### The consequence that shapes the whole pipeline

> If the backbone is frozen, its output for a given image never changes
> across epochs.

So there is no reason to recompute it every epoch. Run every image through
CLIP **once**, cache the 512-dim vectors to disk, then train the head on those
vectors. Training becomes a tiny tabular problem — 10,000 rows × 512 floats —
finishing in seconds on CPU instead of hours.

**This is why the pipeline is split into `extract_embeddings.py` and
`train_head.py`.** The split is a direct consequence of freezing, not an
arbitrary organisational choice.

### The bet underneath it

CLIP saw ~2B image–text pairs, so its embedding space already encodes texture,
lighting coherence, and object plausibility. We are betting that "real vs
AI-generated" is close to **linearly separable** in that space, and that we
only need to find the boundary.

If that bet is wrong, no amount of head capacity rescues it. Catching that
failure is the job of the eval step.

---

## 2. Design decisions (and the alternatives)

### `open_clip` over `transformers`

Both give you CLIP.

- `transformers` — more familiar, better documented, `CLIPModel.from_pretrained`.
- `open_clip` — the LAION reimplementation; access to LAION-2B checkpoints,
  which outperform OpenAI's original weights on essentially every benchmark.

**Decider:** `open_clip` returns its preprocessing pipeline as a plain,
readable `torchvision.Compose`. `transformers` hides resize/crop/normalise
inside a `CLIPProcessor` config. In Stage 2 the API must reproduce
training-time preprocessing *exactly* — a mismatch there is a silent accuracy
killer with no error message. That code should be visible.

### CPU-only PyTorch, despite having a GPU

Reasoning:

- The only compute-heavy step is embedding extraction, and it is one-time.
- Deployment is CPU-only, so a CPU dev environment means training and serving
  numerics match exactly.
- CUDA wheels are ~3 GB versus ~200 MB.

All code is device-agnostic (`--device auto|cpu|cuda`), so swapping later
changes nothing else. **This is a judgment call, not a rule** — installing the
CUDA build here is equally defensible.

#### "Why CPU instead of GPU?" — the longer answer

This is really two questions, with opposite answers.

**The head training genuinely should not use a GPU.** It is a 131k-parameter
MLP over 10,000 cached vectors; one batch is a `64×512 @ 512×256` matmul, about
17 MFLOPs. Work that small is dominated by overhead, not math — each kernel
launch costs ~5–10 µs plus a host→device transfer per batch, so the scheduling
outlasts the arithmetic. The GPU version would plausibly be *slower*. General
rule: **GPUs win on throughput, not latency**, and need enough work per launch
to amortise the overhead. Head training finishes in ~9 seconds on CPU; there is
nothing to accelerate.

**Extraction is where a GPU would genuinely help.** 14,000 images × 4.4 GFLOPs
≈ 60 TFLOPs. The 1660 Ti does ~5.4 TFLOPS fp32 against maybe 0.1 effective on
CPU — 25 img/s versus a likely 300–500. Ten minutes versus under one. So the
CPU choice cost ~9 minutes, once, to avoid a 3 GB download and a driver
variable in a project with five more stages of new things to debug.

**A correction to the original framing:** "CPU-only means training and serving
numerics match exactly" is true but weak. CPU/CUDA fp32 divergence is ~1e-6
from different reduction orderings, and would only flip a prediction sitting
almost exactly on the decision boundary.

The argument that actually carries weight: **the measured 25 img/s is
production data you need.** Stage 5 picks between Render/Fly.io/Modal and
Stage 2 decides sync vs async `/predict`; both hinge on how long one CPU
inference really takes, on a target box weaker than this laptop. Extracting on
GPU teaches you nothing about the environment you ship to.

If switching anyway: note the 1660 Ti is a **TU116** chip with *no tensor
cores* (RTX-only that generation), so fp16/autocast will not give the usual
extra 2–3× — it is plain fp32 CUDA cores.

```powershell
uv pip install --index-strategy unsafe-best-match `
    --extra-index-url https://download.pytorch.org/whl/cu126 `
    --force-reinstall torch torchvision
.\.venv\Scripts\python.exe -m src.extract_embeddings --device cuda --batch-size 256
```

### Label convention flip

CIFAKE ships `0 = FAKE, 1 = REAL`. We re-map to `0 = real, 1 = ai`.

In binary classification, class 1 is the **positive** class, and
precision/recall are reported *with respect to the positive class*. The
product question is "is this AI-generated?", so AI-generated must be positive.
Otherwise "recall" silently means *"of all the real photos, how many did we
catch"* — not the number anyone cares about.

Pick a convention early, write it in `config.py`, never think about it again.

### Subset size — the real tradeoff

CIFAKE has 50k images per class in train. Extraction cost is linear in image
count; head training cost is negligible either way.

The curve is steeply diminishing: a linear probe on CLIP features typically
saturates within a few thousand examples per class, because you are fitting
~512 parameters, not learning representations.

**Chosen: 5,000/class train, 2,000/class test** (14k images, ~10 min CPU
extraction). 2,000/class test gives roughly a ±0.6% 95% confidence interval on
accuracy — tight enough that the number means something. The full 50k/class
would multiply extraction time by 10 for maybe a point of accuracy.

### PyTorch for training, scikit-learn only for metrics

`sklearn.linear_model.LogisticRegression` would genuinely work for a linear
probe on cached embeddings — one line, solved to optimality with L-BFGS rather
than stochastic descent, probably within a point of the PyTorch version. It is
the *pragmatic* choice for this specific problem.

Rejected for two reasons:

1. **Learning.** Writing the training loop by hand is where the freeze/gradient
   mechanics stop being an explanation and become something you watched
   execute. A `LogisticRegression` call teaches nothing about `requires_grad`.
2. **Serving.** The head gets loaded in Stage 2's FastAPI app. As a PyTorch
   module: `model.load_state_dict(...)`, one framework end to end. As an
   sklearn estimator: pickling a scikit-learn object into the production
   container, adding sklearn to the serving image, stitching two frameworks
   together at the inference boundary. That is a real production smell.

**Honest cost:** an MLP head trained with Adam has hyperparameters (learning
rate, epochs, dropout, hidden width) that `LogisticRegression` simply does not
have. More surface area to explain, more places to go subtly wrong.

---

## 3. An honest problem with the dataset

**CIFAKE images are 32×32 pixels. CLIP expects 224×224.**

Every image gets bicubically upscaled 7×, which means the high-frequency
generator fingerprints real AI-detectors rely on — JPEG-ish artifacts,
upsampling residue, frequency-domain traces — are **physically destroyed**
before CLIP ever sees them.

Worse: every "fake" in CIFAKE came from *one* model (Stable Diffusion 1.4) and
every "real" from CIFAR-10.

**Consequence:** test accuracy will be very high, and it will substantially
overstate real-world performance on a 1024px Midjourney image. That is the
**generalization gap**, and it is baked into the dataset choice — not fixable
by better training.

Proceeding with CIFAKE anyway: it is the right choice for learning the
pipeline, it is small and fast, and the point of the eval step is to make the
gap *articulable* rather than a blindside. It goes in the final README
honestly.

---

## 4. The dataset script — `src/download_data.py`

Downloads `dragonintelligence/CIFAKE-image-dataset` (parquet, 100k train / 20k
test, ~50 MB), samples a balanced subset, writes PNGs to
`data/cifake/{train,test}/{real,ai}/`.

### Why loose PNGs instead of the HuggingFace dataset object?

1. You can open the folder and **look at the data** — which matters more than
   people admit when a model misbehaves.
2. The extraction script stays framework-agnostic; it just globs a folder.
3. Later stages need real image files on disk anyway (manual API testing in
   Stage 2, the CI smoke test in Stage 6).

### Things worth understanding

**Sampling is per-class, not per-split.** The naive version is
`dataset.shuffle().select(range(10000))`. It works here because CIFAKE is
exactly 50/50 — and it silently produces a skewed training set the first time
you point it at a dataset that isn't. Class imbalance biases the decision
boundary toward the majority class, and it surfaces as a mysteriously bad
recall number three scripts later. Enforcing balance at sample time costs four
lines.

**`np.asarray(dataset["label"])` decodes zero images.** Parquet is columnar, so
pulling one column touches only that column's bytes. This is how the script
picks 5,000 rows out of 100,000 without paying to decode the other 95,000.

> Pattern: *read the cheap column → compute indices → `.select()` lazily.*
> Doing it the other way round is a classic accidental full scan.

**Files are named by their original dataset row index** (`050267.png`), so a
suspicious image can always be traced upstream.

---

## 5. Embedding extraction — `src/extract_embeddings.py`

Loads CLIP frozen, pushes all 14,000 images through once, writes
`data/embeddings/{split}.npz` containing `features` (N, 512) float32, `labels`
(N,) int64, `paths` (N,) str, plus provenance stamps.

From here on, **`train.npz` *is* the training set** — a 10,000 × 512 array,
about 20 MB.

### The cost of caching, stated honestly

Caching embeddings makes **data augmentation impossible**. Normally you would
randomly crop/flip each image differently every epoch so the model sees fresh
variations. A cached embedding is computed from one fixed version, so every
epoch sees the identical vector.

You trade augmentation for a 50–100× speedup. For a linear-ish probe over 10k
examples that is a good trade — augmentation mostly helps when training the
feature extractor, which we are not. But it *is* a trade, and if the head
overfits badly, this is a lever we gave up.

### Preprocessing: use CLIP's own transform, don't hand-roll one

`open_clip.create_model_and_transforms()` returns the model *and* the exact
preprocessing pipeline that checkpoint was trained with — bicubic resize to
224, center crop, tensor conversion, normalisation by CLIP's specific channel
means and standard deviations.

Those normalisation constants are **not** the ImageNet ones people reflexively
use, and a mismatch produces no error — just quietly degraded embeddings.
Taking the transform from the model object means it cannot drift.

> This rule matters enormously in Stage 2: the API must use this identical
> transform, sourced the same way, or production accuracy silently diverges
> from eval accuracy.

For our 32×32 inputs this means: bicubic upscale to 224×224, after which the
center crop is a no-op. We are feeding CLIP a blurry 7× enlargement — the
generalization gap, made mechanical.

### L2-normalising the embeddings — a judgment call

CLIP's raw output vectors vary in magnitude image to image. Two options: feed
raw vectors, or project every vector onto the unit sphere (divide by its own
length).

OpenAI's original linear-probe results used raw features; CLIP's own retrieval
usage always normalises. **We normalise**, because it bounds every input
feature to [-1, 1], making the head's optimal learning rate far less sensitive
to data scale — one less hyperparameter to get wrong.

**The catch:** this must be replicated at serving time. It is now part of the
model's *contract*, not a preprocessing detail — hence it is recorded in the
`.npz` metadata rather than left implicit.

### Things worth understanding

**`151.3M parameters, 0 trainable` is step 1, confirmed.** That printed line is
not decoration. If you refactor and it isn't zero, you have accidentally
un-frozen the backbone; the symptom is training that is 100× slower and
mysteriously worse. Printing an invariant you believe about your model is
cheap. Discovering it was violated three scripts downstream is not.

**`@torch.inference_mode()` vs `torch.no_grad()`** — a common interview
question. Both stop autograd from recording operations, which is what saves the
memory: without it, every intermediate activation in all 12 transformer blocks
is retained for a backward pass that never comes, and you OOM or thrash on
large batches.

`inference_mode` goes further — it also disables version-counter tracking on
tensors, so it is strictly faster, but its output tensors can **never**
afterwards participate in autograd.

> Rule: `inference_mode` for pure inference; `no_grad` when the result might be
> fed into something differentiable later.

**The `clip_model` / `clip_pretrained` / `l2_normalised` fields in the `.npz`
are a contract, not metadata.** A trained head is meaningless without knowing
which backbone and which normalisation produced its inputs. Swap the checkpoint
and retrain, but leave the API loading the old head — no error, just degraded
predictions. Stamping the producing configuration into the artifact lets the
consumer **assert** rather than assume.

> This is the single most common way ML systems break in production: the
> inference-time feature pipeline drifts from the training-time one, silently.

**Batching matters on CPU.** At `--batch-size 64` you get ~25 img/s; a
per-image loop is several times slower, because matrix multiply only reaches
good throughput when the matrices are big enough to keep all cores fed.

### Measured throughput

| Device | Rate | 14,000 images |
|---|---|---|
| CPU (this machine) | ~25 img/s | ~10 min |
| GTX 1660 Ti | (not measured) | well under 1 min |

---

## 6. Head training — `src/train_head.py` and `src/model.py`

CLIP is never loaded here. The frozen backbone already did its job, so this
script sees only a (10000, 512) float array — a small tabular problem that
trains in about **9 seconds on CPU**.

### Reading a loss curve

Binary cross-entropy has a meaningful zero point worth memorising:

> A model that outputs 50/50 for everything scores a loss of **ln 2 ≈ 0.693**.

That is the "I know nothing" baseline. Every loss number should be read
relative to it.

**Healthy:** train loss starts near 0.69, drops steeply over the first few
epochs, flattens into a slow decline. Val loss tracks it down, bottoms out,
then plateaus or gently rises. A modest final gap is normal.

The two failure modes look nothing alike and **require opposite fixes**:

| | Symptom | Cause | Fix | Makes it *worse* |
|---|---|---|---|---|
| **Not learning** | train loss flat at ~0.693, or oscillating without trend | lr too low/high; labels misaligned with features (indexing bug); features broken | fix lr, find the bug | adding regularisation |
| **Overfitting** | train → 0 while val bottoms out and climbs; gap widens | model memorising rows | earlier stop, more dropout/weight decay, less capacity, more data | raising the lr |

> Diagnose by the **gap**, not by either number alone. High train + high val =
> underfitting, reach for capacity. Low train + high val = overfitting, reach
> for regularisation. Confusing the two is extremely common.

**Predicted for this setup:** train and val both falling to ~0.1–0.2, val
accuracy ~95–97%, small gap. Mild overfitting at most — a 131k-parameter head
on 8,500 rows of pre-computed features that overfits wildly indicates a bug,
not merely a suboptimal configuration.

### Design decisions

**A validation split carved from train; the test set stays sealed.** 15% held
back for validation. Test is untouched until the eval step.

> The moment you look at test performance and change a hyperparameter in
> response, the test set has become part of your training procedure and its
> accuracy is an optimistic estimate, not an honest one.

Every "we got 99%!" that collapses in production has this somewhere in its
history. **Validation is for making decisions; test is for reporting a number,
once.**

The split is *stratified* — sampled per class so both sides keep the 50/50
ratio. Validation is the measuring instrument; letting its composition wobble
run to run means a comparison between two configurations partly measures the
split rather than the model.

**One output logit with `BCEWithLogitsLoss`, not two logits with
`CrossEntropyLoss`.** Two-logit softmax is over-parameterised for binary — only
the *difference* between logits affects the output, so you train a redundant
degree of freedom. One logit through a sigmoid directly gives `P(ai)`.

The `WithLogits` part is the interview question. The naive version is
`nn.Sigmoid()` then `nn.BCELoss()` — mathematically identical, numerically
dangerous: for a confident prediction sigmoid saturates to exactly 1.0 or 0.0
in float32, `log(0)` is `-inf`, and the resulting NaN gradients poison every
weight. `BCEWithLogitsLoss` fuses the two and uses the log-sum-exp trick to
stay stable at any logit magnitude.

> Consequence: **the model outputs raw logits, not probabilities.** The sigmoid
> moves to inference time — a detail Stage 2 must get right.

**MLP vs linear probe — made comparable rather than decided blind.** Default is
`512 → 256 → ReLU → Dropout → 1`. But for CLIP features a bare linear layer is
often equally good, since the whole premise is that the classes are nearly
linearly separable already. `--hidden 0` trains the pure linear baseline in
seconds. Run both; that is the honest way to learn whether the hidden layer
earns its place.

**Keep the best-validation checkpoint, not the last epoch.** All epochs run,
but we save the weights from the lowest-val-loss epoch. If val loss rises at
the end, the final weights are strictly worse than what we already had. The
alternative — early stopping with a patience counter — is the same idea but
adds a hyperparameter and hides the rest of the curve. On a 9-second run,
seeing the whole curve is more informative.

**`AdamW`, not `Adam`.** AdamW applies weight decay directly to the weights
rather than folding it into the gradient. With adaptive per-parameter step
sizes the two are *not* equivalent, and AdamW's version is the one that
actually regularises as intended.

### Why the head lives in `src/model.py`, not in the training script

Stage 2's API must import **this exact class** to reconstruct the model from a
checkpoint.

> A PyTorch `state_dict` is only weights. It carries no architecture.

If serving rebuilt the head from a hardcoded guess at layer sizes and training
later changed them, `load_state_dict` would either throw or — worse — silently
succeed with wrong shapes. So the checkpoint stores its own architecture
config and `load_head` rebuilds from that rather than from an assumption.

### Things worth understanding

**`optimizer.zero_grad()` is not boilerplate.** PyTorch *accumulates* gradients
into `.grad` rather than overwriting them. Omit the zeroing and every step uses
the running sum of all gradients so far, which diverges almost immediately.
The default looks like a design mistake until you need it — accumulating over
several small batches is how you simulate a large batch that would not fit in
memory. It is opt-out rather than opt-in, deliberately.

**`squeeze(-1)` in `forward()` prevents a genuinely nasty silent bug.** The
final `Linear` emits `(B, 1)`; labels are `(B,)`. Feeding mismatched shapes to
`BCEWithLogitsLoss` **does not error** — they broadcast to a `(B, B)` loss
matrix comparing every prediction against every label. Training proceeds,
loss decreases, and the model learns nonsense. Shape bugs that broadcast
instead of throwing are among the hardest to find.

**`model.train()` / `model.eval()` toggle dropout, and this explains something
that looks wrong.** In early epochs you will often see **validation loss
*below* training loss**, which looks impossible. Two reasons, both benign:

1. Training loss is measured *with dropout active* — a randomly thinned,
   handicapped network — while validation runs the full model.
2. Training loss is averaged across the epoch while the model is still
   improving, so it includes the bad early batches; validation is measured
   once at the end, on the improved weights.

Forgetting `eval()` inflates validation loss, because you would be measuring a
randomly-thinned network rather than the model you intend to ship.

**`logits > 0` is exactly `sigmoid(logits) > 0.5`**, and skips computing the
sigmoid. Sigmoid is monotonic and crosses 0.5 at exactly x=0.

**`weights_only=True` in `torch.load`.** Unpickling a checkpoint is arbitrary
code execution. Our file holds only tensors, dicts, strings and floats, so the
restriction costs nothing — and it matters once a checkpoint gets fetched at
deploy time rather than produced locally.

### Artifacts produced

- `artifacts/head.pt` — `state_dict` + architecture config + serving contract
  (`clip_model`, `clip_pretrained`, `l2_normalised`, `class_names`,
  `positive_class`, hyperparameters, best epoch)
- `artifacts/head.history.json` — per-epoch loss/accuracy, for plotting

---

## 7. Results — MLP vs linear probe

### The first comparison was invalid, and how to tell

Both models were given identical hyperparameters (`lr=1e-3`, 30 epochs) and the
linear probe scored 89.67% against the MLP's 94.07%. That comparison is
worthless. Three signals in the linear run say it never finished training:

1. **Every epoch marked `<- best`, including the last.** Loss still
   monotonically decreasing at epoch 30.
2. **Train/val gap +0.0068 — essentially zero — while absolute loss was 0.38.**
   Near-zero gap *plus* high loss is the signature of underfitting. Overfitting
   has a large gap; convergence has a plateau. This had neither.
3. Val accuracy climbing steadily with no ceiling in sight.

> **Identical hyperparameters are not the same thing as a fair comparison.**
> Each model must be trained to its own convergence before you compare them.

### Why the *simpler* model trained slower (counterintuitive)

513 parameters should fit more easily than 131,585. The cause is the L2
normalisation: every input sits on the unit sphere, so across 512 dimensions
each component averages ~`1/√512 ≈ 0.044`. The logit is `w·x`, so a confident
output (logit ≈ 5) needs a large weight norm. Adam moves each weight by roughly
`lr` per step, so `‖w‖` grows on a **slow linear ramp** — 30 epochs × 34 batches
≈ 1,000 steps is not enough runway.

The MLP escapes this because its output is a *product* of two weight matrices;
both grow at once, so achievable logit scale grows **quadratically** in step
count. The hidden layer buys optimisation speed as well as nonlinearity — two
distinct benefits the first experiment conflated.

Fix: `--hidden 0 --lr 1e-2 --epochs 300`.

### Converged comparison

| | linear probe | MLP (hidden=256) |
|---|---|---|
| parameters | 513 | 131,585 |
| best val loss | 0.1596 (ep 268) | **0.1459 (ep 88)** |
| **test accuracy** | 94.75% | **95.40%** |
| ROC-AUC | 0.9880 | **0.9908** |
| avg precision | 0.9890 | **0.9916** |
| calibration error | **0.0069** | 0.0141 |
| false positives | 102 | **75** |
| false negatives | **108** | 109 |

**Shipped: the MLP** (`artifacts/head.pt`). It wins on accuracy, AUC, and
false-positive count. The +0.65pp accuracy edge is about **2 standard errors**
on 4,000 test samples (SE ≈ 0.34pp) — real but modest, so this is not a
dramatic win. The linear probe is genuinely *better calibrated* and 256× smaller,
and would be a defensible ship for a memory-constrained target.

**The MLP overfits hard after epoch 88** — train loss reaches 0.0009 by epoch
300 while val loss climbs to 0.27. Best-checkpoint selection is what makes the
300-epoch default safe; without it, the shipped model would be far worse. The
training script now reports the gap **at the best epoch** rather than the last,
and prints an explicit note when it detects a late overfit.

---

## 8. Evaluation — `src/eval.py`

### Why accuracy alone is the wrong number to quote

1. **The test set is 50/50 by construction; real traffic is not.** If 5% of
   uploads are genuinely AI-generated, this error profile produces far more
   false accusations than true catches — precision collapses while accuracy is
   unchanged. Accuracy measured on a balanced set does not transfer to an
   unbalanced deployment.
2. **Accuracy fixes the threshold at 0.5 and hides the tradeoff.** The two
   error types have very different costs: calling a real photo "AI-generated"
   is an accusation; missing an AI image is a shrug. One number cannot express
   a choice you have not made yet.
3. **The generalization gap.** 32×32, one generator (SD 1.4), one real source
   (CIFAR-10). This is an **upper bound** on real-world performance, not an
   estimate of it.

### Final numbers (`head.pt`, threshold 0.5, 4,000 sealed test images)

```
                 pred real     pred ai
     true real       1,925          75     <- 75 false accusations
       true ai         109       1,891     <- 109 slipped through

           precision    recall        f1
    real      0.9464    0.9625    0.9544
      ai      0.9619    0.9455    0.9536

  accuracy    0.9540
  ROC-AUC     0.9908      avg precision  0.9916
```

### What the extra metrics are for

**ROC-AUC (0.9908) is threshold-free.** It measures how well the model *ranks*
AI above real, independent of where the cutoff sits. The distinction matters
diagnostically: a poor AUC cannot be fixed by tuning the threshold, but a good
AUC with poor accuracy can. 0.9908 says the ranking is strong and any remaining
accuracy is a threshold choice.

**The threshold sweep is a product decision, not a model property.**

| threshold | precision | recall | use case |
|---|---|---|---|
| 0.10 | 0.9071 | 0.9815 | moderation queue — catch almost everything, humans filter |
| 0.50 | 0.9619 | 0.9455 | balanced default |
| 0.99 | 0.9986 | 0.7295 | public "AI-generated" badge — 1 false accusation in 700, but misses 27% |

Moving from 0.5 to 0.99 trades 22 points of recall for 3.7 points of precision.
Whether that is a good trade depends entirely on what happens to a user who is
wrongly flagged.

**Calibration: does a stated confidence of 0.9 mean right 90% of the time?**
This matters directly because the API returns a confidence to the end user. An
overconfident model claiming 99% while being right 80% of the time is actively
misleading at identical accuracy. Expected Calibration Error is the average gap
between claimed and actual, weighted by bin size.

Measured **ECE = 0.0141** — well calibrated. The gaps are consistently slightly
negative (mild overconfidence), worst in the 0.8–0.9 bin (claims 0.857, is
right 0.786). The confidence number is honest enough to show a user.

**Most confident mistakes** are all `ai called real` at p(ai) ≈ 0.0000 — the
model is not merely wrong on these, it is certain. Worth eyeballing the actual
files; they are listed by path so you can.

### The contract check

`eval.py` refuses to run if the head's stamped `clip_model` / `clip_pretrained`
disagree with the embeddings file. Shapes would still match and predictions
would still be produced — just wrong. This is the assert-don't-assume rule from
§5 being cashed in.
