"""Evaluate a trained head on the sealed test set.

This is the first and only time the test split is touched. Every decision --
architecture, learning rate, epoch count, threshold -- was made against the
validation split. That discipline is what makes the number below an estimate
of future performance rather than a description of past fitting.

Why accuracy alone is the wrong number to quote:

  1. The test set is 50/50 by construction. Real traffic is not. If 5% of
     uploads are actually AI-generated, a detector with 94% accuracy and this
     error profile produces far more false accusations than true catches --
     precision collapses even though accuracy is unchanged. Accuracy measured
     on a balanced set does not transfer to an unbalanced deployment.
  2. Accuracy fixes the threshold at 0.5 and hides the tradeoff. The two error
     types have very different costs: calling a real photo "AI-generated" is
     an accusation; missing an AI image is a shrug. One number cannot express
     a choice you have not yet made.
  3. The generalization gap. CIFAKE is 32x32, one generator (SD 1.4), one real
     source (CIFAR-10). Test accuracy here is an upper bound on real-world
     performance, not an estimate of it.

Run:
    python -m src.eval
    python -m src.eval --head linear.pt
    python -m src.eval --head linear.pt --threshold 0.7
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config import ARTIFACT_DIR, CLASS_NAMES, EMBED_DIR
from src.model import load_head


def load_test_set() -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    path = EMBED_DIR / "test.npz"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run `python -m src.extract_embeddings`")
    blob = np.load(path, allow_pickle=False)
    return (
        torch.from_numpy(blob["features"]),
        blob["labels"],
        blob["paths"],
    )


def check_contract(meta: dict, blob_path) -> None:
    """Refuse to evaluate a head against features it was not trained for."""
    stamped = np.load(blob_path, allow_pickle=False)
    mismatches = [
        f"{key}: head={meta[key]!r} features={str(stamped[key])!r}"
        for key in ("clip_model", "clip_pretrained")
        if meta[key] != str(stamped[key])
    ]
    if mismatches:
        raise SystemExit("backbone contract mismatch:\n  " + "\n  ".join(mismatches))


def render_confusion(cm: np.ndarray) -> str:
    """Confusion matrix with rows = truth, columns = prediction."""
    width = max(len(name) for name in CLASS_NAMES) + 2
    header = " " * (width + 8) + "".join(f"{'pred ' + n:>12}" for n in CLASS_NAMES)
    lines = [header]
    for i, name in enumerate(CLASS_NAMES):
        row = "".join(f"{cm[i][j]:>12,}" for j in range(len(CLASS_NAMES)))
        lines.append(f"{'true ' + name:>{width + 8}}{row}")
    return "\n".join(lines)


def threshold_sweep(labels: np.ndarray, probs: np.ndarray) -> list[dict]:
    """How the precision/recall tradeoff moves as the cutoff changes.

    The threshold is a product decision, not a model property. A content
    moderation queue wants high recall; a public "this is AI" badge wants high
    precision, because a false accusation is expensive.
    """
    rows = []
    for threshold in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        preds = (probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=1, zero_division=0
        )
        rows.append(
            {
                "threshold": threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float((preds == labels).mean()),
                "flagged": int(preds.sum()),
            }
        )
    return rows


def calibration(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> tuple[list[dict], float]:
    """Does a confidence of 0.9 actually mean right 90% of the time?

    The API returns a confidence to the user, so this matters directly: an
    overconfident model that says 99% while being right 80% of the time is
    actively misleading, even at identical accuracy. Expected Calibration
    Error is the average gap between claimed and actual, weighted by bin size.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, ece = [], 0.0

    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        # Confidence = distance from the 0.5 decision boundary, expressed as
        # the probability assigned to whichever class was predicted.
        confidence = np.where(probs >= 0.5, probs, 1.0 - probs)
        correct = (probs >= 0.5).astype(int) == labels

        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue

        claimed = float(confidence[in_bin].mean())
        actual = float(correct[in_bin].mean())
        weight = int(in_bin.sum())
        ece += weight / len(labels) * abs(claimed - actual)
        rows.append(
            {"bin": f"{lo:.1f}-{hi:.1f}", "n": weight, "claimed": claimed, "actual": actual}
        )

    return rows, ece


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="head.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show-mistakes", type=int, default=5)
    args = parser.parse_args()

    head_path = ARTIFACT_DIR / args.head
    if not head_path.exists():
        raise SystemExit(f"missing {head_path} -- run `python -m src.train_head`")

    model, meta = load_head(head_path)
    features, labels, paths = load_test_set()
    check_contract(meta, EMBED_DIR / "test.npz")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"head       : {args.head}  ({n_params:,} parameters)")
    print(f"backbone   : {meta['clip_model']} / {meta['clip_pretrained']}")
    print(f"trained    : {meta['train_size']:,} images, best epoch {meta['best_epoch']}")
    print(
        f"test set   : {len(labels):,} images, "
        + ", ".join(f"{n} {c}" for c, n in zip(CLASS_NAMES, np.bincount(labels), strict=False))
    )
    print(f"threshold  : {args.threshold}")

    with torch.inference_mode():
        probs = torch.sigmoid(model(features)).numpy()
    preds = (probs >= args.threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=range(len(CLASS_NAMES)))
    print("\n--- confusion matrix ---")
    print(render_confusion(cm))

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  false positives : {fp:,}  real photos wrongly flagged as AI")
    print(f"  false negatives : {fn:,}  AI images that slipped through")

    print("\n--- per class ---")
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    print(f"{'':>8}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>8}{precision[i]:>12.4f}{recall[i]:>10.4f}{f1[i]:>10.4f}{support[i]:>10,}")

    accuracy = float((preds == labels).mean())
    # Threshold-free: how well the model *ranks* AI above real, independent of
    # where the cutoff sits. A poor ROC-AUC cannot be fixed by tuning the
    # threshold; a good one with poor accuracy can.
    roc_auc = float(roc_auc_score(labels, probs))
    ap = float(average_precision_score(labels, probs))

    print(f"\n  accuracy    {accuracy:.4f}")
    print(f"  ROC-AUC     {roc_auc:.4f}   (threshold-free ranking quality)")
    print(f"  avg prec    {ap:.4f}")

    print("\n--- threshold sweep (positive class = ai) ---")
    sweep = threshold_sweep(labels, probs)
    print(f"{'thresh':>8}{'precision':>12}{'recall':>10}{'f1':>10}{'accuracy':>11}{'flagged':>10}")
    for row in sweep:
        print(
            f"{row['threshold']:>8.2f}{row['precision']:>12.4f}{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}{row['accuracy']:>11.4f}{row['flagged']:>10,}"
        )

    print("\n--- calibration (does stated confidence mean anything?) ---")
    bins, ece = calibration(labels, probs)
    print(f"{'confidence':>12}{'n':>8}{'claimed':>10}{'actual':>10}{'gap':>9}")
    for row in bins:
        gap = row["actual"] - row["claimed"]
        print(
            f"{row['bin']:>12}{row['n']:>8,}{row['claimed']:>10.3f}{row['actual']:>10.3f}{gap:>+9.3f}"
        )
    verdict = "well calibrated" if ece < 0.05 else "overconfident" if ece >= 0.05 else ""
    print(f"\n  expected calibration error {ece:.4f}  ({verdict})")

    if args.show_mistakes:
        print(f"\n--- {args.show_mistakes} most confident mistakes ---")
        wrong = np.flatnonzero(preds != labels)
        # Rank by how wrong: distance of the probability from the true label.
        worst = wrong[np.argsort(-np.abs(probs[wrong] - labels[wrong]))][: args.show_mistakes]
        for i in worst:
            print(
                f"  {CLASS_NAMES[labels[i]]:>5} called {CLASS_NAMES[preds[i]]:<5} "
                f"p(ai)={probs[i]:.4f}  {paths[i]}"
            )

    report = {
        "head": args.head,
        "n_parameters": n_params,
        "threshold": args.threshold,
        "test_size": int(len(labels)),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "threshold_sweep": sweep,
        "expected_calibration_error": ece,
    }
    out_path = ARTIFACT_DIR / f"eval-{head_path.stem}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
