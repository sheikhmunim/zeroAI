"""Train the classification head on cached CLIP embeddings.

CLIP is never loaded here. The frozen backbone already did its job in
extract_embeddings.py, so this script sees only a (N, 512) float array -- a
small tabular problem that trains in seconds on CPU.

Reading the output:
  * BCE loss for a model that guesses 50/50 on everything is ln(2) = 0.693.
    Every number below should be read relative to that.
  * Healthy    -> train and val both fall, gap stays small.
  * Not learning -> train loss stuck near 0.693. Fix the lr or find the bug;
    regularisation makes this strictly worse.
  * Overfitting  -> train keeps falling, val bottoms out then rises. Fix with
    dropout / weight decay / earlier stop; a higher lr makes this worse.

Run:
    python -m src.train_head
    python -m src.train_head --hidden 0        # pure linear probe baseline
    python -m src.train_head --epochs 60 --lr 3e-4
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    ARTIFACT_DIR,
    CLASS_NAMES,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    EMBED_DIR,
)
from src.model import DetectorHead, save_head


def load_embeddings(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached features and assert they came from the expected backbone."""
    path = EMBED_DIR / f"{split}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run `python -m src.extract_embeddings`")

    blob = np.load(path, allow_pickle=False)

    # The provenance check. A head trained on ViT-B-32 features is meaningless
    # applied to any other backbone, and the failure is silent -- shapes still
    # match, predictions are just wrong. Assert instead of assume.
    if str(blob["clip_model"]) != CLIP_MODEL or str(blob["clip_pretrained"]) != CLIP_PRETRAINED:
        raise SystemExit(
            f"{path} was built with {blob['clip_model']}/{blob['clip_pretrained']}, "
            f"but config.py expects {CLIP_MODEL}/{CLIP_PRETRAINED}. Re-extract."
        )

    features = torch.from_numpy(blob["features"])
    # float, not long: BCEWithLogitsLoss treats targets as probabilities, so
    # they must be floats even though they only ever take the values 0.0/1.0.
    labels = torch.from_numpy(blob["labels"]).float()
    return features, labels


def stratified_split(
    labels: torch.Tensor, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train/val, preserving the class ratio in both.

    A plain random split would leave the validation class balance to chance.
    On 1,500 held-out rows the drift is small, but validation *is* the
    measuring instrument -- letting its composition wobble run to run means
    comparing two configurations partly measures the split, not the model.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []

    for class_id in range(len(CLASS_NAMES)):
        idx = np.flatnonzero(labels.numpy() == class_id)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])

    return np.concatenate(train_idx), np.concatenate(val_idx)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion) -> tuple[float, float]:
    """Return (mean loss, accuracy) over a loader."""
    # eval() disables dropout. Forgetting this inflates validation loss,
    # because you would be measuring a randomly-thinned network rather than
    # the model you intend to ship.
    model.eval()
    total_loss, correct, count = 0.0, 0, 0

    for features, labels in loader:
        logits = model(features)
        total_loss += criterion(logits, labels).item() * len(labels)
        # logit > 0 is exactly equivalent to sigmoid(logit) > 0.5, and skips
        # computing the sigmoid at all.
        correct += ((logits > 0).float() == labels).sum().item()
        count += len(labels)

    return total_loss / count, correct / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=256, help="0 = linear probe")
    parser.add_argument("--dropout", type=float, default=0.2)
    # 300 is well past convergence for the MLP -- it overfits hard after ~90
    # epochs. That is intentional: best-checkpoint selection picks the minimum
    # regardless, and running past the turn makes the overfit visible in the
    # curve instead of leaving you guessing whether more epochs would help.
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="head.pt")
    args = parser.parse_args()

    # Seeding both generators makes the run reproducible: same split, same
    # weight init, same dropout masks, same shuffling.
    torch.manual_seed(args.seed)

    features, labels = load_embeddings("train")
    train_idx, val_idx = stratified_split(labels, args.val_fraction, args.seed)

    print(f"train {len(train_idx)}  val {len(val_idx)}  (test held back entirely)")
    for name, idx in (("train", train_idx), ("val", val_idx)):
        counts = torch.bincount(labels[idx].long(), minlength=len(CLASS_NAMES))
        print(
            f"  {name}: "
            + "  ".join(f"{c}={n}" for c, n in zip(CLASS_NAMES, counts.tolist(), strict=False))
        )

    train_loader = DataLoader(
        TensorDataset(features[train_idx], labels[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,  # reshuffled every epoch; without it the optimizer sees
    )  # the same gradient sequence each pass and can cycle.
    val_loader = DataLoader(TensorDataset(features[val_idx], labels[val_idx]), batch_size=1024)

    model = DetectorHead(hidden=args.hidden, dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    kind = "linear probe" if args.hidden == 0 else f"MLP (hidden={args.hidden})"
    print(f"\nhead: {kind}, {n_params:,} trainable parameters")

    criterion = nn.BCEWithLogitsLoss()
    # AdamW applies weight decay directly to the weights rather than folding it
    # into the gradient the way Adam does. With adaptive per-parameter step
    # sizes the two are not equivalent, and AdamW's version is the one that
    # actually regularises as intended.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"\n{'epoch':>5}  {'train_loss':>10}  {'val_loss':>9}  {'val_acc':>8}")
    print(f"{'':>5}  {'(0.693 = ':>10}  {'random)':>9}")

    best_val_loss = float("inf")
    best_state, best_epoch = None, -1
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()  # re-enables dropout after evaluate() turned it off
        running, seen = 0.0, 0

        for batch_features, batch_labels in train_loader:
            # Zero first: PyTorch *accumulates* gradients into .grad rather
            # than overwriting. Skip this and every step uses the sum of all
            # gradients so far, which diverges almost immediately.
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()  # populates .grad for the head only -- the
            # backbone is not in this graph at all
            optimizer.step()

            running += loss.item() * len(batch_labels)
            seen += len(batch_labels)

        train_loss = running / seen
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        )

        # Keep the best-validation weights, not the last epoch's. If val loss
        # rises at the end, the final weights are strictly worse than these.
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <- best"

        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>7.2%}{marker}")

    print(f"\ntrained in {time.time() - started:.1f}s")
    print(f"best epoch {best_epoch}: val_loss {best_val_loss:.4f}")

    # Diagnose the epoch we actually ship, not the last one we happened to run.
    # Running deliberately past convergence means the final-epoch gap is
    # expected to look terrible; that is not the model being saved.
    best = history[best_epoch - 1]
    gap = best["val_loss"] - best["train_loss"]
    print(f"train/val gap at best epoch: {gap:+.4f}", end="  ")
    if best["train_loss"] > 0.5:
        print("-> still near random; underfitting or a bug, not overfitting")
    elif gap > 0.15:
        print("-> val lagging train; overfitting, try more dropout/weight decay")
    else:
        print("-> healthy")

    if best_epoch == args.epochs:
        print("NOTE: best epoch is the last epoch -- still improving, train longer")
    final_gap = history[-1]["val_loss"] - history[-1]["train_loss"]
    if final_gap > gap + 0.1:
        print(
            f"NOTE: overfit hard after epoch {best_epoch} "
            f"(final gap {final_gap:+.4f}); best-checkpoint selection saved it"
        )

    model.load_state_dict(best_state)
    out_path = ARTIFACT_DIR / args.out
    save_head(
        model,
        out_path,
        meta={
            # The serving contract. Stage 2 reads these to guarantee it builds
            # features the same way training did.
            "clip_model": CLIP_MODEL,
            "clip_pretrained": CLIP_PRETRAINED,
            "l2_normalised": True,
            "class_names": CLASS_NAMES,
            "positive_class": CLASS_NAMES[1],
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_size": len(train_idx),
            "hyperparameters": vars(args),
        },
    )
    print(f"saved {out_path}")

    history_path = ARTIFACT_DIR / f"{out_path.stem}.history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"saved {history_path}")


if __name__ == "__main__":
    main()
