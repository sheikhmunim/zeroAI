"""Run every image through a frozen CLIP backbone once and cache the vectors.

This is the only expensive step in Stage 1. Because the backbone is frozen,
its output for a given image is a constant -- recomputing it every training
epoch would be pure waste. So we compute it once here, write it to disk, and
the training script never loads CLIP at all.

Output: data/embeddings/{split}.npz containing
    features : float32 (N, 512)  -- L2-normalised CLIP image embeddings
    labels   : int64   (N,)      -- 0 = real, 1 = ai  (see src/config.py)
    paths    : str     (N,)      -- source file, for tracing bad predictions

Run:
    python -m src.extract_embeddings
    python -m src.extract_embeddings --device cuda --batch-size 128
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image

from src.config import (
    CIFAKE_DIR,
    CLASS_NAMES,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    EMBED_DIM,
    EMBED_DIR,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_backbone(device: torch.device):
    """Load CLIP's image tower, frozen, in eval mode.

    Returns (model, preprocess). `preprocess` is the exact transform pipeline
    this checkpoint was trained with; we deliberately take it from the model
    rather than writing our own, because CLIP's normalisation constants are
    not the ImageNet ones and a mismatch fails silently.
    """
    import open_clip

    print(f"loading CLIP {CLIP_MODEL} / {CLIP_PRETRAINED} (first run downloads ~600 MB)")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )

    # This is "freezing". Two independent effects:
    #   - autograd stops recording operations on these tensors, so no graph is
    #     built and no gradient buffers are allocated;
    #   - we never hand these parameters to an optimizer, so nothing steps them.
    # The weights are bit-identical before and after the whole project.
    for param in model.parameters():
        param.requires_grad = False

    # eval() switches modules that behave differently at train vs inference
    # time (dropout, batchnorm) into inference behaviour. ViT-B/32 has no
    # batchnorm, but calling it is unconditional good practice -- the day you
    # swap in a backbone that does have it, forgetting this corrupts results.
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_params / 1e6:.1f}M parameters, {n_trainable} trainable")
    return model, preprocess


def collect_files(split: str) -> tuple[list, np.ndarray]:
    """Gather image paths for a split along with their integer labels."""
    paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = CIFAKE_DIR / split / class_name
        if not class_dir.is_dir():
            raise SystemExit(f"missing {class_dir} -- run `python -m src.download_data` first")
        files = sorted(class_dir.glob("*.png"))
        paths.extend(files)
        labels.extend([label] * len(files))
    return paths, np.asarray(labels, dtype=np.int64)


@torch.inference_mode()
def embed_split(
    model, preprocess, split: str, device, batch_size: int, limit: int | None = None
) -> None:
    paths, labels = collect_files(split)

    if limit is not None:
        # Stride across the whole list so both classes stay represented;
        # taking the first `limit` paths would give you nothing but 'real'
        # images, since collect_files() concatenates the classes in order.
        stride = max(1, len(paths) // limit)
        keep = list(range(0, len(paths), stride))[:limit]
        paths = [paths[i] for i in keep]
        labels = labels[keep]

    print(f"\n{split}: {len(paths)} images")

    features = np.empty((len(paths), EMBED_DIM), dtype=np.float32)
    started = time.time()

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]

        # preprocess() takes one PIL image and returns a (3, 224, 224) tensor;
        # stack turns the list into a (B, 3, 224, 224) batch. Batching matters
        # a lot on CPU -- per-image forward passes waste most of the available
        # matrix-multiply throughput.
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)

        embeddings = model.encode_image(batch)

        # L2-normalise: project each row onto the unit sphere. This is part of
        # the model contract, not a preprocessing detail -- serving must do it
        # too, or the head sees inputs on a scale it never trained on.
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        features[start : start + len(chunk)] = embeddings.float().cpu().numpy()

        done = start + len(chunk)
        elapsed = time.time() - started
        rate = done / elapsed
        eta = (len(paths) - done) / rate
        print(
            f"\r  {done}/{len(paths)}  {rate:.1f} img/s  eta {eta:>5.0f}s",
            end="",
            flush=True,
        )

    print(f"\r  {len(paths)}/{len(paths)} in {time.time() - started:.0f}s" + " " * 20)

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMBED_DIR / (f"{split}.npz" if limit is None else f"{split}.smoke.npz")
    np.savez_compressed(
        out_path,
        features=features,
        labels=labels,
        paths=np.asarray([str(p) for p in paths]),
        # Stamped so the training script can refuse to train on embeddings
        # produced by a different backbone or a different normalisation rule.
        clip_model=CLIP_MODEL,
        clip_pretrained=CLIP_PRETRAINED,
        l2_normalised=True,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {out_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only embed N images per split; writes {split}.smoke.npz (for testing)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")

    model, preprocess = load_backbone(device)
    for split in args.splits:
        embed_split(model, preprocess, split, device, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
