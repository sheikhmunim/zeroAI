"""Download CIFAKE and write a balanced subset to disk as PNG files.

CIFAKE (Bird & Lotfi, 2023) pairs the 60k real photographs of CIFAR-10 with
60k Stable-Diffusion-generated images of the same ten classes. Everything is
32x32 pixels.

Why write loose PNGs instead of just using the HuggingFace dataset object?
Three reasons, all practical rather than technical:
  1. You can open the folder and actually look at the data, which matters a
     lot more than people admit when a model misbehaves.
  2. The extraction script stays framework-agnostic -- it just globs a folder.
  3. Later stages need a couple of real image files on disk anyway (manual
     API testing in Stage 2, the CI smoke test in Stage 6).

Run:
    python -m src.download_data
    python -m src.download_data --train-per-class 1000 --test-per-class 500
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from src.config import CIFAKE_DIR, CLASS_NAMES, HF_DATASET, LABEL_AI, LABEL_REAL

# Upstream label ids -> ours. CIFAKE uses 0=FAKE, 1=REAL; we want AI to be the
# positive class (see config.py), so this mapping is a deliberate flip.
UPSTREAM_TO_OURS = {0: LABEL_AI, 1: LABEL_REAL}


def sample_balanced_indices(
    labels: np.ndarray, per_class: int, rng: np.random.Generator
) -> dict[int, np.ndarray]:
    """Pick `per_class` random row indices for each of our two classes.

    Sampling per class rather than taking a random slice of the whole split
    guarantees exact balance. CIFAKE happens to already be 50/50, but relying
    on that means the script silently produces a skewed subset the day you
    point it at a dataset that isn't.
    """
    chosen: dict[int, np.ndarray] = {}
    for upstream_label, our_label in UPSTREAM_TO_OURS.items():
        idx = np.flatnonzero(labels == upstream_label)
        if len(idx) < per_class:
            raise SystemExit(
                f"asked for {per_class} of class '{CLASS_NAMES[our_label]}' "
                f"but the split only has {len(idx)}"
            )
        chosen[our_label] = rng.choice(idx, size=per_class, replace=False)
    return chosen


def export_split(dataset, split_name: str, per_class: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    # Reading just the label column is cheap -- the parquet file is columnar,
    # so this does not decode a single image.
    labels = np.asarray(dataset["label"])
    chosen = sample_balanced_indices(labels, per_class, rng)

    counts = {}
    for our_label, indices in chosen.items():
        class_name = CLASS_NAMES[our_label]
        out_dir = CIFAKE_DIR / split_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # .select() is lazy; images are only decoded as we iterate below.
        subset = dataset.select(indices.tolist())
        for row_position, record in enumerate(subset):
            # Name files by their original dataset row so a suspicious image
            # can always be traced back upstream.
            original_row = int(indices[row_position])
            record["image"].save(out_dir / f"{original_row:06d}.png")

        counts[class_name] = len(indices)
        print(f"  {split_name}/{class_name}: {len(indices)} images -> {out_dir}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-class", type=int, default=5000)
    parser.add_argument("--test-per-class", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete any existing data/cifake/ before writing",
    )
    args = parser.parse_args()

    if CIFAKE_DIR.exists():
        if not args.force:
            raise SystemExit(f"{CIFAKE_DIR} already exists. Re-run with --force to replace it.")
        shutil.rmtree(CIFAKE_DIR)

    # Imported here rather than at module top so that `--help` responds
    # instantly instead of waiting on a multi-second library import.
    from datasets import load_dataset

    print(f"loading {HF_DATASET} (~50 MB, cached under ~/.cache/huggingface)")
    dataset = load_dataset(HF_DATASET)

    manifest = {
        "source": HF_DATASET,
        "seed": args.seed,
        "class_names": CLASS_NAMES,
        "splits": {},
    }
    for split_name, per_class in (
        ("train", args.train_per_class),
        ("test", args.test_per_class),
    ):
        print(f"exporting {split_name} ({per_class} per class)")
        manifest["splits"][split_name] = export_split(
            dataset[split_name], split_name, per_class, args.seed
        )

    manifest_path = CIFAKE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
