"""Shared paths and constants.

Every script imports from here so that "where does the data live" and "which
label means what" are stated exactly once. Getting these out of sync between
the training script and the serving code is one of the more common ways an
ML project quietly produces wrong answers.
"""

from pathlib import Path

# config.py lives at <root>/src/config.py, so parents[1] is the project root.
# Deriving paths from __file__ rather than the current working directory means
# the scripts behave the same whether you run them from the root, from src/,
# or from inside a container.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CIFAKE_DIR = DATA_DIR / "cifake"  # extracted PNGs, split/class subdirs
EMBED_DIR = DATA_DIR / "embeddings"  # cached CLIP feature vectors (.npz)
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"  # trained head weights + metadata

# Upstream HuggingFace dataset. Labels there are 0=FAKE, 1=REAL.
HF_DATASET = "dragonintelligence/CIFAKE-image-dataset"

# --- Our label convention -------------------------------------------------
# Index == integer label. Class 1 is the "positive" class, i.e. the thing the
# detector is trying to find, so precision/recall read the way you'd expect.
CLASS_NAMES = ["real", "ai"]
LABEL_REAL = 0
LABEL_AI = 1

# --- Backbone -------------------------------------------------------------
# ViT-B/32 is the smallest/fastest CLIP variant: 32x32 patches means only 49
# image tokens, so it is ~4x cheaper than ViT-B/16 at similar quality for a
# linear probe. laion2b_s34b_b79k is the LAION-2B checkpoint.
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512  # output width of ViT-B-32's image tower
