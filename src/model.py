"""The trainable classification head, plus its save/load contract.

This module is deliberately separate from the training script, because the
FastAPI app in Stage 2 must import *this exact class* to reconstruct the model
from a checkpoint. A PyTorch state_dict is only weights -- it carries no
architecture. If serving rebuilt the head from a hardcoded guess at the layer
sizes and training later changed them, load_state_dict would either throw or,
worse, silently succeed with the wrong shapes.

So the checkpoint stores its own architecture config, and `load_head` rebuilds
from that rather than from an assumption.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.config import EMBED_DIM


class DetectorHead(nn.Module):
    """Maps a CLIP image embedding to a single logit.

    Output is a raw logit, NOT a probability. BCEWithLogitsLoss needs logits
    (it fuses the sigmoid internally for numerical stability), so applying a
    sigmoid here would both break training and double-apply it at inference.
    Callers that want P(ai) call torch.sigmoid() themselves.
    """

    def __init__(self, in_dim: int = EMBED_DIM, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.config = {"in_dim": in_dim, "hidden": hidden, "dropout": dropout}

        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            # hidden=0 gives a pure linear probe -- the baseline that tells you
            # whether the hidden layer is actually earning its keep.
            self.net = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze(-1) turns (B, 1) into (B,) so the shape matches the (B,)
        # float label tensor BCEWithLogitsLoss expects. Mismatched shapes here
        # do not error -- they broadcast into a (B, B) loss and quietly train
        # nonsense, which is a genuinely nasty bug to track down.
        return self.net(x).squeeze(-1)


def save_head(model: DetectorHead, path: Path, meta: dict) -> None:
    """Write weights + architecture + provenance as one self-describing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config,
            "meta": meta,
        },
        path,
    )


def load_head(path: Path, map_location: str = "cpu") -> tuple[DetectorHead, dict]:
    """Rebuild a head from a checkpoint. Returns (model in eval mode, meta)."""
    # weights_only=True refuses to unpickle arbitrary Python objects. Our
    # checkpoint is only tensors, dicts, str and float, so this costs nothing
    # -- and torch.load on an untrusted file without it is arbitrary code
    # execution, which matters once a checkpoint is fetched at deploy time.
    blob = torch.load(path, map_location=map_location, weights_only=True)

    model = DetectorHead(**blob["config"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["meta"]
