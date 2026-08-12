"""The full image -> prediction pipeline, shared by the API and any CLI use.

Everything that has to match between training and serving lives here, in one
place, so it cannot drift:

  * the backbone identity is read from the checkpoint's own metadata, not from
    config.py -- the artifact declares which backbone it was trained against,
    and this loads exactly that;
  * the preprocessing transform is taken from the model object, so the resize,
    crop and normalisation constants are guaranteed to be the ones the
    checkpoint expects;
  * L2 normalisation is applied if and only if the checkpoint says it was
    applied during training.

Feature-pipeline drift between training and serving is the single most common
way an ML system produces confidently wrong answers in production, and it never
raises an exception. This module exists to make that drift impossible rather
than merely unlikely.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from PIL import Image

from src.config import ARTIFACT_DIR
from src.model import load_head

DEFAULT_HEAD = ARTIFACT_DIR / "head.pt"


class Detector:
    """Loads once, predicts many times. Not cheap to construct -- see api.py."""

    def __init__(self, head_path: Path = DEFAULT_HEAD, device: str = "cpu") -> None:
        if not head_path.exists():
            raise FileNotFoundError(
                f"{head_path} not found -- train the head first (python -m src.train_head)"
            )

        # torch defaults to one compute thread per physical core, measured
        # against the *host*. Inside a container limited to 1 vCPU it will
        # still spawn e.g. 8 threads, which then fight for one core -- the
        # context-switching makes inference slower than single-threaded, and
        # each thread costs stack and arena memory. TORCH_THREADS lets the
        # deployment match the CPU allocation it actually paid for.
        threads = os.getenv("TORCH_THREADS")
        if threads:
            torch.set_num_threads(int(threads))

        self.device = torch.device(device)
        self.head, self.meta = load_head(head_path, map_location=device)
        self.head.to(self.device)

        import open_clip

        # Built from the checkpoint's metadata, NOT from config.py. If someone
        # edits config.py to a different backbone, this still loads the one the
        # head was actually trained on, and predictions stay correct.
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            self.meta["clip_model"], pretrained=self.meta["clip_pretrained"]
        )

        # Keep the image tower only, and let the text tower be garbage
        # collected. CLIP is two encoders trained jointly; we classify images
        # and never embed a single string, so the ~63M-parameter text
        # transformer plus its 49k-token embedding table is dead weight -- about
        # 250 MB of resident memory that would sit there for the process
        # lifetime doing nothing.
        #
        # This is exact, not an approximation: open_clip's encode_image() is
        # literally `self.visual(image)` when normalize=False, so calling the
        # visual tower directly produces bit-identical embeddings.
        backbone = clip_model.visual
        del clip_model

        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval().to(self.device)

        self.backbone = backbone
        self.preprocess = preprocess
        self.l2_normalise = bool(self.meta.get("l2_normalised", True))
        self.class_names = list(self.meta["class_names"])
        self.head_path = head_path

    @torch.inference_mode()
    def predict(self, images: list[Image.Image], threshold: float = 0.5) -> list[dict]:
        """Classify a batch of PIL images."""
        started = time.perf_counter()

        batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(
            self.device
        )

        features = self.backbone(batch)
        if self.l2_normalise:
            features = features / features.norm(dim=-1, keepdim=True)

        # The head emits raw logits; the sigmoid lives here, at inference time,
        # because BCEWithLogitsLoss fused it during training. Applying a sigmoid
        # inside the model would double-apply it.
        probabilities = torch.sigmoid(self.head(features.float()))

        elapsed_ms = (time.perf_counter() - started) * 1000
        per_image_ms = elapsed_ms / len(images)

        results = []
        for p_ai in probabilities.tolist():
            is_ai = p_ai >= threshold
            results.append(
                {
                    "label": self.class_names[1] if is_ai else self.class_names[0],
                    # P(predicted class) -- confidence in the answer actually
                    # given, not P(ai). A confident "real" reads 0.98, not 0.02.
                    #
                    # Note this is only >= 0.5 when threshold == 0.5. With a
                    # threshold of 0.45 and p_ai = 0.499 the label is "ai" while
                    # this value is 0.499, because the model marginally favours
                    # "real" and the *policy* overrode it. That is the correct
                    # number; it just means a caller cannot render it as
                    # "N% confident" without checking the threshold first.
                    "confidence": p_ai if is_ai else 1.0 - p_ai,
                    "p_ai": p_ai,
                    "threshold": threshold,
                    "inference_ms": round(per_image_ms, 1),
                }
            )
        return results

    def warmup(self) -> float:
        """Run one throwaway prediction so the first real request isn't slow.

        The first forward pass through a fresh torch process pays for lazy
        kernel selection, thread-pool spin-up and allocator warm-up -- often
        several times the steady-state latency. Paying that during startup
        instead of on a user's request is close to free and removes an ugly
        outlier from your p99.
        """
        started = time.perf_counter()
        self.predict([Image.new("RGB", (224, 224), color=(128, 128, 128))])
        return (time.perf_counter() - started) * 1000

    def info(self) -> dict:
        """Model provenance, surfaced through /health for debuggability."""
        return {
            "head": self.head_path.name,
            "head_parameters": sum(p.numel() for p in self.head.parameters()),
            "backbone": f"{self.meta['clip_model']}/{self.meta['clip_pretrained']}",
            "l2_normalised": self.l2_normalise,
            "class_names": self.class_names,
            "positive_class": self.meta.get("positive_class"),
            "trained_on": self.meta.get("train_size"),
            "best_epoch": self.meta.get("best_epoch"),
            "device": str(self.device),
        }
