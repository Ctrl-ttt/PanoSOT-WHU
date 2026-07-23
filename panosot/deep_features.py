from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import os
import warnings
from typing import Any

import numpy as np

from .models import build_backbone


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the deep feature pipeline. "
            "Install optional dependencies such as torch and torchvision to use this module."
        ) from exc
    return torch


@dataclass
class FeatureConfig:
    backbone_name: str = "mobilenet_v3_small"
    device: str = "cuda"
    use_amp: bool = True
    template_size: int = 112
    coarse_search_size: int = 224
    refine_search_size: int = 160
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    pretrained: bool = True
    cache_dir: str | None = None


class DeepFeatureExtractor:
    """Feature extractor wrapper around a lightweight vision backbone."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self._torch = _require_torch()
        self.device = self._resolve_device(config.device)
        self._configure_cache_dir(config.cache_dir)
        self.model = self._build_model_with_fallback()
        self.model.to(self.device)
        self.model.eval()

        torch = self._torch
        self._mean = torch.tensor(config.mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(config.std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _configure_cache_dir(self, cache_dir: str | None) -> None:
        if not cache_dir:
            return
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(cache_path))
        self._torch.hub.set_dir(str(cache_path))

    def _build_model_with_fallback(self) -> Any:
        try:
            return build_backbone(self.config.backbone_name, pretrained=self.config.pretrained)
        except Exception as exc:
            if not self.config.pretrained:
                raise
            warnings.warn(
                f"Falling back to pretrained=False for {self.config.backbone_name}: {exc}",
                RuntimeWarning,
            )
            return build_backbone(self.config.backbone_name, pretrained=False)

    def _resolve_device(self, requested_device: str) -> Any:
        torch = self._torch
        normalized = requested_device.strip().lower()
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(normalized)

    def _amp_context(self) -> Any:
        torch = self._torch
        if self.config.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def preprocess_patch(self, patch: np.ndarray, out_size: int) -> Any:
        if patch.ndim != 3 or patch.shape[2] != 3:
            raise ValueError("Expected patch to have shape [H, W, 3].")

        torch = self._torch
        if patch.dtype != np.float32:
            patch = patch.astype(np.float32)
        patch = np.clip(patch, 0.0, 1.0, out=patch)

        tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = (tensor - self._mean) / self._std
        return tensor

    def _forward(self, patch: np.ndarray, out_size: int) -> Any:
        tensor = self.preprocess_patch(patch, out_size)
        with self._torch.no_grad():
            with self._amp_context():
                features = self.model(tensor)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return features

    def extract_template_feature(self, patch: np.ndarray) -> Any:
        return self._forward(patch, self.config.template_size)

    def extract_search_feature(self, patch: np.ndarray, refine: bool = False) -> Any:
        out_size = self.config.refine_search_size if refine else self.config.coarse_search_size
        return self._forward(patch, out_size)
