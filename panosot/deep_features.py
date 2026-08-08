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
    feature_layer: int | None = 12
    normalize_features: bool = False
    template_size: int = 112
    coarse_search_size: int = 224
    refine_search_size: int = 160
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    pretrained: bool = True
    cache_dir: str | None = None
    tracking_adapter_path: str | None = None
    # MobileNet convolution blocks benefit from channels-last tensors on CUDA.
    # Keep this opt-in at the config level but enabled by the tracker CLI when
    # CUDA is requested; CPU behavior remains unchanged.
    use_channels_last: bool = True
    cudnn_benchmark: bool = True


class DeepFeatureExtractor:
    """Feature extractor wrapper around a lightweight vision backbone."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self._torch = _require_torch()
        torch = self._torch
        self.device = self._resolve_device(config.device)
        self._configure_cache_dir(config.cache_dir)
        self.model = self._build_model_with_fallback()
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            if config.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            if config.use_channels_last:
                try:
                    self.model.to(memory_format=torch.channels_last)
                except Exception:
                    # Some optional backbones do not expose a channels-last
                    # compatible parameter layout; fall back transparently.
                    pass
        self.forward_calls = 0

        self._mean = torch.tensor(config.mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(config.std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.adapter = None
        self.adapter_enabled = True
        if config.tracking_adapter_path:
            self.load_tracking_adapter(config.tracking_adapter_path)

    def _configure_cache_dir(self, cache_dir: str | None) -> None:
        if not cache_dir:
            return
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(cache_path))
        self._torch.hub.set_dir(str(cache_path))

    def _build_model_with_fallback(self) -> Any:
        try:
            return build_backbone(
                self.config.backbone_name,
                pretrained=self.config.pretrained,
                feature_layer=self.config.feature_layer,
            )
        except Exception as exc:
            if not self.config.pretrained:
                raise
            warnings.warn(
                f"Falling back to pretrained=False for {self.config.backbone_name}: {exc}",
                RuntimeWarning,
            )
            return build_backbone(
                self.config.backbone_name,
                pretrained=False,
                feature_layer=self.config.feature_layer,
            )

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

    def load_tracking_adapter(self, checkpoint_path: str | Path) -> None:
        """Load a trained 1x1 tracking adapter without changing the backbone."""
        from .models import ResidualTrackingProjection, TrackingProjection

        checkpoint = self._torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        channels = int(checkpoint["channels"])
        adapter_cls = ResidualTrackingProjection if checkpoint.get("residual", False) else TrackingProjection
        adapter = adapter_cls(channels).to(self.device)
        adapter.load_state_dict(checkpoint["state_dict"])
        adapter.eval()
        self.adapter = adapter

    def _postprocess_features(self, features: Any) -> Any:
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if self.adapter is not None and getattr(self, "adapter_enabled", True):
            # Backbone features are produced under inference_mode. Keep the
            # frozen adapter in the same mode; otherwise Conv2d attempts to
            # save backward state for an inference tensor at runtime.
            with self._torch.inference_mode():
                with self._amp_context():
                    features = self.adapter(features)
        if self.config.normalize_features:
            features = self._torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-6)
        return features

    def set_tracking_adapter_enabled(self, enabled: bool) -> None:
        """Enable/disable the optional adapter for the current sequence."""
        self.adapter_enabled = bool(enabled)

    def preprocess_patch(
        self,
        patch: np.ndarray,
        out_size: int,
        *,
        assume_normalized: bool = False,
    ) -> Any:
        """Convert one RGB patch to a model input tensor.

        Tracker-generated tangent patches are already float32 in [0, 1].
        Callers handling external data should keep the default safe path.
        """
        if patch.ndim != 3 or patch.shape[2] != 3:
            raise ValueError("Expected patch to have shape [H, W, 3].")

        torch = self._torch
        if patch.dtype != np.float32:
            patch = patch.astype(np.float32)
        if not assume_normalized:
            patch = np.clip(patch, 0.0, 1.0, out=patch)

        tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = (tensor - self._mean) / self._std
        if self.device.type == "cuda" and self.config.use_channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def preprocess_patches(
        self,
        patches: list[np.ndarray],
        out_size: int,
        *,
        assume_normalized: bool = False,
    ) -> Any:
        if not patches:
            raise ValueError("patches must not be empty.")
        if any(patch.ndim != 3 or patch.shape[2] != 3 for patch in patches):
            raise ValueError("Expected every patch to have shape [H, W, 3].")

        normalized = []
        for patch in patches:
            if patch.dtype != np.float32:
                patch = patch.astype(np.float32)
            normalized.append(
                patch if assume_normalized else np.clip(patch, 0.0, 1.0)
            )

        torch = self._torch
        tensor = torch.from_numpy(np.stack(normalized, axis=0)).permute(0, 3, 1, 2).to(self.device)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = (tensor - self._mean) / self._std
        if self.device.type == "cuda" and self.config.use_channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def _forward(
        self,
        patch: np.ndarray,
        out_size: int,
        *,
        assume_normalized: bool = False,
    ) -> Any:
        tensor = self.preprocess_patch(
            patch, out_size, assume_normalized=assume_normalized,
        )
        self.forward_calls += 1
        with self._torch.inference_mode():
            with self._amp_context():
                features = self.model(tensor)
        return self._postprocess_features(features)

    def reset_stats(self) -> None:
        self.forward_calls = 0

    def extract_template_feature(
        self,
        patch: np.ndarray,
        *,
        assume_normalized: bool = False,
    ) -> Any:
        return self._forward(
            patch,
            self.config.template_size,
            assume_normalized=assume_normalized,
        )

    def extract_search_feature(
        self,
        patch: np.ndarray,
        refine: bool = False,
        *,
        assume_normalized: bool = False,
        out_size: int | None = None,
    ) -> Any:
        size = int(out_size or (self.config.refine_search_size if refine else self.config.coarse_search_size))
        return self._forward(patch, size, assume_normalized=assume_normalized)

    def _forward_batch(
        self,
        patches: list[np.ndarray],
        out_size: int,
        chunk_size: int = 32,
        *,
        assume_normalized: bool = False,
    ) -> Any:
        if not patches:
            raise ValueError("patches must not be empty.")

        torch = self._torch
        features_by_chunk: list[Any] = []
        for start in range(0, len(patches), chunk_size):
            tensor = self.preprocess_patches(
                patches[start : start + chunk_size],
                out_size,
                assume_normalized=assume_normalized,
            )
            self.forward_calls += 1
            with torch.inference_mode():
                with self._amp_context():
                    features = self.model(tensor)
            features_by_chunk.append(self._postprocess_features(features))

        if len(features_by_chunk) == 1:
            return features_by_chunk[0]
        return torch.cat(features_by_chunk, dim=0)

    def extract_template_features_batch(
        self,
        patches: list[np.ndarray],
        chunk_size: int = 32,
        *,
        assume_normalized: bool = False,
    ) -> Any:
        return self._forward_batch(
            patches,
            self.config.template_size,
            chunk_size,
            assume_normalized=assume_normalized,
        )

    def extract_search_features_batch(
        self,
        patches: list[np.ndarray],
        refine: bool = False,
        chunk_size: int = 32,
        *,
        assume_normalized: bool = False,
        out_size: int | None = None,
    ) -> Any:
        size = int(out_size or (self.config.refine_search_size if refine else self.config.coarse_search_size))
        return self._forward_batch(
            patches,
            size,
            chunk_size,
            assume_normalized=assume_normalized,
        )
