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
        self.forward_calls = 0

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
        self.forward_calls += 1
        with self._torch.inference_mode():
            with self._amp_context():
                features = self.model(tensor)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if self.config.normalize_features:
            features = self._torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-6)
        return features

    def reset_stats(self) -> None:
        self.forward_calls = 0

    def extract_template_feature(self, patch: np.ndarray) -> Any:
        return self._forward(patch, self.config.template_size)

    def extract_search_feature(self, patch: np.ndarray, refine: bool = False) -> Any:
        out_size = self.config.refine_search_size if refine else self.config.coarse_search_size
        return self._forward(patch, out_size)

    # ---------- 批量特征提取（P0-1 / P0-2 / P1-1）----------

    def _forward_batch(self, patches: list[np.ndarray], out_size: int, chunk_size: int = 32) -> Any:
        """批量前向：将多个 patch 拼成一个大 batch，一次推理完成。

        Args:
            patches: N 个 [H, W, 3] float32 数组
            out_size: 统一缩放尺寸
            chunk_size: 分块大小，防止 GPU OOM

        Returns:
            [N, C, h, w] 的特征张量
        """
        if not patches:
            raise ValueError("patches must not be empty.")
        torch = self._torch
        results: list[Any] = []
        for start in range(0, len(patches), chunk_size):
            chunk = patches[start : start + chunk_size]
            tensors = [self.preprocess_patch(p, out_size) for p in chunk]
            batch_tensor = torch.cat(tensors, dim=0)  # [chunk, 3, out_size, out_size]
            self.forward_calls += 1
            with torch.inference_mode():
                with self._amp_context():
                    features = self.model(batch_tensor)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            if self.config.normalize_features:
                features = torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-6)
            results.append(features)
        if len(results) == 1:
            return results[0]
        return torch.cat(results, dim=0)

    def extract_search_features_batch(
        self,
        patches: list[np.ndarray],
        refine: bool = False,
        chunk_size: int = 32,
    ) -> Any:
        """批量提取搜索区域特征，一次或数次前向替代 N 次。"""
        out_size = self.config.refine_search_size if refine else self.config.coarse_search_size
        return self._forward_batch(patches, out_size, chunk_size=chunk_size)

    def extract_template_features_batch(
        self,
        patches: list[np.ndarray],
        chunk_size: int = 32,
    ) -> Any:
        """批量提取模板特征，一次或数次前向替代 N 次。"""
        return self._forward_batch(patches, self.config.template_size, chunk_size=chunk_size)

    # ---------- 模型预热（P2-3）----------

    def warmup(self) -> None:
        """执行虚拟前向以触发 CUDA kernel 编译，消除首帧延迟。"""
        if getattr(self, "_warmup_done", False):
            return
        torch = self._torch
        dummy = np.zeros((self.config.template_size, self.config.template_size, 3), dtype=np.float32)
        with torch.inference_mode():
            with self._amp_context():
                _ = self.extract_template_feature(dummy)
                _ = self.extract_search_feature(dummy, refine=False)
                _ = self.extract_search_feature(dummy, refine=True)
        self._warmup_done = True
