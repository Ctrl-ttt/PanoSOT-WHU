from __future__ import annotations

from dataclasses import fields
import warnings
from typing import Any

from .deep_features import DeepFeatureExtractor, FeatureConfig
from .models import build_similarity_head
from .tracker import PanoSOTTracker, TrackerConfig


def build_tracker(
    use_deep_features: bool = False,
    backbone_name: str = "mobilenet_v3_small",
    device: str = "auto",
    **kwargs: Any,
) -> PanoSOTTracker:
    cache_dir = kwargs.pop("cache_dir", None)
    config = TrackerConfig(use_deep_features=use_deep_features, backbone_name=backbone_name, device=device)

    # P3-1: 类型安全校验，未知参数发出警告
    valid_fields = {f.name for f in fields(TrackerConfig)}
    unknown = set(kwargs.keys()) - valid_fields
    if unknown:
        warnings.warn(
            f"Unknown TrackerConfig parameters (will be ignored): {sorted(unknown)}",
            UserWarning,
            stacklevel=2,
        )
        for key in unknown:
            kwargs.pop(key)

    for key, value in kwargs.items():
        setattr(config, key, value)

    deep_extractor = None
    similarity_head = None
    if use_deep_features:
        resolved_device = device
        if device == "auto":
            try:
                import torch

                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                resolved_device = "cpu"

        feat_config = FeatureConfig(
            backbone_name=backbone_name,
            device=resolved_device,
            use_amp=(resolved_device.startswith("cuda")),
            feature_layer=config.deep_feature_layer,
            normalize_features=config.normalize_deep_features,
            template_size=config.deep_template_size,
            coarse_search_size=config.coarse_search_size,
            refine_search_size=config.refine_search_size,
            cache_dir=cache_dir,
        )
        deep_extractor = DeepFeatureExtractor(feat_config)
        # P2-3: 模型预热，消除首帧 CUDA kernel 编译延迟
        deep_extractor.warmup()
        similarity_head = build_similarity_head("depthwise_xcorr")
        config.device = resolved_device

    return PanoSOTTracker(config=config, deep_extractor=deep_extractor, similarity_head=similarity_head)
