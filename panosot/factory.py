from __future__ import annotations

from pathlib import Path
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
    tracking_adapter_path = kwargs.pop("tracking_adapter_path", None)
    if cache_dir is None:
        project_cache = Path(__file__).resolve().parents[1] / ".cache" / "torch"
        if project_cache.is_dir():
            cache_dir = str(project_cache)
    config = TrackerConfig(use_deep_features=use_deep_features, backbone_name=backbone_name, device=device)

    for key, value in kwargs.items():
        if hasattr(config, key):
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
            tracking_adapter_path=tracking_adapter_path,
        )
        deep_extractor = DeepFeatureExtractor(feat_config)
        similarity_head = build_similarity_head("depthwise_xcorr")
        config.device = resolved_device

    return PanoSOTTracker(config=config, deep_extractor=deep_extractor, similarity_head=similarity_head)
