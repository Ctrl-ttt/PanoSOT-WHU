from __future__ import annotations

from typing import Any


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the deep feature pipeline. "
            "Install optional dependencies such as torch and torchvision to use this module."
        ) from exc
    return torch, nn


class DepthwiseXCorrHead:
    """Depthwise cross-correlation similarity head for template-search matching."""

    def __init__(self) -> None:
        torch, nn = _require_torch()
        self._torch = torch
        self._nn = nn

    def _channelwise_conv(self, template_feat: Any, search_feat: Any) -> Any:
        torch = self._torch
        batch = int(template_feat.shape[0])
        channels = int(template_feat.shape[1])
        kernel_h = int(template_feat.shape[-2])
        kernel_w = int(template_feat.shape[-1])
        search_h = int(search_feat.shape[-2])
        search_w = int(search_feat.shape[-1])

        kernel = template_feat.reshape(batch * channels, 1, kernel_h, kernel_w)
        search = search_feat.reshape(1, batch * channels, search_h, search_w)
        response = torch.nn.functional.conv2d(search, kernel, groups=batch * channels)
        return response.reshape(batch, channels, response.shape[-2], response.shape[-1])

    def forward(self, template_feat: Any, search_feat: Any) -> Any:
        if template_feat.ndim != 4 or search_feat.ndim != 4:
            raise ValueError("Expected template and search features to have shape [B, C, H, W].")
        if template_feat.shape[1] != search_feat.shape[1]:
            raise ValueError("Template and search features must have the same channel count.")
        if search_feat.shape[0] == 1 and template_feat.shape[0] > 1:
            search_feat = search_feat.expand(template_feat.shape[0], -1, -1, -1)
        if template_feat.shape[0] != search_feat.shape[0]:
            raise ValueError("Template and search batches must have the same batch size.")

        response = self._channelwise_conv(template_feat, search_feat)
        area = max(int(template_feat.shape[-2]) * int(template_feat.shape[-1]), 1)
        return response.mean(dim=1, keepdim=True) / area

    __call__ = forward


def build_backbone(name: str, pretrained: bool = True, feature_layer: int | None = 12) -> Any:
    torch, nn = _require_torch()
    normalized_name = name.strip().lower()

    if normalized_name == "mobilenet_v3_small":
        try:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as exc:
            raise ImportError(
                "torchvision is required to build the mobilenet_v3_small backbone."
            ) from exc

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        features = model.features
        if feature_layer is None:
            return features
        layers = list(features.children())
        if feature_layer < 0 or feature_layer >= len(layers):
            raise ValueError(f"feature_layer must be in [0, {len(layers) - 1}], got {feature_layer}")
        return nn.Sequential(*layers[: feature_layer + 1])

    if normalized_name.startswith("timm:"):
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for timm-backed backbones.") from exc

        timm_name = normalized_name.split(":", 1)[1]
        return timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            features_only=False,
        )

    raise ValueError(f"Unsupported backbone: {name}")


def build_similarity_head(name: str = "depthwise_xcorr") -> Any:
    normalized_name = name.strip().lower()
    if normalized_name == "depthwise_xcorr":
        return DepthwiseXCorrHead()
    raise ValueError(f"Unsupported similarity head: {name}")
