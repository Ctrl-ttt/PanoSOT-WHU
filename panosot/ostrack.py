"""OSTrack single-stream ViT backend (backend upgrade, Phase 2).

Self-contained reimplementation of OSTrack (ECCV 2022, "Joint Feature
Learning and Relation Modeling for Tracking") that does not depend on timm:

- a ViT-Base/16 backbone whose template and search tokens are concatenated and
  processed jointly (single-stream relation modeling),
- a center-prediction box head with three parallel branches (score / offset /
  size), and
- weight loading helpers compatible with the official checkpoint layout
  (net.* prefixed .pth files) and with safetensors conversions that keep the
  backbone.* / box_head.* key names.

The default variant mirrors OSTrack-384 (template 192, search 384).  A
256-sized variant (template 128, search 256) is provided for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional
import urllib.request

import numpy as np

__all__ = [
    "OSTrackConfig",
    "OSTrackVisionTransformer",
    "OSTrackBoxHead",
    "OSTrackNet",
    "build_ostrack",
    "load_ostrack_checkpoint",
    "load_mae_backbone",
    "download_ostrack_384_weights",
    "remap_checkpoint_keys",
    "hann2d",
    "OSTRACK_MEAN",
    "OSTRACK_STD",
]

OSTRACK_MEAN = (0.485, 0.456, 0.406)
OSTRACK_STD = (0.229, 0.224, 0.225)

# MIT-licensed safetensors conversion of the official OSTrack-384 checkpoint
# (served through the HF mirror; the canonical upstream is botaoye/OSTrack).
OSTRACK_384_URL = (
    "https://hf-mirror.com/eek/OSTrack_vitb_384_mae_ce_32x4_ep300/"
    "resolve/main/OSTrack_vitb_384_mae_ce_32x4_ep300.safetensors"
)
OSTRACK_384_FILENAME = "OSTrack_vitb_384_mae_ce_32x4_ep300.safetensors"
OSTRACK_384_EXPECTED_BYTES = 371_338_948

# MAE-pretrained ViT-B backbone initialization (the same source OSTrack uses).
MAE_VIT_BASE_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the OSTrack backend. "
            "Install torch and torchvision to use this module."
        ) from exc
    return torch, nn, F


# nn.Module bases below need torch available at import time; the OSTrack
# backend module is torch-only by design.
torch, nn, _F = _require_torch()


def _trunc_normal_(tensor: Any, std: float = 0.02) -> None:
    torch, _, _ = _require_torch()
    torch.nn.init.trunc_normal_(tensor, std=std)


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample; identity at inference."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Any) -> Any:
        torch, _, _ = _require_torch()
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        torch, nn, _ = _require_torch()
        super().__init__()
        hidden = int(hidden_features or in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: Any) -> Any:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    """Multi-head attention in the concatenated qkv form used by the ViT."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        torch, nn, _ = _require_torch()
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(float(proj_drop))

    def forward(self, x: Any) -> Any:
        torch, _, _ = _require_torch()
        batch, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        torch, nn, _ = _require_torch()
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: Any) -> Any:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """Linear projection of image patches (16x16, stride 16)."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        torch, nn, _ = _require_torch()
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Any) -> Any:
        return self.proj(x)


@dataclass
class OSTrackConfig:
    template_size: int = 192
    search_size: int = 384
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    head_channel: int = 256
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # The downloadable 384 checkpoint trains the size branch with a sigmoid
    # output in [0, 1] (fraction of the search patch).  Official cell-unit
    # checkpoints can set this to False to use the raw cell counts instead.
    size_sigmoid: bool = True

    @property
    def z_patches(self) -> int:
        return (self.template_size // self.patch_size) ** 2

    @property
    def x_patches(self) -> int:
        return (self.search_size // self.patch_size) ** 2

    @property
    def feat_sz(self) -> int:
        return self.search_size // self.patch_size


def ostrack_384() -> OSTrackConfig:
    return OSTrackConfig(template_size=192, search_size=384)


def ostrack_256() -> OSTrackConfig:
    return OSTrackConfig(template_size=128, search_size=256)


class OSTrackVisionTransformer(nn.Module):
    """ViT-Base/16 backbone with separate template/search positional embeds.

    cls_token and the single-view pos_embed are kept (matching the checkpoint
    layout) but unused in the forward pass: OSTrack prepends no class token
    and positions the template/search streams with pos_embed_z / pos_embed_x.
    """

    def __init__(self, config: OSTrackConfig) -> None:
        torch, nn, _ = _require_torch()
        super().__init__()
        embed_dim = int(config.embed_dim)
        self.config = config
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size, in_chans=3, embed_dim=embed_dim
        )
        # Vestigial single-stream embedding kept for checkpoint compatibility.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        base_patches = (224 // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, base_patches + 1, embed_dim))
        self.pos_embed_z = nn.Parameter(torch.zeros(1, config.z_patches, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, config.x_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=float(config.drop_rate))

        dpr = [
            float(v)
            for v in torch.linspace(0.0, float(config.drop_path_rate), config.depth)
        ]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        _trunc_normal_(self.pos_embed, std=0.02)
        _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.pos_embed_z, std=0.02)
        _trunc_normal_(self.pos_embed_x, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: Any) -> None:
        torch, nn, _ = _require_torch()
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def patchify(self, images: Any) -> Any:
        tokens = self.patch_embed(images)
        return tokens.flatten(2).transpose(1, 2)

    def forward_tokens(self, z_tokens: Any, x_tokens: Any) -> Any:
        torch, _, _ = _require_torch()
        z = z_tokens + self.pos_embed_z
        x = x_tokens + self.pos_embed_x
        tokens = self.pos_drop(torch.cat((z, x), dim=1))
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class OSTrackBoxHead(nn.Module):
    """Center prediction head: score / offset / size parallel branches.

    Input is the concatenated backbone token sequence; only the search tokens
    are consumed.  cal_bbox follows the official convention and returns a
    normalized cxcywh box in [0, 1] (fraction of the search patch).
    """

    def __init__(self, config: OSTrackConfig) -> None:
        torch, nn, _ = _require_torch()
        super().__init__()
        self.config = config
        self.feat_sz = int(config.feat_sz)
        self.stride = int(config.patch_size)
        inplanes = int(config.embed_dim)
        channel = int(config.head_channel)
        for branch in ("ctr", "offset", "size"):
            in_c = inplanes
            for i in range(4):
                out_c = channel // (2 ** i)
                setattr(self, f"conv{i + 1}_{branch}", self._conv_relu(in_c, out_c))
                in_c = out_c
            out_dim = 1 if branch == "ctr" else 2
            setattr(
                self, f"conv5_{branch}", nn.Conv2d(channel // 8, out_dim, kernel_size=1)
            )

    @staticmethod
    def _conv_relu(in_c: int, out_c: int) -> Any:
        torch, nn, _ = _require_torch()
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def _branch(self, x: Any, name: str) -> Any:
        for i in range(1, 5):
            x = getattr(self, f"conv{i}_{name}")(x)
        return getattr(self, f"conv5_{name}")(x)

    def forward(self, tokens: Any, z_len: int) -> tuple[Any, Any, Any]:
        torch, _, _ = _require_torch()
        search_tokens = tokens[:, z_len:]
        batch = search_tokens.shape[0]
        features = search_tokens.transpose(1, 2).reshape(
            batch, self.config.embed_dim, self.feat_sz, self.feat_sz
        )
        score = self._branch(features, "ctr")
        offset = self._branch(features, "offset")
        size = self._branch(features, "size")
        score = torch.clamp(torch.sigmoid(score), min=1e-4, max=1.0 - 1e-4)
        if self.config.size_sigmoid:
            size = torch.clamp(torch.sigmoid(size), min=1e-4, max=1.0 - 1e-4)
        return score, offset, size

    def cal_bbox(
        self,
        score_map: Any,
        size_map: Any,
        offset_map: Any,
        return_score: bool = False,
    ) -> Any:
        """Return normalized cxcywh boxes (fraction of the search patch)."""
        torch, _, _ = _require_torch()
        batch = score_map.shape[0]
        flat = score_map.flatten(1)
        max_score, idx = torch.max(flat, dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz
        idx2 = idx.unsqueeze(1).expand(batch, 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx2).float()
        offset = offset_map.flatten(2).gather(dim=2, index=idx2).squeeze(-1).float()
        cx = (idx_x.float() + offset[:, :1]) / self.feat_sz
        cy = (idx_y.float() + offset[:, 1:]) / self.feat_sz
        if self.config.size_sigmoid:
            wh = size.squeeze(-1)  # already the normalized fraction of search
        else:
            wh = (size / self.feat_sz).squeeze(-1)
        bbox = torch.cat((cx, cy, wh), dim=1)
        if return_score:
            return bbox, max_score
        return bbox


class OSTrackNet(nn.Module):
    """Full tracker network: backbone + box head."""

    def __init__(self, config: Optional[OSTrackConfig] = None) -> None:
        torch, nn, _ = _require_torch()
        super().__init__()
        self.config = config or ostrack_384()
        self.backbone = OSTrackVisionTransformer(self.config)
        self.box_head = OSTrackBoxHead(self.config)
        self._z_tokens: Any = None

    def initialize(self, z: Any) -> Any:
        """Cache template tokens; call once per sequence."""
        self._z_tokens = self.backbone.patchify(z)
        return self._z_tokens

    def forward(self, z: Any, x: Any) -> dict[str, Any]:
        z_tokens = self.backbone.patchify(z)
        x_tokens = self.backbone.patchify(x)
        tokens = self.backbone.forward_tokens(z_tokens, x_tokens)
        score, offset, size = self.box_head(tokens, z_tokens.shape[1])
        return {"score_map": score, "offset_map": offset, "size_map": size}

    def forward_search(self, x: Any, z_tokens: Optional[Any] = None) -> dict[str, Any]:
        """Run the search branch against cached (or supplied) template tokens."""
        z_tokens = z_tokens if z_tokens is not None else self._z_tokens
        if z_tokens is None:
            raise RuntimeError("OSTrackNet.initialize(z) must be called first.")
        x_tokens = self.backbone.patchify(x)
        # Grid relocalization batches multiple search patches against one
        # shared template.  Broadcast the cached template tokens across the
        # search batch so forward_tokens can concatenate them on dim=1.
        if x_tokens.shape[0] != z_tokens.shape[0]:
            if z_tokens.shape[0] == 1:
                z_tokens = z_tokens.expand(x_tokens.shape[0], -1, -1)
            else:
                raise ValueError(
                    "forward_search batch mismatch: template batch "
                    f"{z_tokens.shape[0]} vs search batch {x_tokens.shape[0]}"
                )
        tokens = self.backbone.forward_tokens(z_tokens, x_tokens)
        score, offset, size = self.box_head(tokens, z_tokens.shape[1])
        return {"score_map": score, "offset_map": offset, "size_map": size}

    def track(
        self,
        z: Any,
        x: Any,
        window: Optional[Any] = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Return (normalized bbox, best score, raw head outputs)."""
        out = self.forward(z, x)
        score = out["score_map"]
        if window is not None:
            score = score * window
        bbox, best = self.box_head.cal_bbox(
            score, out["size_map"], out["offset_map"], return_score=True
        )
        return bbox, best, out


def build_ostrack(variant: str = "384") -> OSTrackNet:
    """Build an OSTrack network for "384" or "256"."""
    normalized = str(variant).strip().lower()
    if normalized in {"384", "vitb_384"}:
        config = ostrack_384()
    elif normalized in {"256", "vitb_256"}:
        config = ostrack_256()
    else:
        raise ValueError(f"Unsupported OSTrack variant: {variant}")
    return OSTrackNet(config)


def remap_checkpoint_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize checkpoint key prefixes to the local layout.

    Accepts the official net.* prefix and plain pos_embed_z / pos_embed_x
    keys (the official module keeps them at the net level).
    """
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized = str(key)
        for prefix in ("net.", "model.", "module."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        if normalized in {"pos_embed_z", "pos_embed_x"}:
            normalized = "backbone." + normalized
        remapped[normalized] = value
    return remapped


def _load_safetensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required to load .safetensors checkpoints "
            "(pip install safetensors)"
        ) from exc
    return load_file(str(path))


def load_ostrack_checkpoint(
    model: OSTrackNet,
    checkpoint_path: str | Path,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load an OSTrack checkpoint (.pth or .safetensors) into the model."""
    torch, _, _ = _require_torch()
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"OSTrack checkpoint not found: {path}")
    if path.suffix.lower() == ".safetensors":
        state_dict = _load_safetensors(path)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("net")
                or checkpoint.get("state_dict")
                or checkpoint.get("model")
                or checkpoint
            )
        else:
            raise ValueError(f"Unsupported checkpoint container in {path}")
    state_dict = remap_checkpoint_keys(state_dict)
    return model.load_state_dict(state_dict, strict=strict)


def load_mae_backbone(
    model: OSTrackNet | OSTrackVisionTransformer,
    path: str | Path,
) -> None:
    """Initialize the backbone with MAE-pretrained ViT-B weights."""
    torch, _, _ = _require_torch()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    backbone = model.backbone if isinstance(model, OSTrackNet) else model
    encoder = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("decoder")
        and key not in {"head.weight", "head.bias"}
    }
    backbone.load_state_dict(encoder, strict=False)


def _reporthook(blocks: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(blocks * block_size, total)
    percent = 100.0 * done / total
    print(
        f"\r[ostrack] downloading weights {done / 1e6:7.1f}/{total / 1e6:7.1f} MB ({percent:4.1f}%)",
        end="",
        flush=True,
    )


def download_ostrack_384_weights(
    cache_dir: Optional[str | Path] = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download the OSTrack-384 safetensors checkpoint into cache_dir."""
    if cache_dir is None:
        env_cache = os.environ.get("OSTRACK_CACHE_DIR")
        cache_dir = (
            Path(env_cache) if env_cache else Path.home() / ".cache" / "panosot-ostrack"
        )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / OSTRACK_384_FILENAME
    if (
        dest.is_file()
        and dest.stat().st_size == OSTRACK_384_EXPECTED_BYTES
        and not force
    ):
        return dest

    url = os.environ.get("OSTRACK_384_URL", OSTRACK_384_URL)
    tmp = dest.with_suffix(".part")
    if tmp.exists():
        tmp.unlink()
    print(f"[ostrack] downloading weights from {url}")
    reporthook = _reporthook if progress else None
    urllib.request.urlretrieve(url, tmp, reporthook=reporthook)
    if progress:
        print()
    actual = tmp.stat().st_size
    if actual != OSTRACK_384_EXPECTED_BYTES:
        tmp.unlink()
        raise RuntimeError(
            f"Downloaded OSTrack weights have unexpected size {actual} "
            f"(expected {OSTRACK_384_EXPECTED_BYTES})"
        )
    os.replace(tmp, dest)
    return dest


def hann2d(size: int, centered: bool = True, device: Optional[Any] = None) -> Any:
    """2D Hann window of shape [1, 1, size, size], peaking at 1.0 center.

    With periodic=True the window peaks at the center cell (index size // 2
    for even sizes).  Following the official OSTrack tracker the window is
    multiplied directly into the score map, so it must peak at 1.0.
    The centered flag is kept for API compatibility.
    """
    torch, _, _ = _require_torch()
    hann = torch.hann_window(size, periodic=True)
    window = torch.outer(hann, hann)
    window = window.view(1, 1, size, size)
    if device is not None:
        window = window.to(device)
    return window
