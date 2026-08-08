"""Train a lightweight tracker adapter from 360VOTS temporal annotations.

The training pair matches the runtime geometry: a template is extracted at
the annotated target in frame t, while the search patch in frame t+delta is
centred on the previous target state.  Only a residual 1x1 projection is
trained; the ImageNet backbone stays frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import zipfile

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.geometry import erp_bbox_to_state, state_size_to_fov, tangent_patch
from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.models import DepthwiseXCorrHead, ResidualTrackingProjection
from tools.train_airsim360_adapter import localization_loss


def hard_negative_loss(response_bank, torch, target_fractions, margin: float = 0.01):
    """Penalize mismatched template/search pairs in a batch.

    ``response_bank`` has shape [num_searches, num_templates, 1, H, W].
    The diagonal pair is the annotated positive; every off-diagonal pair is
    a hard negative and must stay below the positive target response.
    """
    if response_bank.ndim != 5 or response_bank.shape[0] != response_bank.shape[1]:
        raise ValueError("Expected square response bank [B, B, 1, H, W].")
    batch = int(response_bank.shape[0])
    height, width = int(response_bank.shape[-2]), int(response_bank.shape[-1])
    fractions = torch.as_tensor(target_fractions, dtype=torch.float32, device=response_bank.device)
    target_x = torch.floor(
        fractions[:, 0] * (2 * (width - 1)) - 0.5 * (width - 1) + 0.5
    ).long().clamp_(0, width - 1)
    target_y = torch.floor(
        fractions[:, 1] * (2 * (height - 1)) - 0.5 * (height - 1) + 0.5
    ).long().clamp_(0, height - 1)
    diagonal = response_bank[torch.arange(batch, device=response_bank.device),
                             torch.arange(batch, device=response_bank.device), 0]
    positive = diagonal[torch.arange(batch, device=response_bank.device), target_y, target_x]
    all_peaks = response_bank[:, :, 0].flatten(start_dim=2).max(dim=2).values
    mask = torch.eye(batch, dtype=torch.bool, device=response_bank.device)
    negatives = all_peaks.masked_fill(mask, float("-inf")).max(dim=1).values
    return torch.nn.functional.softplus(negatives - positive + float(margin)).mean()


def target_location_loss(response, torch, target_fractions):
    """Return response value at each annotated target location."""
    batch, _, height, width = response.shape
    fractions = torch.as_tensor(target_fractions, dtype=torch.float32, device=response.device)
    target_x = torch.floor(
        fractions[:, 0] * (2 * (width - 1)) - 0.5 * (width - 1) + 0.5
    ).long().clamp_(0, width - 1)
    target_y = torch.floor(
        fractions[:, 1] * (2 * (height - 1)) - 0.5 * (height - 1) + 0.5
    ).long().clamp_(0, height - 1)
    return response[torch.arange(batch, device=response.device), 0, target_y, target_x]


def appearance_augment(patches: list[np.ndarray], rng: random.Random) -> list[np.ndarray]:
    """Apply photometric-only augmentation that preserves localization labels."""
    output: list[np.ndarray] = []
    for patch in patches:
        gain = rng.uniform(0.80, 1.20)
        bias = rng.uniform(-0.06, 0.06)
        output.append(np.clip(patch.astype(np.float32) * gain + bias, 0.0, 1.0))
    return output


def spatial_negative_patches(
    patches: list[np.ndarray], rng: random.Random, min_shift: float = 0.30,
) -> list[np.ndarray]:
    """Create same-frame background negatives by translating the search crop.

    The tangent search patch is deliberately shifted by a substantial fraction
    of its width/height.  ``np.roll`` keeps the tensor shape fixed and models
    the ERP wrap-around without adding ZIP decoding or another backbone pass.
    """
    output: list[np.ndarray] = []
    for patch in patches:
        height, width = patch.shape[:2]
        min_dx = max(int(width * float(min_shift)), 1)
        min_dy = max(int(height * float(min_shift)), 1)
        dx = rng.choice([rng.randint(min_dx, max(width // 2, min_dx)),
                         -rng.randint(min_dx, max(width // 2, min_dx))])
        dy = rng.choice([rng.randint(min_dy, max(height // 2, min_dy)),
                         -rng.randint(min_dy, max(height // 2, min_dy))])
        output.append(np.roll(np.roll(patch, dx, axis=1), dy, axis=0).copy())
    return output


def _decode(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as stream:
        return np.asarray(Image.open(stream).convert("RGB"), dtype=np.float32) / np.float32(255.0)


def _box(label: dict, frame_name: str) -> np.ndarray:
    item = label[Path(frame_name).name]["bbox"]
    return np.asarray([
        float(item["cx"] - 0.5 * item["w"]),
        float(item["cy"] - 0.5 * item["h"]),
        float(item["w"]),
        float(item["h"]),
    ], dtype=np.float32)


def _target_fraction(previous: np.ndarray, current: np.ndarray, width: int, height: int) -> tuple[float, float]:
    prev = erp_bbox_to_state(previous, width, height)
    curr = erp_bbox_to_state(current, width, height)
    dx = (curr.lon - prev.lon + np.pi) % (2.0 * np.pi) - np.pi
    dy = curr.lat - prev.lat
    fov_x, fov_y = state_size_to_fov(prev, enlarge=4.0)
    return (
        float(np.clip(0.5 + dx / max(fov_x, 1e-6), 0.02, 0.98)),
        float(np.clip(0.5 - dy / max(fov_y, 1e-6), 0.02, 0.98)),
    )


def build_pairs(
    zip_root: Path,
    subset: set[str] | None,
    *,
    max_samples: int,
    stride: int,
    max_per_sequence: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[float, float]]]:
    rng = random.Random(seed)
    paths = sorted(zip_root.glob("*.zip"))
    if subset:
        paths = [path for path in paths if path.stem in subset]
    if not paths:
        raise SystemExit(f"No 360VOTS zip files found under {zip_root}")

    templates: list[np.ndarray] = []
    searches: list[np.ndarray] = []
    targets: list[tuple[float, float]] = []
    candidates: list[tuple[Path, str, str, np.ndarray, np.ndarray]] = []
    for path in paths:
        with zipfile.ZipFile(path, "r") as zf:
            prefix = path.stem + "/"
            label = json.loads(zf.read(prefix + "label.json"))
            names = sorted(
                name for name in zf.namelist()
                if name.startswith(prefix + "image/")
                and name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            )
            if len(names) < 2:
                continue
            local: list[tuple[Path, str, str, np.ndarray, np.ndarray]] = []
            step = max(int(stride), 1)
            for index in range(0, len(names) - step, step):
                previous = _box(label, names[index])
                current = _box(label, names[index + step])
                # Ignore malformed/empty annotations, but retain tiny targets:
                # their rapid scale change is one of the target failure modes.
                if previous[2] <= 1 or previous[3] <= 1 or current[2] <= 1 or current[3] <= 1:
                    continue
                local.append((path, names[index], names[index + step], previous, current))
            rng.shuffle(local)
            candidates.extend(local[:max(int(max_per_sequence), 1)])

    rng.shuffle(candidates)
    candidates = candidates[: max(int(max_samples), 1)]
    print(f"Selected temporal pairs: {len(candidates)}", flush=True)
    for index, (path, previous_name, current_name, previous, current) in enumerate(candidates, 1):
        with zipfile.ZipFile(path, "r") as zf:
            previous_frame = _decode(zf, previous_name)
            current_frame = _decode(zf, current_name)
        height, width = previous_frame.shape[:2]
        previous_state = erp_bbox_to_state(previous, width, height)
        fov_x, fov_y = state_size_to_fov(previous_state, enlarge=2.0)
        template = tangent_patch(
            previous_frame, previous_state.lon, previous_state.lat,
            fov_x, fov_y, 112, 112,
        )
        search_fov_x, search_fov_y = state_size_to_fov(previous_state, enlarge=4.0)
        search = tangent_patch(
            current_frame, previous_state.lon, previous_state.lat,
            search_fov_x, search_fov_y, 224, 224,
        )
        templates.append(template.astype(np.float16))
        searches.append(search.astype(np.float16))
        targets.append(_target_fraction(previous, current, width, height))
        if index % 100 == 0 or index == len(candidates):
            print(f"Prepared pairs: {index}/{len(candidates)}", flush=True)
    return templates, searches, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 360VOTS temporal adapter.")
    parser.add_argument("--zip-root", type=Path, default=Path(r"D:\360VOTS\360VOT-test"))
    parser.add_argument("--subset", default=None, help="Comma-separated sequence ids.")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-per-sequence", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--feature-layer", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.01)
    parser.add_argument("--appearance-augment", action="store_true")
    parser.add_argument("--spatial-negative-weight", type=float, default=0.0)
    parser.add_argument("--spatial-negative-margin", type=float, default=0.01)
    parser.add_argument("--spatial-negative-shift", type=float, default=0.30)
    args = parser.parse_args()

    import torch

    subset = {value.strip() for value in args.subset.split(",")} if args.subset else None
    templates, searches, targets = build_pairs(
        args.zip_root, subset,
        max_samples=args.max_samples,
        stride=args.stride,
        max_per_sequence=args.max_per_sequence,
        seed=args.seed,
    )
    if args.cache is not None:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.cache,
            templates=np.asarray(templates, dtype=np.float16),
            searches=np.asarray(searches, dtype=np.float16),
            targets=np.asarray(targets, dtype=np.float32),
        )
        print(f"Saved pair cache: {args.cache.resolve()}", flush=True)

    extractor = DeepFeatureExtractor(FeatureConfig(
        device=args.device,
        use_amp=args.device.startswith("cuda"),
        normalize_features=False,
        feature_layer=args.feature_layer,
        use_channels_last=args.device.startswith("cuda"),
        cudnn_benchmark=args.device.startswith("cuda"),
    ))
    for parameter in extractor.model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        example = extractor.preprocess_patch(
            np.asarray(templates[0], dtype=np.float32), 112,
            assume_normalized=True,
        )
        channels = int(extractor.model(example).shape[1])
    adapter = ResidualTrackingProjection(channels).to(extractor.device).train()
    xcorr = DepthwiseXCorrHead()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=extractor.device.type == "cuda")
    indices = list(range(len(targets)))
    for epoch in range(max(int(args.epochs), 1)):
        random.Random(args.seed + epoch).shuffle(indices)
        losses: list[float] = []
        for start in range(0, len(indices), max(int(args.batch_size), 1)):
            batch = indices[start : start + max(int(args.batch_size), 1)]
            t_np = [np.asarray(templates[i], dtype=np.float32) for i in batch]
            s_np = [np.asarray(searches[i], dtype=np.float32) for i in batch]
            if args.appearance_augment:
                rng = random.Random(args.seed + epoch * 100000 + start)
                t_np = appearance_augment(t_np, rng)
                s_np = appearance_augment(s_np, rng)
            negative_s_np = None
            if args.spatial_negative_weight > 0.0:
                rng = random.Random(args.seed + 700000 + epoch * 100000 + start)
                negative_s_np = spatial_negative_patches(
                    s_np, rng, min_shift=float(args.spatial_negative_shift),
                )
            t_tensor = extractor.preprocess_patches(t_np, 112, assume_normalized=True)
            s_tensor = extractor.preprocess_patches(s_np, 224, assume_normalized=True)
            negative_s_tensor = (
                extractor.preprocess_patches(negative_s_np, 224, assume_normalized=True)
                if negative_s_np is not None else None
            )
            # The backbone is frozen, but its outputs feed the trainable
            # residual adapter.  ``inference_mode`` marks tensors as
            # inference tensors and PyTorch refuses to save them for the
            # adapter's backward pass; ``no_grad`` keeps them ordinary
            # tensors while still avoiding backbone autograd/memory cost.
            with torch.no_grad():
                with extractor._amp_context():
                    t_feat = extractor.model(t_tensor)
                    s_feat = extractor.model(s_tensor)
                    negative_s_feat = (
                        extractor.model(negative_s_tensor)
                        if negative_s_tensor is not None else None
                    )
            with torch.autocast(
                device_type=extractor.device.type,
                dtype=torch.float16,
                enabled=extractor.device.type == "cuda",
            ):
                t_adapt = torch.nn.functional.normalize(adapter(t_feat), dim=1)
                s_adapt = torch.nn.functional.normalize(adapter(s_feat), dim=1)
                response = xcorr(t_adapt, s_adapt)
                loss = localization_loss(response, torch, [targets[i] for i in batch])
                if args.hard_negative_weight > 0.0 and len(batch) > 1:
                    response_bank = xcorr.forward_bank_to_searches(t_adapt, s_adapt)
                    loss = loss + float(args.hard_negative_weight) * hard_negative_loss(
                        response_bank, torch, [targets[i] for i in batch],
                        margin=float(args.hard_negative_margin),
                    )
                if negative_s_feat is not None and args.spatial_negative_weight > 0.0:
                    negative_adapt = torch.nn.functional.normalize(
                        adapter(negative_s_feat), dim=1,
                    )
                    negative_response = xcorr(t_adapt, negative_adapt)
                    positive_peak = target_location_loss(
                        response, torch, [targets[i] for i in batch],
                    )
                    negative_peak = target_location_loss(
                        negative_response, torch, [targets[i] for i in batch],
                    )
                    spatial_loss = torch.nn.functional.softplus(
                        negative_peak - positive_peak + float(args.spatial_negative_margin)
                    ).mean()
                    loss = loss + float(args.spatial_negative_weight) * spatial_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        print(f"epoch={epoch + 1}/{max(int(args.epochs), 1)} loss={sum(losses) / max(len(losses), 1):.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "channels": channels,
        "feature_layer": int(args.feature_layer),
        "residual": True,
        "source": "360VOTS-temporal",
        "state_dict": adapter.eval().state_dict(),
    }, args.output)
    print(f"Saved adapter: {args.output.resolve()}")


if __name__ == "__main__":
    main()
