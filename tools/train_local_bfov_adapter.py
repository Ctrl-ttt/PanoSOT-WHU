"""Train a lightweight tracking adapter from local BFoV video sequences.

Expected data layout:

    train/train_sim/seqlist.txt
    train/train_sim/seq_0001/video.mp4
    train/train_sim/seq_0001/groundtruth.txt
    train/train_sim/seq_0001/init.txt

Each groundtruth row is ``clon,clat,fov_h,fov_v`` in degrees.  The script
builds temporal template/search pairs with the same tangent-plane geometry as
the runtime tracker and trains only the small residual 1x1 adapter.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.geometry import erp_bbox_to_state, state_size_to_fov, tangent_patch
from panosot.models import DepthwiseXCorrHead, ResidualTrackingProjection
from tools.train_360vots_adapter import (
    appearance_augment,
    hard_negative_loss,
    spatial_negative_patches,
    target_location_loss,
)
from tools.train_airsim360_adapter import localization_loss


@dataclass(frozen=True)
class LocalSequence:
    name: str
    path: Path
    video_path: Path
    groundtruth_path: Path


@dataclass
class PairSpec:
    sequence: LocalSequence
    previous_index: int
    current_index: int
    previous_bbox: np.ndarray
    current_bbox: np.ndarray
    target_fraction: tuple[float, float]
    width: int
    height: int


def read_bfov_file(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.replace(",", " ").split()
        if len(tokens) < 4:
            raise ValueError(f"Malformed BFoV row in {path}:{line_number}: {line!r}")
        rows.append([float(value) for value in tokens[:4]])
    if not rows:
        raise ValueError(f"No BFoV rows found in {path}")
    return np.asarray(rows, dtype=np.float32)


def bfov_deg_to_erp_bbox(bfov: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Convert BFoV degrees ``clon,clat,fov_h,fov_v`` to ERP ``x,y,w,h``."""
    clon, clat, fov_h, fov_v = [float(value) for value in np.asarray(bfov).reshape(4)]
    if not np.all(np.isfinite([clon, clat, fov_h, fov_v])):
        raise ValueError(f"Invalid BFoV values: {bfov!r}")
    fov_h = float(np.clip(abs(fov_h), 1e-3, 360.0))
    fov_v = float(np.clip(abs(fov_v), 1e-3, 180.0))
    clat = float(np.clip(clat, -90.0, 90.0))

    # Match the runtime submission geometry: the ERP horizontal extent grows
    # with latitude because a fixed spherical FoV spans more pixels away from
    # the equator.
    cos_lat = max(float(np.cos(np.deg2rad(clat))), 1e-6)
    width = fov_h / cos_lat / 360.0 * float(image_width)
    height = fov_v / 180.0 * float(image_height)
    cx = ((clon + 180.0) % 360.0) / 360.0 * float(image_width)
    cy = (90.0 - clat) / 180.0 * float(image_height)
    x = (cx - 0.5 * width) % float(image_width)
    y = float(np.clip(cy - 0.5 * height, 0.0, float(image_height) - height))
    return np.asarray([x, y, width, height], dtype=np.float32)


def target_fraction(
    previous: np.ndarray,
    current: np.ndarray,
    width: int,
    height: int,
    *,
    search_enlarge: float = 4.0,
) -> tuple[float, float]:
    prev = erp_bbox_to_state(previous, width, height)
    curr = erp_bbox_to_state(current, width, height)
    dx = (curr.lon - prev.lon + np.pi) % (2.0 * np.pi) - np.pi
    dy = curr.lat - prev.lat
    fov_x, fov_y = state_size_to_fov(prev, enlarge=float(search_enlarge))
    return (
        float(np.clip(0.5 + dx / max(fov_x, 1e-6), 0.02, 0.98)),
        float(np.clip(0.5 - dy / max(fov_y, 1e-6), 0.02, 0.98)),
    )


def discover_sequences(roots: list[Path], subset: set[str] | None = None) -> list[LocalSequence]:
    sequences: list[LocalSequence] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            raise SystemExit(f"Training root does not exist: {root}")
        seqlist = root / "seqlist.txt"
        names = (
            [line.strip() for line in seqlist.read_text(encoding="utf-8").splitlines() if line.strip()]
            if seqlist.is_file()
            else [path.name for path in sorted(root.iterdir()) if path.is_dir()]
        )
        for name in names:
            sequence_path = root / name
            resolved = sequence_path.resolve()
            if resolved in seen:
                continue
            display_name = f"{root.name}/{name}"
            if subset and name not in subset and display_name not in subset:
                continue
            video_path = sequence_path / "video.mp4"
            groundtruth_path = sequence_path / "groundtruth.txt"
            if not video_path.is_file() or not groundtruth_path.is_file():
                continue
            sequences.append(LocalSequence(display_name, sequence_path, video_path, groundtruth_path))
            seen.add(resolved)
    return sequences


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to read training videos.") from exc
    return cv2


def video_info(video_path: Path) -> tuple[int, int, int]:
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read first frame from: {video_path}")
            height, width = frame.shape[:2]
        return width, height, max(frame_count, 0)
    finally:
        capture.release()


def _valid_bfov(row: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(row)) and float(abs(row[2])) > 1e-3 and float(abs(row[3])) > 1e-3)


def select_pair_specs(
    roots: list[Path],
    subset: set[str] | None,
    *,
    max_samples: int,
    max_per_sequence: int,
    delta: int,
    sample_step: int,
    search_enlarge: float,
    seed: int,
) -> list[PairSpec]:
    rng = random.Random(seed)
    sequences = discover_sequences(roots, subset)
    if not sequences:
        raise SystemExit(f"No local BFoV sequences found under: {', '.join(str(root) for root in roots)}")

    candidates: list[PairSpec] = []
    for sequence in sequences:
        labels = read_bfov_file(sequence.groundtruth_path)
        width, height, frame_count = video_info(sequence.video_path)
        usable = min(len(labels), frame_count if frame_count > 0 else len(labels))
        local: list[PairSpec] = []
        for index in range(0, max(usable - int(delta), 0), max(int(sample_step), 1)):
            current_index = index + int(delta)
            previous_bfov = labels[index]
            current_bfov = labels[current_index]
            if not _valid_bfov(previous_bfov) or not _valid_bfov(current_bfov):
                continue
            previous = bfov_deg_to_erp_bbox(previous_bfov, width, height)
            current = bfov_deg_to_erp_bbox(current_bfov, width, height)
            if previous[2] <= 1 or previous[3] <= 1 or current[2] <= 1 or current[3] <= 1:
                continue
            local.append(
                PairSpec(
                    sequence=sequence,
                    previous_index=index,
                    current_index=current_index,
                    previous_bbox=previous,
                    current_bbox=current,
                    target_fraction=target_fraction(
                        previous,
                        current,
                        width,
                        height,
                        search_enlarge=search_enlarge,
                    ),
                    width=width,
                    height=height,
                )
            )
        rng.shuffle(local)
        candidates.extend(local[: max(int(max_per_sequence), 1)])
        print(f"{sequence.name}: selected {min(len(local), max(int(max_per_sequence), 1))}/{len(local)} pairs", flush=True)

    rng.shuffle(candidates)
    candidates = candidates[: max(int(max_samples), 1)]
    print(f"Selected temporal pairs: {len(candidates)}", flush=True)
    if not candidates:
        raise SystemExit("No valid BFoV temporal pairs were selected.")
    return candidates


def read_video_frames(video_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    cv2 = _require_cv2()
    unique_indices = sorted(set(int(index) for index in frame_indices))
    capture = cv2.VideoCapture(str(video_path))
    frames: dict[int, np.ndarray] = {}
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        for index in unique_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {index} from: {video_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames[index] = frame_rgb.astype(np.float32) / np.float32(255.0)
    finally:
        capture.release()
    return frames


def build_pairs(
    roots: list[Path],
    subset: set[str] | None,
    *,
    max_samples: int,
    max_per_sequence: int,
    delta: int,
    sample_step: int,
    template_enlarge: float,
    search_enlarge: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[float, float]]]:
    specs = select_pair_specs(
        roots,
        subset,
        max_samples=max_samples,
        max_per_sequence=max_per_sequence,
        delta=delta,
        sample_step=sample_step,
        search_enlarge=search_enlarge,
        seed=seed,
    )
    by_video: dict[Path, list[PairSpec]] = {}
    for spec in specs:
        by_video.setdefault(spec.sequence.video_path, []).append(spec)

    templates: list[np.ndarray] = []
    searches: list[np.ndarray] = []
    targets: list[tuple[float, float]] = []
    prepared = 0
    for video_path, group in by_video.items():
        needed = [index for spec in group for index in (spec.previous_index, spec.current_index)]
        frames = read_video_frames(video_path, needed)
        for spec in group:
            previous_frame = frames[spec.previous_index]
            current_frame = frames[spec.current_index]
            previous_state = erp_bbox_to_state(spec.previous_bbox, spec.width, spec.height)
            fov_x, fov_y = state_size_to_fov(previous_state, enlarge=float(template_enlarge))
            template = tangent_patch(
                previous_frame,
                previous_state.lon,
                previous_state.lat,
                fov_x,
                fov_y,
                112,
                112,
            )
            search_fov_x, search_fov_y = state_size_to_fov(previous_state, enlarge=float(search_enlarge))
            search = tangent_patch(
                current_frame,
                previous_state.lon,
                previous_state.lat,
                search_fov_x,
                search_fov_y,
                224,
                224,
            )
            templates.append(template.astype(np.float16))
            searches.append(search.astype(np.float16))
            targets.append(spec.target_fraction)
            prepared += 1
            if prepared % 100 == 0 or prepared == len(specs):
                print(f"Prepared pairs: {prepared}/{len(specs)}", flush=True)
    return templates, searches, targets


def load_pair_cache(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[float, float]]]:
    data = np.load(path)
    templates = [array.astype(np.float16) for array in data["templates"]]
    searches = [array.astype(np.float16) for array in data["searches"]]
    targets = [tuple(float(value) for value in row) for row in data["targets"]]
    return templates, searches, targets


def save_pair_cache(
    path: Path,
    templates: list[np.ndarray],
    searches: list[np.ndarray],
    targets: list[tuple[float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        templates=np.asarray(templates, dtype=np.float16),
        searches=np.asarray(searches, dtype=np.float16),
        targets=np.asarray(targets, dtype=np.float32),
    )
    print(f"Saved pair cache: {path.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local BFoV temporal adapter.")
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help="Training root containing seqlist.txt and sequence folders. Can be repeated.",
    )
    parser.add_argument("--subset", default=None, help="Comma-separated sequence names, e.g. seq_0001 or train_sim/seq_0001.")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-per-sequence", type=int, default=20)
    parser.add_argument("--delta", type=int, default=1, help="Frame gap between template and search annotations.")
    parser.add_argument("--sample-step", type=int, default=1, help="Stride used when enumerating temporal pairs.")
    parser.add_argument("--template-enlarge", type=float, default=4.0, help="Template tangent FoV scale; default matches batch_evaluate.")
    parser.add_argument("--search-enlarge", type=float, default=4.0, help="Search tangent FoV scale.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone", default="mobilenet_v3_small")
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=12,
        help="MobileNet feature layer. Defaults to 12 to match the submission tracker.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional torch model cache directory.")
    parser.add_argument("--cache", type=Path, default=None, help="Save prepared template/search pairs to NPZ.")
    parser.add_argument("--cache-input", type=Path, default=None, help="Load prepared pairs from NPZ and skip video decoding.")
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.01)
    parser.add_argument("--appearance-augment", action="store_true")
    parser.add_argument("--feature-reg", type=float, default=0.0, help="Feature preservation regularization weight.")
    parser.add_argument("--spatial-negative-weight", type=float, default=0.0)
    parser.add_argument("--spatial-negative-margin", type=float, default=0.01)
    parser.add_argument("--spatial-negative-shift", type=float, default=0.30)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    subset = {value.strip() for value in args.subset.split(",")} if args.subset else None
    roots = args.root or [PROJECT_ROOT / "train" / "train_sim", PROJECT_ROOT / "train" / "train_real"]
    if args.cache_input is not None:
        templates, searches, targets = load_pair_cache(args.cache_input)
        print(f"Loaded pair cache: {args.cache_input.resolve()}", flush=True)
    else:
        templates, searches, targets = build_pairs(
            roots,
            subset,
            max_samples=args.max_samples,
            max_per_sequence=args.max_per_sequence,
            delta=max(int(args.delta), 1),
            sample_step=max(int(args.sample_step), 1),
            template_enlarge=max(float(args.template_enlarge), 1e-3),
            search_enlarge=max(float(args.search_enlarge), 1e-3),
            seed=args.seed,
        )
        if args.cache is not None:
            save_pair_cache(args.cache, templates, searches, targets)
    if not templates:
        raise SystemExit("No training pairs available.")
    print(f"Training samples: {len(targets)}", flush=True)

    extractor = DeepFeatureExtractor(FeatureConfig(
        backbone_name=args.backbone,
        device=args.device,
        use_amp=args.device.startswith("cuda"),
        normalize_features=False,
        feature_layer=args.feature_layer,
        pretrained=bool(args.pretrained),
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        use_channels_last=args.device.startswith("cuda"),
        cudnn_benchmark=args.device.startswith("cuda"),
    ))
    for parameter in extractor.model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        example = extractor.preprocess_patch(
            np.asarray(templates[0], dtype=np.float32),
            112,
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
                    s_np,
                    rng,
                    min_shift=float(args.spatial_negative_shift),
                )

            t_tensor = extractor.preprocess_patches(t_np, 112, assume_normalized=True)
            s_tensor = extractor.preprocess_patches(s_np, 224, assume_normalized=True)
            negative_s_tensor = (
                extractor.preprocess_patches(negative_s_np, 224, assume_normalized=True)
                if negative_s_np is not None else None
            )
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
                t_projected = adapter(t_feat)
                s_projected = adapter(s_feat)
                t_adapt = torch.nn.functional.normalize(t_projected, dim=1)
                s_adapt = torch.nn.functional.normalize(s_projected, dim=1)
                response = xcorr(t_adapt, s_adapt)
                loss = localization_loss(response, torch, [targets[i] for i in batch])
                if args.feature_reg > 0.0:
                    loss = loss + float(args.feature_reg) * (
                        torch.nn.functional.mse_loss(t_projected, t_feat)
                        + torch.nn.functional.mse_loss(s_projected, s_feat)
                    )
                if args.hard_negative_weight > 0.0 and len(batch) > 1:
                    response_bank = xcorr.forward_bank_to_searches(t_adapt, s_adapt)
                    loss = loss + float(args.hard_negative_weight) * hard_negative_loss(
                        response_bank,
                        torch,
                        [targets[i] for i in batch],
                        margin=float(args.hard_negative_margin),
                    )
                if negative_s_feat is not None and args.spatial_negative_weight > 0.0:
                    negative_adapt = torch.nn.functional.normalize(adapter(negative_s_feat), dim=1)
                    negative_response = xcorr(t_adapt, negative_adapt)
                    positive_peak = target_location_loss(response, torch, [targets[i] for i in batch])
                    negative_peak = target_location_loss(negative_response, torch, [targets[i] for i in batch])
                    spatial_loss = torch.nn.functional.softplus(
                        negative_peak - positive_peak + float(args.spatial_negative_margin)
                    ).mean()
                    loss = loss + float(args.spatial_negative_weight) * spatial_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        print(
            f"epoch={epoch + 1}/{max(int(args.epochs), 1)} "
            f"loss={sum(losses) / max(len(losses), 1):.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "channels": channels,
            "feature_layer": int(args.feature_layer),
            "backbone": args.backbone,
            "residual": True,
            "source": "local-bfov-temporal",
            "state_dict": adapter.eval().state_dict(),
        },
        args.output,
    )
    print(f"Saved adapter: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
