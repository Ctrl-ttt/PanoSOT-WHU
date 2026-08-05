"""Train a lightweight panoramic tracking adapter from AirSim360 instances.

The script freezes the ImageNet MobileNet backbone and optimizes only a 1x1
projection layer.  This fits an 8 GB GPU while making the runtime tracker
learn correspondence between two augmented views of the same AirSim object.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.airsim360 import AirSim360PairDataset, AirSim360TrainingArchive
from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.models import TrackingProjection


def _resize_patch(image: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize variable-size instance crops before forming a CUDA batch."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for AirSim360 training.") from exc
    if image.shape[:2] == (size, size):
        return image
    image_u8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    resized = Image.fromarray(image_u8, mode="RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply inexpensive RGB correspondence augmentations."""
    patch = image.astype(np.float32) / 255.0
    if rng.random() < 0.5:
        patch = patch[:, ::-1].copy()
    gain = rng.uniform(0.7, 1.3)
    bias = rng.uniform(-0.08, 0.08)
    return _resize_patch(np.clip(patch * gain + bias, 0.0, 1.0))


def _batch_tensor(extractor: DeepFeatureExtractor, patches: list[np.ndarray], size: int):
    return extractor.preprocess_patches(patches, size, assume_normalized=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AirSim360 tracker adapter.")
    parser.add_argument("--raw", type=Path, required=True, help="Path to nyc_Raw.zip")
    parser.add_argument(
        "--instance", type=Path, required=True, help="Path to nyc_instance_panorama.zip"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "airsim360_adapter.pt",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    if not args.raw.is_file() or args.raw.suffix == ".part":
        raise SystemExit(f"Raw archive is incomplete or missing: {args.raw}")
    if not args.instance.is_file():
        raise SystemExit(f"Instance archive is missing: {args.instance}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    archive = AirSim360TrainingArchive(args.raw, args.instance)
    dataset = AirSim360PairDataset(
        archive, max_samples=args.max_samples, seed=args.seed
    )
    if not dataset:
        raise SystemExit("No matched RGB/instance samples were found.")
    print(f"Training samples: {len(dataset)}")

    extractor = DeepFeatureExtractor(
        FeatureConfig(
            device=args.device,
            use_amp=args.device.startswith("cuda"),
            normalize_features=True,
        )
    )
    for parameter in extractor.model.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        example = _batch_tensor(extractor, [_augment(dataset[0][0], random)], 112)
        channels = int(extractor.model(example).shape[1])
    adapter = TrackingProjection(channels).to(extractor.device).train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=extractor.device.type == "cuda")

    indices = list(range(len(dataset)))
    for epoch in range(args.epochs):
        random.shuffle(indices)
        losses: list[float] = []
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            templates, searches = zip(*(dataset[index] for index in batch_indices))
            templates = [_augment(patch, random) for patch in templates]
            searches = [_augment(patch, random) for patch in searches]
            template_tensor = _batch_tensor(extractor, templates, 112)
            search_tensor = _batch_tensor(extractor, searches, 112)
            with torch.no_grad():
                with extractor._amp_context():
                    template_features = extractor.model(template_tensor)
                    search_features = extractor.model(search_tensor)
            with torch.autocast(
                device_type=extractor.device.type,
                dtype=torch.float16,
                enabled=extractor.device.type == "cuda",
            ):
                template_features = torch.nn.functional.normalize(
                    adapter(template_features), dim=1
                )
                search_features = torch.nn.functional.normalize(
                    adapter(search_features), dim=1
                )
                logits = template_features.mean((-1, -2)) @ search_features.mean((-1, -2)).T
                labels = torch.arange(len(batch_indices), device=extractor.device)
                loss = 0.5 * (
                    torch.nn.functional.cross_entropy(logits * 10.0, labels)
                    + torch.nn.functional.cross_entropy(logits.T * 10.0, labels)
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        print(f"epoch={epoch + 1}/{args.epochs} loss={sum(losses) / len(losses):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"channels": channels, "state_dict": adapter.eval().state_dict()}, args.output)
    print("Saved adapter:", args.output.resolve())


if __name__ == "__main__":
    main()
