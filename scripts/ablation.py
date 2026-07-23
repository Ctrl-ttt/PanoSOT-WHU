"""消融实验脚本：对比不同参数组合下的跟踪性能。

使用方式：
  python scripts/ablation.py --data-root data/360VOTS --output results/ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import IMAGE_SUFFIXES, load_boxes, load_image
from panosot.metrics import otb_metrics
from panosot.tracker import PanoSOTTracker, TrackerConfig


@dataclass
class AblationVariant:
    """一组消融参数变体。"""
    label: str
    scale_factors: Optional[tuple[float, ...]] = None
    template_size: Optional[int] = None
    search_enlarge: Optional[float] = None
    local_grid_radius: Optional[int] = None
    local_step_factor: Optional[float] = None
    relocalize_stride_deg: Optional[float] = None
    relocalize_topk: Optional[int] = None
    motion_momentum: Optional[float] = None

    def apply(self, base: TrackerConfig) -> TrackerConfig:
        """返回应用了此变体的新配置。"""
        from dataclasses import replace
        kwargs = {}
        for fld in ("scale_factors", "template_size", "search_enlarge",
                     "local_grid_radius", "local_step_factor",
                     "relocalize_stride_deg", "relocalize_topk",
                     "motion_momentum"):
            val = getattr(self, fld)
            if val is not None:
                kwargs[fld] = val
        return replace(base, **kwargs)


# ---------- 尺度搜索消融实验组 ----------

SCALE_EXPERIMENTS: list[AblationVariant] = [
    AblationVariant("baseline (±8%, 3步)", scale_factors=(0.93, 1.0, 1.08)),
    AblationVariant("±15%, 3步", scale_factors=(0.87, 1.0, 1.15)),
    AblationVariant("±20%, 3步", scale_factors=(0.83, 1.0, 1.20)),
    AblationVariant("±25%, 3步", scale_factors=(0.80, 1.0, 1.25)),
    AblationVariant("±30%, 3步", scale_factors=(0.77, 1.0, 1.30)),
    AblationVariant("±35%, 3步", scale_factors=(0.74, 1.0, 1.35)),
    AblationVariant("±15%, 5步", scale_factors=(0.87, 0.93, 1.0, 1.07, 1.15)),
    AblationVariant("±25%, 5步", scale_factors=(0.80, 0.89, 1.0, 1.12, 1.25)),
    AblationVariant("±35%, 5步", scale_factors=(0.74, 0.86, 1.0, 1.16, 1.35)),
    AblationVariant("±40%, 7步", scale_factors=(0.71, 0.80, 0.89, 1.0, 1.12, 1.25, 1.40)),
]

# ---------- 搜索范围消融实验组 ----------

SEARCH_EXPERIMENTS: list[AblationVariant] = [
    AblationVariant("search×2.0", search_enlarge=2.0),
    AblationVariant("search×2.5", search_enlarge=2.5),
    AblationVariant("search×3.0 (baseline)", search_enlarge=3.0),
    AblationVariant("search×3.5", search_enlarge=3.5),
    AblationVariant("search×4.0", search_enlarge=4.0),
    AblationVariant("search×5.0", search_enlarge=5.0),
]

# ---------- 模板尺寸消融实验组 ----------

TEMPLATE_EXPERIMENTS: list[AblationVariant] = [
    AblationVariant("模板32", template_size=32),
    AblationVariant("模板48 (baseline)", template_size=48),
    AblationVariant("模板64", template_size=64),
    AblationVariant("模板96", template_size=96),
    AblationVariant("模板128", template_size=128),
]

# ---------- 重定位消融实验组 ----------

RELOC_EXPERIMENTS: list[AblationVariant] = [
    AblationVariant("reloc=12°×9°", relocalize_stride_deg=12.0),
    AblationVariant("reloc=18°×12°", relocalize_stride_deg=18.0),
    AblationVariant("reloc=24°×18° (baseline)", relocalize_stride_deg=24.0),
    AblationVariant("reloc=36°×24°", relocalize_stride_deg=36.0),
    AblationVariant("reloc=48°×36°", relocalize_stride_deg=48.0),
    AblationVariant("topk=1", relocalize_topk=1),
    AblationVariant("topk=3 (baseline)", relocalize_topk=3),
    AblationVariant("topk=5", relocalize_topk=5),
    AblationVariant("topk=8", relocalize_topk=8),
]

# ---------- 综合推荐实验组 ----------

COMBO_EXPERIMENTS: list[AblationVariant] = [
    AblationVariant("宽尺度+大搜索+粗重定位",
        scale_factors=(0.74, 0.86, 1.0, 1.16, 1.35),
        search_enlarge=4.0,
        relocalize_stride_deg=18.0,
        relocalize_topk=5,
        template_size=48,
    ),
    AblationVariant("宽尺度+大搜索+细重定位",
        scale_factors=(0.74, 0.86, 1.0, 1.16, 1.35),
        search_enlarge=3.0,
        relocalize_stride_deg=12.0,
        relocalize_topk=5,
        template_size=64,
    ),
    AblationVariant("极宽尺度+大搜索+小模板",
        scale_factors=(0.71, 0.80, 0.89, 1.0, 1.12, 1.25, 1.40),
        search_enlarge=4.0,
        relocalize_stride_deg=18.0,
        relocalize_topk=5,
        template_size=48,
    ),
]

ALL_GROUPS = {
    "scale": SCALE_EXPERIMENTS,
    "search": SEARCH_EXPERIMENTS,
    "template": TEMPLATE_EXPERIMENTS,
    "reloc": RELOC_EXPERIMENTS,
    "combo": COMBO_EXPERIMENTS,
}


# ---------- 评测 ----------

@dataclass
class AblationResult:
    label: str
    num_frames: int
    elapsed_sec: float
    fps: float
    success_rate: float
    auc: float
    mean_iou: float
    error: Optional[str] = None


def run_variant(
    image_dir: Path,
    init_box_path: Path,
    gt_path: Path,
    variant: AblationVariant,
    base_config: TrackerConfig,
    max_frames: int = 0,
    *,
    deep_extractor=None,
    similarity_head=None,
) -> AblationResult:
    config = variant.apply(base_config)
    frame_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]

    init_box = load_boxes(str(init_box_path))[0]

    tracker = PanoSOTTracker(
        config=config,
        deep_extractor=deep_extractor,
        similarity_head=similarity_head,
    )
    t0 = time.perf_counter()
    try:
        predictions = tracker.track_sequence(
            (load_image(p) for p in frame_paths), init_box,
        )
    except Exception as exc:
        return AblationResult(
            variant.label, len(frame_paths), 0, 0, 0, 0, 0, error=str(exc),
        )

    elapsed = time.perf_counter() - t0
    gt_boxes = load_boxes(str(gt_path))
    n = min(len(predictions), len(gt_boxes))
    metrics = otb_metrics(predictions[:n], gt_boxes[:n],
                          image_width=float(load_image(frame_paths[0]).shape[1]))

    return AblationResult(
        label=variant.label,
        num_frames=len(frame_paths),
        elapsed_sec=elapsed,
        fps=len(frame_paths) / elapsed if elapsed > 0 else 0,
        success_rate=metrics["success_rate"],
        auc=metrics["auc"],
        mean_iou=metrics["mean_iou"],
    )


# ---------- 主入口 ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="消融实验：对比不同参数组合。",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="CSV 输出路径。")
    parser.add_argument(
        "--group",
        default="scale",
        choices=list(ALL_GROUPS.keys()) + ["all"],
        help="实验组名 (scale/search/template/reloc/combo/all)。",
    )
    parser.add_argument("--max-frames", type=int, default=0,
                        help="每序列最多处理帧数（0=全部）。")
    parser.add_argument("--deep", action="store_true", help="启用深度特征模式。")
    parser.add_argument("--backbone", default="mobilenet_v3_small")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    from scripts.batch_evaluate import discover_sequences
    sequences = discover_sequences(data_root)
    if not sequences:
        print(f"未找到有效序列 in {data_root}")
        raise SystemExit(1)

    # 收集变体
    if args.group == "all":
        variants: list[AblationVariant] = []
        for g in ("scale", "search", "template", "reloc", "combo"):
            variants.extend(ALL_GROUPS[g])
    else:
        variants = ALL_GROUPS[args.group]

    print(f"实验组: {args.group}, 共 {len(variants)} 个变体, 序列数: {len(sequences)}")

    base_config = TrackerConfig(
        use_deep_features=args.deep,
        backbone_name=args.backbone,
        device=args.device,
    )

    deep_extractor = None
    similarity_head = None
    if args.deep:
        from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
        from panosot.models import build_similarity_head
        feat_config = FeatureConfig(
            backbone_name=args.backbone,
            device=args.device,
            use_amp=args.device.startswith("cuda"),
        )
        deep_extractor = DeepFeatureExtractor(feat_config)
        similarity_head = build_similarity_head("depthwise_xcorr")
        print(f"深度特征模式: backbone={args.backbone}, device={args.device}")

    results: list[AblationResult] = []

    for seq_name, image_dir, init_box, gt in sequences:
        for idx, var in enumerate(variants, 1):
            print(f"  [{var.label}] ...", end=" ", flush=True)
            res = run_variant(
                image_dir, init_box, gt, var, base_config, args.max_frames,
                deep_extractor=deep_extractor,
                similarity_head=similarity_head,
            )
            results.append(res)
            if res.error:
                print(f"错误: {res.error}")
            else:
                print(f"{res.num_frames}f {res.elapsed_sec:.1f}s "
                      f"SR={res.success_rate:.4f} AUC={res.auc:.4f} mIoU={res.mean_iou:.4f}")

    # 输出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "num_frames", "elapsed_sec", "fps",
                          "success_rate", "auc", "mean_iou", "error"])
        for r in results:
            writer.writerow([r.label, r.num_frames, r.elapsed_sec, r.fps,
                             r.success_rate, r.auc, r.mean_iou, r.error or ""])

    # JSON
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(
        [{"label": r.label, "num_frames": r.num_frames, "elapsed_sec": r.elapsed_sec,
          "fps": r.fps, "success_rate": r.success_rate, "auc": r.auc,
          "mean_iou": r.mean_iou, "error": r.error} for r in results],
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")

    # 终端排名
    print(f"\n{'='*80}")
    print(f"排名 (按 AUC 降序):")
    print(f"{'排名':<5} {'标签':<35} {'SR@0.5':>10} {'AUC':>10} {'mIoU':>10} {'FPS':>8}")
    sorted_results = sorted(
        [r for r in results if r.error is None],
        key=lambda r: r.auc, reverse=True,
    )
    for i, r in enumerate(sorted_results, 1):
        print(f"{i:<5} {r.label:<35} {r.success_rate:>10.6f} {r.auc:>10.6f} "
              f"{r.mean_iou:>10.6f} {r.fps:>7.2f}")

    print(f"\n结果已保存至: {output_path}")
    print(f"详细结果: {json_path}")


if __name__ == "__main__":
    main()
