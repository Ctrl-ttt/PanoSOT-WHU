"""批量评测：遍历所有序列，运行 tracker 并计算 OTB 指标，输出汇总表。

使用方式：
  python scripts/batch_evaluate.py --data-root data/360VOTS

支持深度特征模式：
  python scripts/batch_evaluate.py --data-root data/360VOTS --deep
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import IMAGE_SUFFIXES, load_boxes, load_image, save_boxes
from panosot.metrics import otb_metrics
from panosot.tracker import PanoSOTTracker, TrackerConfig

# ---------- 序列发现 ----------

IMAGE_DIR_NAMES = ("image", "images", "img", "frames")


def _find_image_dir(seq_dir: Path) -> Optional[Path]:
    """在序列目录中定位图像子目录。"""
    for name in IMAGE_DIR_NAMES:
        candidate = seq_dir / name
        if candidate.is_dir():
            return candidate
    # 没有子目录：检查序列目录本身是否包含图片
    if any(seq_dir.glob("*" + ext) for ext in IMAGE_SUFFIXES):
        return seq_dir
    return None


def _is_valid_seq_dir(path: Path) -> Optional[Path]:
    """如果 path 是一个有效序列目录，返回图像目录；否则返回 None。"""
    image_dir = _find_image_dir(path)
    if image_dir is None:
        return None
    if not (path / "init_box.txt").is_file():
        return None
    if not (path / "groundtruth.txt").is_file():
        return None
    return image_dir


def discover_sequences(data_root: Path) -> list[tuple[str, Path, Path, Path]]:
    """发现所有可评测的序列。

    返回列表，每项为 (序列名, 图像目录, init_box 文件, groundtruth 文件)。
    两种情况：
    - data_root 本身是序列（含 image/ + init_box.txt + groundtruth.txt）
    - data_root 包含多个序列子目录
    """
    # 先检查 data_root 自身是否是序列
    image_dir = _is_valid_seq_dir(data_root)
    if image_dir is not None:
        return [(data_root.name, image_dir, data_root / "init_box.txt", data_root / "groundtruth.txt")]

    # 否则扫描子目录
    sequences = []
    for subdir in sorted(data_root.iterdir()):
        if not subdir.is_dir():
            continue
        image_dir = _is_valid_seq_dir(subdir)
        if image_dir is not None:
            sequences.append((subdir.name, image_dir, subdir / "init_box.txt", subdir / "groundtruth.txt"))

    return sequences


# ---------- 单序列评测 ----------

@dataclass
class SeqResult:
    name: str
    num_frames: int
    elapsed_sec: float
    success_rate: float
    auc: float
    mean_iou: float
    error: Optional[str] = None


def evaluate_one(
    seq_name: str,
    image_dir: Path,
    init_box_path: Path,
    gt_path: Path,
    config: TrackerConfig,
    *,
    deep_extractor=None,
    similarity_head=None,
    output_pred: bool = False,
    max_frames: int = 0,
) -> SeqResult:
    """评测单条序列，返回 SeqResult。"""

    # 加载数据
    frame_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        return SeqResult(seq_name, 0, 0, 0.0, 0.0, 0.0, error="no_frames")

    try:
        init_box = load_boxes(str(init_box_path))[0]
    except Exception as exc:
        return SeqResult(seq_name, 0, 0, 0.0, 0.0, 0.0, error=f"init_box: {exc}")

    # 流式加载 + 跟踪（不一次性加载所有帧避免内存爆炸）
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
        import traceback
        traceback.print_exc()
        return SeqResult(seq_name, len(frame_paths), 0, 0.0, 0.0, 0.0, error=f"track: {exc}")
    elapsed = time.perf_counter() - t0

    # 对标 groundtruth（长度可能不同，取较短者）
    gt_boxes = load_boxes(str(gt_path))
    n = min(len(predictions), len(gt_boxes))
    pred_arr = predictions[:n]
    gt_arr = gt_boxes[:n]

    image_width = float(load_image(frame_paths[0]).shape[1])
    metrics = otb_metrics(pred_arr, gt_arr, image_width=image_width)

    # 可选：写出预测结果
    if output_pred:
        out_path = image_dir.parent / "pred.txt"
        save_boxes(out_path, pred_arr)

    return SeqResult(
        name=seq_name,
        num_frames=len(frame_paths),
        elapsed_sec=elapsed,
        success_rate=metrics["success_rate"],
        auc=metrics["auc"],
        mean_iou=metrics["mean_iou"],
    )


# ---------- 主入口 ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量评测：遍历所有序列，运行 tracker + 计算 OTB 指标。",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="数据根目录，内含多个序列子目录（每个序列含 image/、init_box.txt、groundtruth.txt）。",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="启用深度特征模式。",
    )
    parser.add_argument(
        "--backbone",
        default="mobilenet_v3_small",
        help="深度特征 backbone 名。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="计算设备 (auto / cpu / cuda)。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="汇总结果输出路径（CSV）。默认打印到 stdout。",
    )
    parser.add_argument(
        "--save-pred",
        action="store_true",
        help="是否在每条序列目录下保存 pred.txt。",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="逗号分隔的序列名列表，只评测指定序列（不指定则全部评测）。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="每序列最多处理帧数（0 表示全部）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"错误：数据根目录不存在 {data_root}", file=sys.stderr)
        raise SystemExit(1)

    sequences = discover_sequences(data_root)
    if not sequences:
        print(f"未找到任何有效序列（需要 image/ + init_box.txt + groundtruth.txt）in {data_root}")
        raise SystemExit(1)

    # 子集过滤
    if args.subset:
        subset_names = {s.strip() for s in args.subset.split(",")}
        sequences = [s for s in sequences if s[0] in subset_names]
        if not sequences:
            print(f"subset 过滤后无匹配序列")
            raise SystemExit(1)

    print(f"发现 {len(sequences)} 条序列，开始评测...\n")

    # 配置（使用调优后的参数）
    config = TrackerConfig(
        use_deep_features=args.deep,
        backbone_name=args.backbone,
        device=args.device,
        deep_template_enlarge=4.0,
        confirmation_frames=2,
        update_quality_threshold=0.65,
        # 深度模式独立参数 — 补上队友新增的参数体系
        deep_confirmation_frames=2,
        deep_update_quality_threshold=0.65,
        deep_template_update_ema=0.08,
        deep_template_update_background=0.02,
        deep_motion_momentum=0.5,
    )

    deep_extractor = None
    similarity_head = None
    if args.deep:
        from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
        from panosot.models import build_similarity_head

        resolved_device = args.device
        if resolved_device == "auto":
            import torch
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        feat_config = FeatureConfig(
            backbone_name=args.backbone,
            device=resolved_device,
            use_amp=resolved_device.startswith("cuda"),
        )
        config.device = resolved_device
        deep_extractor = DeepFeatureExtractor(feat_config)
        similarity_head = build_similarity_head("depthwise_xcorr")
        print(f"深度特征模式: backbone={args.backbone}, device={resolved_device}")

    # 逐条评测
    results: list[SeqResult] = []
    for idx, (name, image_dir, init_box, gt) in enumerate(sequences, 1):
        print(f"[{idx}/{len(sequences)}] {name} ...", end=" ", flush=True)
        res = evaluate_one(
            name, image_dir, init_box, gt, config,
            deep_extractor=deep_extractor,
            similarity_head=similarity_head,
            output_pred=args.save_pred,
            max_frames=args.max_frames,
        )
        results.append(res)

        if res.error:
            print(f"错误: {res.error}")
        else:
            print(
                f"frames={res.num_frames} "
                f"elapsed={res.elapsed_sec:.1f}s "
                f"SR={res.success_rate:.4f} "
                f"AUC={res.auc:.4f} "
                f"mIoU={res.mean_iou:.4f}"
            )

    # 汇总
    print("\n" + "=" * 70)
    valid = [r for r in results if r.error is None]
    if valid:
        avg_sr = sum(r.success_rate for r in valid) / len(valid)
        avg_auc = sum(r.auc for r in valid) / len(valid)
        avg_iou = sum(r.mean_iou for r in valid) / len(valid)
        total_time = sum(r.elapsed_sec for r in valid)
        avg_fps = sum(r.num_frames / r.elapsed_sec for r in valid) / len(valid) if valid else 0

        print(f"总序列数: {len(results)}  (成功: {len(valid)}, 失败: {len(results) - len(valid)})")
        print(f"平均 SR@0.5: {avg_sr:.6f}")
        print(f"平均 AUC:    {avg_auc:.6f}")
        print(f"平均 mIoU:   {avg_iou:.6f}")
        print(f"总耗时:       {total_time:.1f}s")
        print(f"平均 FPS:     {avg_fps:.2f}")
    else:
        print("所有序列评测失败")

    # 写出 CSV
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "num_frames", "elapsed_sec", "success_rate", "auc", "mean_iou", "error"])
            for r in results:
                writer.writerow([
                    r.name, r.num_frames, r.elapsed_sec,
                    r.success_rate, r.auc, r.mean_iou,
                    r.error or "",
                ])
        print(f"\n结果已保存至: {output_path}")

    # 同时输出 JSON
    json_path = args.output and Path(args.output).with_suffix(".json") or None
    if json_path:
        json_results = {
            "summary": {
                "total": len(results),
                "succeeded": len(valid),
                "failed": len(results) - len(valid),
                "avg_success_rate": avg_sr if valid else None,
                "avg_auc": avg_auc if valid else None,
                "avg_mean_iou": avg_iou if valid else None,
                "total_elapsed_sec": total_time,
                "avg_fps": avg_fps if valid else None,
            },
            "sequences": [
                {
                    "name": r.name,
                    "num_frames": r.num_frames,
                    "elapsed_sec": r.elapsed_sec,
                    "success_rate": r.success_rate,
                    "auc": r.auc,
                    "mean_iou": r.mean_iou,
                    "error": r.error,
                }
                for r in results
            ],
        }
        json_path.write_text(json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"详细结果已保存至: {json_path}")


if __name__ == "__main__":
    main()
