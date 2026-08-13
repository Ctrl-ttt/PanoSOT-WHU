"""P2 失败帧分析：逐帧 IoU + 发散点检测 + 目标属性分析。

输出：
- 逐帧 IoU 时序数据
- 发散帧索引（开始跟丢的位置）
- 发散前后目标属性对比（位置/尺度/速度）

使用方式：
  python tools/failure_analysis.py --seq data/0027
  python tools/failure_analysis.py --seq data/0029
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panosot.io import IMAGE_SUFFIXES, load_boxes, load_image
from panosot.metrics import circular_iou_xywh
from panosot.tracker import PanoSOTTracker, TrackerConfig
from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.models import build_similarity_head


def _state_to_xywh(tracker: PanoSOTTracker, state) -> list[float] | None:
    if state is None or tracker.frame_shape is None:
        return None
    height, width = tracker.frame_shape
    return tracker._state_to_output_bbox(state, width, height).astype(float).tolist()


def _install_candidate_probe(tracker: PanoSOTTracker, rows: list[dict]) -> None:
    """Capture deep/NCC handoff decisions without changing tracker behavior."""
    original_local_search = tracker._local_search
    original_ncc = tracker._predict_with_ncc
    original_fallback = tracker._try_handcrafted_fallback
    original_track = tracker.track

    def local_search(self, frame, predicted):
        state, score = original_local_search(frame, predicted)
        rows.append({
            "frame": self._frame_count,
            "kind": "deep_probe",
            "score": float(score),
            "bbox": _state_to_xywh(self, state),
            "psr": float(self.runtime_stats.last_psr),
        })
        return state, score

    def predict_with_ncc(self, frame, gray=None):
        state, score, reliable = original_ncc(frame, gray=gray)
        rows.append({
            "frame": self._frame_count,
            "kind": "ncc",
            "score": float(score),
            "bbox": _state_to_xywh(self, state),
            "reliable": bool(reliable),
            "ncc_bbox": self._ncc_bbox.astype(float).tolist() if self._ncc_bbox is not None else None,
        })
        return state, score, reliable

    def fallback(self, frame, predicted, gray=None):
        state, score, source = original_fallback(frame, predicted, gray=gray)
        rows.append({
            "frame": self._frame_count,
            "kind": "fallback",
            "source": source,
            "score": float(score),
            "bbox": _state_to_xywh(self, state),
        })
        return state, score, source

    def track(self, frame):
        result = original_track(frame)
        rows.append({
            "frame": self._frame_count,
            "kind": "final",
            "score": float(self.runtime_stats.last_score),
            "bbox": result.astype(float).tolist(),
            "source": "ncc" if self._last_ncc_reliable else "deep_or_motion",
            "last_psr": float(self.runtime_stats.last_psr),
            "fallback_accepts": int(self.runtime_stats.fallback_accepts),
        })
        return result

    tracker._local_search = MethodType(local_search, tracker)
    tracker._predict_with_ncc = MethodType(predict_with_ncc, tracker)
    tracker._try_handcrafted_fallback = MethodType(fallback, tracker)
    tracker.track = MethodType(track, tracker)


def _summarize_candidate_probe(rows: list[dict], gt_boxes: np.ndarray, image_width: float) -> dict:
    by_frame: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_frame.setdefault(int(row["frame"]), {})[str(row["kind"])] = row

    disagreements = []
    for frame, entries in sorted(by_frame.items()):
        deep = entries.get("deep_probe")
        ncc = entries.get("ncc")
        final = entries.get("final")
        if deep is None or ncc is None:
            continue
        deep_iou = None
        ncc_iou = None
        final_iou = None
        gt_index = min(frame, len(gt_boxes) - 1)
        if gt_index >= 0:
            gt = gt_boxes[gt_index]
            if deep.get("bbox") is not None:
                deep_iou = float(circular_iou_xywh(np.asarray(deep["bbox"]), gt, image_width))
            if ncc.get("bbox") is not None:
                ncc_iou = float(circular_iou_xywh(np.asarray(ncc["bbox"]), gt, image_width))
            if final is not None and final.get("bbox") is not None:
                final_iou = float(circular_iou_xywh(np.asarray(final["bbox"]), gt, image_width))
        if deep_iou is not None and ncc_iou is not None and abs(deep_iou - ncc_iou) >= 0.20:
            disagreements.append({
                "frame": frame,
                "deep_score": deep["score"],
                "ncc_score": ncc["score"],
                "deep_iou": deep_iou,
                "ncc_iou": ncc_iou,
                "final_iou": final_iou,
                "deep_bbox": deep["bbox"],
                "ncc_bbox": ncc["bbox"],
                "fallback": entries.get("fallback", {}).get("source"),
            })

    return {
        "rows": rows,
        "disagreement_count": len(disagreements),
        "largest_disagreements": sorted(
            disagreements,
            key=lambda item: abs(item["deep_iou"] - item["ncc_iou"]),
            reverse=True,
        )[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="失败帧分析")
    parser.add_argument("--seq", required=True, help="序列目录路径，如 data/0027")
    parser.add_argument("--output-dir", default="results", help="输出目录")
    parser.add_argument("--device", default="auto", help="device (auto / cpu / cuda)")
    parser.add_argument("--backbone", default="mobilenet_v3_small", help="deep backbone")
    parser.add_argument("--max-frames", type=int, default=0, help="maximum frames; 0 means all")
    parser.add_argument(
        "--tiny-probe-growth-guard",
        action="store_true",
        help="P4: enable the tiny_probe_growth guard for diagnosis.",
    )
    return parser.parse_args()


def segment_stats(ious, gt_area, gt_lat, gt_speed, gt_scale_change, img_area,
                  start: int, end: int, label: str) -> dict:
    seg_ious = ious[start:end]
    seg_area = gt_area[start:end]
    seg_lat = gt_lat[start:end]
    seg_speed = gt_speed[start:end]
    seg_scale = gt_scale_change[start:end]
    return {
        "label": label,
        "frames": f"{start}-{end}",
        "avg_iou": float(np.mean(seg_ious)),
        "max_iou": float(np.max(seg_ious)),
        "avg_area_px": float(np.mean(seg_area)),
        "avg_area_pct": float(np.mean(seg_area) / img_area * 100),
        "lat_range_deg": f"{float(np.min(seg_lat)):.1f} ~ {float(np.max(seg_lat)):.1f}",
        "is_polar": bool(np.any(np.abs(seg_lat) > 60)),
        "avg_speed_px": float(np.mean(seg_speed)),
        "avg_scale_change": float(np.mean(seg_scale)),
    }


def analyze_one(
    seq_dir: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    backbone: str = "mobilenet_v3_small",
    max_frames: int = 0,
    tiny_probe_growth_guard: bool = False,
) -> dict:
    seq_name = seq_dir.name
    image_dir = seq_dir / "image"
    init_box_path = seq_dir / "init_box.txt"
    if not init_box_path.is_file():
        init_box_path = seq_dir / "init.txt"
    gt_path = seq_dir / "groundtruth.txt"
    if not gt_path.is_file():
        gt_path = seq_dir / "gt.txt"

    frame_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]
    init_box = load_boxes(str(init_box_path))[0]
    gt_boxes = load_boxes(str(gt_path))

    first_img = load_image(frame_paths[0])
    img_w = float(first_img.shape[1])
    img_h = float(first_img.shape[0])
    img_area = img_w * img_h

    print(f"\n序列 {seq_name}: {len(frame_paths)} 帧, 图像: {img_w:.0f}x{img_h:.0f}")
    print("跟踪中...")

    cfg = TrackerConfig(
        use_deep_features=True,
        deep_template_enlarge=4.0,
        confirmation_frames=2,
        update_quality_threshold=0.65,
        tiny_probe_growth_guard_enabled=tiny_probe_growth_guard,
    )
    resolved_device = device
    if resolved_device == "auto":
        import torch
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but this interpreter has no usable CUDA.")
    fe = DeepFeatureExtractor(FeatureConfig(
        backbone_name=backbone,
        device=resolved_device,
        use_amp=resolved_device.startswith("cuda"),
        feature_layer=cfg.deep_feature_layer,
        normalize_features=cfg.normalize_deep_features,
        template_size=cfg.deep_template_size,
        coarse_search_size=cfg.coarse_search_size,
        refine_search_size=cfg.refine_search_size,
        cache_dir=str(PROJECT_ROOT / ".cache" / "torch"),
    ))
    sh = build_similarity_head("depthwise_xcorr")
    tracker = PanoSOTTracker(config=cfg, deep_extractor=fe, similarity_head=sh)
    candidate_rows: list[dict] = []
    _install_candidate_probe(tracker, candidate_rows)

    t0 = time.perf_counter()
    predictions = tracker.track_sequence((load_image(p) for p in frame_paths), init_box)
    elapsed = time.perf_counter() - t0

    n = min(len(predictions), len(gt_boxes))
    preds = predictions[:n]
    gts = gt_boxes[:n]

    # --- 逐帧 IoU ---
    ious = np.array([circular_iou_xywh(p, g, img_w) for p, g in zip(preds, gts)], dtype=np.float32)

    # 目标属性（从 GT 计算）
    gt_cx = gts[:, 0] + 0.5 * gts[:, 2]
    gt_cy = gts[:, 1] + 0.5 * gts[:, 3]
    gt_area = gts[:, 2] * gts[:, 3]
    gt_lat = 90.0 - gt_cy / img_h * 180.0

    gt_dx = np.diff(gt_cx, prepend=gt_cx[0])
    gt_dy = np.diff(gt_cy, prepend=gt_cy[0])
    gt_speed = np.sqrt(gt_dx**2 + gt_dy**2)

    gt_darea = np.diff(gt_area, prepend=gt_area[0])
    gt_scale_change = np.abs(gt_darea) / (gt_area + 1.0)

    # --- 发散点检测 ---
    WINDOW = 10
    smooth_iou = np.convolve(ious, np.ones(WINDOW)/WINDOW, mode='same')
    peak_frame = int(np.argmax(smooth_iou[:min(30, n)]))
    low_mask = np.where(ious[peak_frame:] < 0.02)[0]
    divergence_idx = int(peak_frame + low_mask[0]) if len(low_mask) > 0 else n - 1

    # --- 分段 ---
    before = segment_stats(ious, gt_area, gt_lat, gt_speed, gt_scale_change, img_area,
                           0, max(divergence_idx, 30), "跟踪初期")
    after = segment_stats(ious, gt_area, gt_lat, gt_speed, gt_scale_change, img_area,
                          divergence_idx, n, "发散之后")
    overall = segment_stats(ious, gt_area, gt_lat, gt_speed, gt_scale_change, img_area,
                            0, n, "全序列")

    good_mask = ious > 0.1
    good_frames = np.where(good_mask)[0]
    if len(good_frames) > 0:
        best_seg = segment_stats(ious, gt_area, gt_lat, gt_speed, gt_scale_change, img_area,
                                 int(good_frames[0]), int(good_frames[-1] + 1),
                                 f"最佳区段({good_frames[0]}-{good_frames[-1]})")
    else:
        best_seg = None

    # --- 最差帧 ---
    worst_k = min(10, n)
    worst_idx = np.argsort(ious)[:worst_k]
    worst_frames = []
    for idx in sorted(worst_idx):
        worst_frames.append({
            "frame": int(idx),
            "iou": float(ious[idx]),
            "gt_area_px": float(gt_area[idx]),
            "gt_lat_deg": float(gt_lat[idx]),
            "gt_speed_px": float(gt_speed[idx]),
            "gt_scale_change": float(gt_scale_change[idx]),
            "pred_box": preds[idx].tolist(),
            "gt_box": gts[idx].tolist(),
        })

    # --- 输出 ---
    segments = [overall, before, after]
    if best_seg:
        segments.append(best_seg)

    report = {
        "sequence": seq_name,
        "total_frames": n,
        "elapsed_sec": round(elapsed, 1),
        "fps": round(n / elapsed, 2),
        "divergence_frame": int(divergence_idx),
        "overall_metrics": {
            "success_rate": float((ious > 0.5).mean()),
            "auc": float(np.mean([(ious > t).mean() for t in np.linspace(0, 1, 21)])),
            "mean_iou": float(np.mean(ious)),
            "median_iou": float(np.median(ious)),
            "pct_iou_gt_0p1": float((ious > 0.1).mean() * 100),
            "pct_iou_gt_0p3": float((ious > 0.3).mean() * 100),
        },
        "iou_by_bin": {
            "0.00-0.05": int(np.sum(ious < 0.05)),
            "0.05-0.10": int(np.sum((ious >= 0.05) & (ious < 0.10))),
            "0.10-0.30": int(np.sum((ious >= 0.10) & (ious < 0.30))),
            "0.30-0.50": int(np.sum((ious >= 0.30) & (ious < 0.50))),
            "0.50+": int(np.sum(ious >= 0.50)),
        },
        "segments": segments,
        "worst_frames": worst_frames,
        "runtime_stats": tracker.get_runtime_stats(),
        "candidate_probe": _summarize_candidate_probe(candidate_rows, gts, img_w),
    }

    # 打印报告
    print(f"\n{'=' * 65}")
    print(f"  失败帧分析报告 — {seq_name}")
    print(f"{'=' * 65}")
    print(f"\n总帧数: {n}  总耗时: {elapsed:.0f}s  FPS: {n/elapsed:.2f}")
    print(f"发散帧: #{divergence_idx}  (跟踪能力开始崩溃的帧)")
    print(f"\n整体指标:")
    print(f"  Success Rate@0.5: {report['overall_metrics']['success_rate']:.4f}")
    print(f"  AUC:              {report['overall_metrics']['auc']:.4f}")
    print(f"  Mean IoU:         {report['overall_metrics']['mean_iou']:.4f}")
    print(f"  Median IoU:       {report['overall_metrics']['median_iou']:.4f}")

    print(f"\nIoU 分布:")
    for bin_label, count in report["iou_by_bin"].items():
        bar = "█" * int(count / n * 50)
        print(f"  {bin_label}: {count:4d} 帧 ({count/n*100:5.1f}%) {bar}")

    print(f"\n分段分析:")
    print(f"  {'区段':<25} {'帧数':>8} {'Avg IoU':>8} {'面积%':>7} {'速度':>7} {'极区':>5}")
    print(f"  {'-'*60}")
    for seg in report["segments"]:
        polar_flag = "是" if seg["is_polar"] else "否"
        print(f"  {seg['label']:<25} {seg['frames']:>8} {seg['avg_iou']:>8.4f} {seg['avg_area_pct']:>6.2f}% {seg['avg_speed_px']:>6.1f}  {polar_flag:>5}")

    print(f"\n最差 5 帧:")
    print(f"  {'帧号':>6} {'IoU':>8} {'目标纬度':>8} {'面积(px)':>10} {'移动速度':>8} {'尺度变化':>8}")
    print(f"  {'-'*55}")
    for wf in worst_frames[:5]:
        print(f"  {wf['frame']:>6} {wf['iou']:>8.4f} {wf['gt_lat_deg']:>7.1f}° {wf['gt_area_px']:>9.0f} {wf['gt_speed_px']:>7.1f} {wf['gt_scale_change']:>7.3f}")

    # 保存 JSON
    out_path = output_dir / f"failure_analysis_{seq_name}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n完整报告已保存至 {out_path}")
    print(f"{'=' * 65}")

    return report


def main() -> None:
    args = parse_args()
    seq_dir = Path(args.seq)
    output_dir = Path(args.output_dir)

    if not seq_dir.is_dir():
        print(f"错误：序列目录不存在 {seq_dir}")
        raise SystemExit(1)

    analyze_one(
        seq_dir,
        output_dir,
        device=args.device,
        backbone=args.backbone,
        max_frames=args.max_frames,
        tiny_probe_growth_guard=args.tiny_probe_growth_guard,
    )


if __name__ == "__main__":
    main()
