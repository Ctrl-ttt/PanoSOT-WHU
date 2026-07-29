from __future__ import annotations

import numpy as np


def circular_iou_xywh(box_a: np.ndarray, box_b: np.ndarray, image_width: float) -> float:
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]

    best = 0.0
    for shift in (-image_width, 0.0, image_width):
        shifted_bx = bx + shift
        inter_w = max(0.0, min(ax + aw, shifted_bx + bw) - max(ax, shifted_bx))
        inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        inter = inter_w * inter_h
        union = aw * ah + bw * bh - inter
        if union > 0.0:
            best = max(best, inter / union)
    return best


def success_curve(
    predictions: np.ndarray,
    targets: np.ndarray,
    image_width: float,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21, dtype=np.float32)

    ious = np.array(
        [circular_iou_xywh(pred, tgt, image_width) for pred, tgt in zip(predictions, targets)],
        dtype=np.float32,
    )
    success = np.array([(ious > threshold).mean() for threshold in thresholds], dtype=np.float32)
    return thresholds, success


def otb_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    image_width: float,
) -> dict[str, float]:
    thresholds, success = success_curve(predictions, targets, image_width)
    return {
        "success_rate": float(success[np.argmin(np.abs(thresholds - 0.5))]),
        "auc": float(success.mean()),
        "mean_iou": float(
            np.mean(
                [
                    circular_iou_xywh(pred, tgt, image_width)
                    for pred, tgt in zip(predictions, targets)
                ]
            )
        ),
    }
