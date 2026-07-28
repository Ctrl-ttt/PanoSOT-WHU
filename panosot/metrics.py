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


def circular_iou_xywh_batch(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    image_width: float,
) -> np.ndarray:
    """批量计算 circular IoU，处理 360° wrap-around。

    Args:
        boxes_a: [N, 4] xywh 格式
        boxes_b: [N, 4] xywh 格式
        image_width: 图像宽度（用于经度环绕）

    Returns:
        [N] IoU 数组
    """
    ax = boxes_a[:, 0]
    ay = boxes_a[:, 1]
    aw = boxes_a[:, 2]
    ah = boxes_a[:, 3]
    bx = boxes_b[:, 0]
    by = boxes_b[:, 1]
    bw = boxes_b[:, 2]
    bh = boxes_b[:, 3]

    best = np.zeros(len(boxes_a), dtype=np.float32)
    for shift in (-image_width, 0.0, image_width):
        shifted_bx = bx + shift
        inter_w = np.maximum(0.0, np.minimum(ax + aw, shifted_bx + bw) - np.maximum(ax, shifted_bx))
        inter_h = np.maximum(0.0, np.minimum(ay + ah, by + bh) - np.maximum(ay, by))
        inter = inter_w * inter_h
        union = aw * ah + bw * bh - inter
        valid = union > 0.0
        iou = np.where(valid, inter / np.maximum(union, 1e-8), 0.0)
        best = np.maximum(best, iou)
    return best


def success_curve(
    predictions: np.ndarray,
    targets: np.ndarray,
    image_width: float,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21, dtype=np.float32)

    ious = circular_iou_xywh_batch(predictions, targets, image_width)
    success = np.array([(ious > threshold).mean() for threshold in thresholds], dtype=np.float32)
    return thresholds, success, ious


def otb_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    image_width: float,
) -> dict[str, float]:
    thresholds, success, ious = success_curve(predictions, targets, image_width)
    return {
        "success_rate": float(success[np.argmin(np.abs(thresholds - 0.5))]),
        "auc": float(success.mean()),
        "mean_iou": float(ious.mean()),
    }
