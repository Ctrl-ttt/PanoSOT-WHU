"""Merge batch_evaluate_zips JSON outputs and compute weighted summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    seq = {}
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("sequences", []):
            seq[str(item["name"])] = item
    rows = [seq[k] for k in sorted(seq)]
    ok = [x for x in rows if not x.get("error")]
    total_frames = sum(int(x.get("num_frames", 0)) for x in ok)
    summary = {
        "total": len(rows),
        "succeeded": len(ok),
        "failed": len(rows) - len(ok),
        "avg_auc_sequence_macro": sum(float(x.get("auc", 0.0)) for x in ok) / max(len(ok), 1),
        "avg_mean_iou_sequence_macro": sum(float(x.get("mean_iou", 0.0)) for x in ok) / max(len(ok), 1),
        "avg_fps_sequence_macro": sum(float(x.get("fps", 0.0)) for x in ok) / max(len(ok), 1),
        "total_frames": total_frames,
    }
    args.output.write_text(json.dumps({"summary": summary, "sequences": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
