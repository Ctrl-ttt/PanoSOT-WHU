from __future__ import annotations
import json, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.batch_evaluate_zips import decode_image, boxes_from_label
from panosot.tracker import PanoSOTTracker, TrackerConfig
from panosot.deep_features import DeepFeatureExtractor, FeatureConfig
from panosot.models import build_similarity_head

path = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
step = float(sys.argv[3]) if len(sys.argv) > 3 else None
with zipfile.ZipFile(path) as z:
    prefix = path.stem + '/'
    names = sorted(x for x in z.namelist() if x.startswith(prefix + 'image/') and x.lower().endswith(('.jpg','.jpeg','.png','.bmp')))[:limit]
    gt = boxes_from_label(json.loads(z.read(prefix + 'label.json')), names)
    ex = DeepFeatureExtractor(FeatureConfig(device='cuda', use_amp=True, cache_dir='.cache/torch'))
    cfg = TrackerConfig(use_deep_features=True, device='cuda', polar_erp_recovery_enabled=True, polar_erp_recovery_adaptive_height=True)
    if step is not None: cfg.small_target_max_scale_step = step
    tr = PanoSOTTracker(cfg, ex, build_similarity_head())
    first = decode_image(z, names[0]); tr.initialize(first, gt[0])
    print('idx gt pred psr score ncc')
    print(0, gt[0].round(1), gt[0].round(1), tr.runtime_stats.last_psr, tr.runtime_stats.last_score, tr._ncc_last_score)
    for i, name in enumerate(names[1:], 1):
        pred = tr.track(decode_image(z, name))
        print(i, gt[i].round(1), pred.round(1), round(tr.runtime_stats.last_psr, 3), round(tr.runtime_stats.last_score, 3), round(tr._ncc_last_score, 3), tr._last_flow_reliable, round(tr._last_flow_inlier_ratio,2), round(tr._last_flow_spread,2))

