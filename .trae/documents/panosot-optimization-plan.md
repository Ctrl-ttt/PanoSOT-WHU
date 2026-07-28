# PanoSOT 全面性能优化实施计划

## Context

PanoSOT 是面向 360° 全景视频实时单目标跟踪的轻量级基线。当前深度模式在正常帧 ~5-8 FPS，重定位帧暴跌到 ~0.5 FPS；宏平均 AUC 约 0.25，仍有较大提升空间。本计划通过批量化、缓存、冗余消除三大手段，将 FPS 提升到 15-20+ 并增强重定位能力，助力竞赛突围。

---

## 修改文件清单

| 文件 | 修改类型 | 涉及优化项 |
|------|---------|-----------|
| `panosot/deep_features.py` | 新增方法 | P0-1, P0-2, P1-1, P2-3 |
| `panosot/geometry.py` | 新增函数 + 重构 | P0-1, P1-2 |
| `panosot/models.py` | 新增方法 | P0-1 |
| `panosot/tracker.py` | 重构核心方法 | P0-1, P0-2, P1-1, P2-2, P3-2 |
| `panosot/metrics.py` | 新增函数 + 重构 | P2-1 |
| `panosot/io.py` | 新增函数 | P3-3 |
| `panosot/factory.py` | 增强校验 | P3-1 |

---

## P0-1: 全局重定位批量化（最高优先级）

**问题**: `_global_relocalize` 对 ~120 个球面网格候选逐个调用 `_reloc_score_deep`，每次 1 次前向 → 300+ 次前向/重定位帧。

### 步骤

**1. `deep_features.py` — 新增 `extract_search_features_batch`**

```python
def extract_search_features_batch(self, patches: list[np.ndarray], refine: bool = False, chunk_size: int = 32) -> Any:
```

- 将 N 个 patch 拼成 `[N, 3, H, W]`，一次 `self.model(batch_tensor)` 前向
- `chunk_size` 分块防止 OOM（4GB GPU 上 chunk_size=32 安全）
- `forward_calls` 只 +1（而非 N）

**2. `geometry.py` — 新增 `tangent_patches_batch`**

```python
def tangent_patches_batch(frame, lons, lats, fov_x, fov_y, out_h, out_w) -> np.ndarray:  # [N, out_h, out_w, 3]
```

- 向量化计算 N 个中心点的方向向量和采样坐标
- 复用 P1-2 的 `_cached_tangent_grid` 缓存基础网格
- `bilinear_sample` 扩展支持 `[N, H, W, 3]` 输出

**3. `models.py` — 新增 `DepthwiseXCorrHead.forward_n_to_m`**

```python
def forward_n_to_m(self, template_feat, search_feat) -> Any:
    # template_feat: [K, C, h, w], search_feat: [N, C, H, W]
    # 返回: [N, 1, H', W'] — 每个 search 对最佳 template 的响应
```

实现：将 K 个 template 作为 `conv2d` 的 kernel，N 个 search 拼成大 batch，用 groups 完成批量计算。

**4. `tracker.py` — 新增 `_global_relocalize_deep_batched`**

重写深度模式的重定位分支：

```
1. 生成所有粗网格候选坐标 (N ≈ 120)
2. 对每个尺度:
   a. tangent_patches_batch → N 个 patch
   b. extract_search_features_batch → [N, C, h, w]（1次前向）
   c. forward_n_to_m 与 init_bank 批量匹配 → N 个分数（1次前向）
   d. 加距离惩罚，取 top-k
3. 仅对 top-k (3个) 候选做精搜索:
   a. 每个候选 5 个偏移 → 15 个 patch
   b. extract_search_features_batch → 1次前向
   c. 批量匹配 → 分数
4. 最佳候选做 1 次 local_search
```

**前向次数: ~3(尺度) × 2(粗+精) + 1 = ~7 次**（vs 原来 300+ 次）

---

## P0-2: 深度局部搜索前向合并

**问题**: `_local_search_deep` 对 5 个 scale 各做 1 coarse + 1 refine = 10 次前向/帧。

### 步骤

**1. 批量 coarse 特征提取**

5 个尺度的搜索中心相同 (predicted.lon, predicted.lat)，仅 FoV 不同。将 5 个不同 FoV 的 patch 拼成 batch：

```python
coarse_patches = [self._extract_search_patch(frame, predicted.lon, predicted.lat, fov_x, fov_y) for fov_x, fov_y in fovs]
coarse_feats = self.deep_extractor.extract_search_features_batch(coarse_patches, refine=False)  # 1次前向
```

**2. Top-K 尺度筛选，仅对 top-2 做 refine**

```python
# 所有 coarse 结果排序
coarse_results.sort(key=lambda x: x[0], reverse=True)
# 仅 top-2 做 refine → 2 次前向
```

**前向次数: 1(批量coarse) + 2(refine) = 3 次/帧**（vs 原来 10 次）

**3. `_fused_template_response` 优化**

当前对每个模板 bank 分别调用 `_best_bank_response`（3次 similarity_head）。优化为将 3 个 bank 的特征 concat 后一次 `forward_n_to_m`：

```python
# 3 banks × K_rotations 个特征 → [3K, C, h, w]
# search_feat [1, C, H, W] → expand
# 一次 forward_n_to_m → 取每个 bank 的最佳响应
```

---

## P1-1: 模板更新冗余计算消除

**问题**: `_update_templates` → `_find_best_template` 重新提取 search_feat（已在 local_search 中算过）。

### 步骤

**1. 缓存 search_feat**

`__init__` 新增 `self._cached_search_feat` 和 `self._cached_search_state`。

`_local_search_deep` 返回前缓存最佳 search_feat。`_find_best_template` 优先复用缓存。

**2. `_extract_template_feat_bank` 批量化**

```python
# 原来: 4-10 个旋转角逐个前向
# 改为: 批量旋转 + extract_template_features_batch → 1 次前向
```

新增 `deep_features.py` 的 `extract_template_features_batch` 方法（与 `extract_search_features_batch` 类似）。

**3. 消除 age 替换时二次 `_find_best_template`**

替换后直接取新模板索引 = `len(self._templates) - 1`，无需完整重扫。

---

## P1-2: tangent_patch 采样网格缓存

**问题**: `tangent_patch` 每次重建 meshgrid + tan 计算。

### 步骤

**1. 提取 `_compute_tangent_grid(fov_x, fov_y, out_h, out_w)` 函数**

**2. `functools.lru_cache(maxsize=16)` 缓存**

FoV 量化到 0.001 弧度精度后作为 cache key。同一帧内 5-7 种 FoV 对应 5-7 个缓存条目，命中率 ~80%。

**3. `tangent_patch` 内部重构**

```python
def tangent_patch(frame, lon, lat, fov_x, fov_y, out_h, out_w):
    delta_x, delta_y = _cached_tangent_grid(round(fov_x, 3), round(fov_y, 3), out_h, out_w)
    # 仅计算与中心点相关的 center/east/north 向量（每次不同）
    center = np.array([cos(lat)*cos(lon), ...])
    east = np.array([-sin(lon), cos(lon), 0.0])
    north = np.array([...])
    dirs = center + delta_x[..., None] * east + delta_y[..., None] * north
    # ... bilinear_sample 不变
```

---

## P2-1: metrics IoU 向量化

**1. 新增 `circular_iou_xywh_batch(boxes_a, boxes_b, image_width) -> np.ndarray`**

用 numpy 向量化替代 Python 循环，3 次 shift 计算用 `np.maximum` 取最优。

**2. `success_curve` 改用 batch 版本，并新增返回 `ious`**

```python
def success_curve(predictions, targets, image_width, thresholds=None):
    ious = circular_iou_xywh_batch(predictions, targets, image_width)
    # ... 
    return thresholds, success, ious
```

**3. `otb_metrics` 复用 `success_curve` 返回的 `ious`，不再重复计算**

---

## P2-2: _rotate_patch GPU 化

深度模式 + CUDA 下，用 `torch.nn.functional.affine_grid` + `grid_sample` 在 GPU 旋转，避免 PIL CPU 路径。

新增 `_rotate_patch_gpu` 方法，非 CUDA 时回退到原 PIL 路径。

---

## P2-3: 模型 Warmup

`DeepFeatureExtractor` 新增 `warmup()` 方法，执行 3 次 dummy 前向（template/coarse/refine 各 1 次）。在 `factory.py` 的 `build_tracker` 中自动调用。

---

## P3-1: factory 类型安全

`build_tracker` 中用 `dataclasses.fields(TrackerConfig)` 构建白名单，未知 kwargs 抛 `TypeError`。先打印 warning 一个版本后再 raise。

---

## P3-2: debug 惰性求值

`_record_debug_frame` 中在 `wants_frame` 检查之前跳过所有 metadata/f-string 构造。非录制帧开销降至两次条件判断。

---

## P3-3: 惰性序列加载

`io.py` 新增 `load_sequence_lazy` 返回生成器。`track_sequence` 已兼容 Iterable，无需修改。

---

## 实施时间线

```
Week 1:
  Day 1-2: P0-1 全局重定位批量化（deep_features + geometry + models + tracker）
  Day 3:   P1-2 tangent_patch 缓存 + P2-3 模型 warmup
  Day 4-5: P0-2 深度局部搜索合并

Week 2:
  Day 6:   P1-1 模板更新冗余消除 + P2-2 rotate GPU化
  Day 7:   P2-1 metrics 向量化 + P3-1 factory 类型安全
  Day 8:   P3-2 debug 惰性 + P3-3 惰性加载
  Day 9-10: 集成测试、性能验证、回归测试
```

---

## 验证方案

1. **单元测试**: 每个优化合入后运行 `python -m unittest discover -s tests -v`，确保 4 个现有测试通过
2. **FPS 对比**: 用 `tools/profile_tracker.py` 在 360VOTS_smoke 上对比优化前后 FPS
3. **精度守卫**: 在 360VOTS_unpacked 全序列上跑 `tools/batch_evaluate.py`，确保 AUC 退化 < 0.5%
4. **前向计数**: 检查 `tracker.get_runtime_stats()["deep_forward_calls"]` 确认下降幅度
5. **显存监控**: 批量重定位在 4GB GPU 上测试无 OOM
