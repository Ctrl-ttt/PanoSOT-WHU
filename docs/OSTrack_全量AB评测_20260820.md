# OSTrack 后端全量 A/B 评测报告（与手写特征基线对比）

- 日期：2026-08-20
- 协议：train 全量 130 序列 BFoV 视频；OTB 圆形 IoU AUC；手写基线 `results/train_bfov_hand_all.json`（CPU），OSTrack `results/train_bfov_ostrack_full.json`（GPU RTX 4060 Laptop、torch 2.13.0+cu130、`--no-relocalize`、平均 FPS 14.1、总耗时 5.5h）。
- 说明：OSTrack 评测关闭 relocalize（队友实现中该路径有 batch bug 会崩，见下文遗留问题）。

## 一、最终汇总

| 子集 | n | 手写 AUC | OSTrack AUC | Δ |
|---|---|---|---|---|
| real | 47 | 0.1689 | **0.4681** | **+0.2992** |
| sim | 83 | 0.1348 | **0.4981** | **+0.3633** |
| **全部** | 130 | 0.1472 | **0.4873** | **+0.3401（3.3×）** |

- **胜 110 / 负 11 / 平 9**（|Δ|>0.01 计胜负）。
- AUC>0.5 的序列：手写 7/130 → **OSTrack 70/130**；AUC<0.02：手写 35 → **OSTrack 8**。

## 二、最大增益（top5）

| 序列 | 手写 | OSTrack | Δ |
|---|---|---|---|
| sim_0043 | 0.0023 | 0.8438 | +0.8415 |
| sim_0003 | 0.0037 | 0.8242 | +0.8205 |
| sim_0030 | 0.0124 | 0.8316 | +0.8193 |
| sim_0039 | 0.0069 | 0.8184 | +0.8115 |
| sim_0063 | 0.0214 | 0.8273 | +0.8059 |
| real_0047 | 0.0453 | 0.8367 | +0.7914 |

## 三、主要回退（bottom5）

| 序列 | 手写 | OSTrack | Δ |
|---|---|---|---|
| real_0033 | 0.4576 | 0.1429 | -0.3147 |
| real_0042 | 0.4312 | 0.2142 | -0.2170 |
| sim_0033 | 0.7247 | 0.6116 | -0.1130 |
| real_0007 | 0.1949 | 0.1326 | -0.0623 |
| sim_0026 | 0.1820 | 0.1236 | -0.0584 |

## 四、结论

1. **OSTrack-384 零微调全面碾压手写特征基线（3.3×）**，且 sim 增益（+0.363）大于 real（+0.299）——此前手写基线上「sim 远跳 = 架构上限」的判断被推翻：OSTrack 的 4×FOV 搜索窗 + 单流 ViT 外观匹配能桥接大部分 10~30° 跳与尺度膨胀。
2. **剩余 8 条失败序列（sim_0034/0056/0053/0061/0057/0031/0002/0045）** 是 50°+ 极端远跳 + 户外→室内外观剧变场景，仍需 relocalize（修 bug 后）或检测式重捕获。
3. **回退集中在 11 条**（real_0033/0042/0007、sim_0033/0026 等）：OSTrack 在这些序列上反而不如调优过的手写分支，提示**混合路由（按序列/目标特征选后端）是下一阶段最优解**——router v3 已具备手写+deep 路由框架，可扩展 ostrack 分支。

## 五、遗留问题（建议队友处理）

- **OSTrack relocalize batch bug（已修复，2026-08-20）**：`forward_tokens` 中 `torch.cat((z, x), dim=1)` 报 `Expected size 1 but got size 8`——根因是 `OSTrackNet.forward_search` 未把单份缓存模板 token 广播到批量网格搜索的 batch 上。修复：`forward_search` 内 `z_tokens.expand(x_tokens.shape[0], -1, -1)`（batch=1 模板广播到 batch=8 搜索），并对真实 batch 不匹配抛清晰错误。回归测试 `test_forward_search_broadcasts_template_to_batched_search`（批量 8 输出形状 + 与单样本逐元素一致）已加入 `tests/test_ostrack.py`，21 项全过。
- **修复验证（GPU, reloc ON）**：sim_seq_0002 由崩溃 → AUC 0.2047（无 reloc 时 0.0025）。8 条最差序列重跑：平均 AUC 0.1179 → **0.4210**；其中 sim_0034 +0.726（→0.727）、sim_0056 +0.644（→0.646）、sim_0053 +0.422（→0.423）、sim_0045 +0.253、sim_0002 +0.202、sim_0061 +0.103、sim_0057 +0.075。唯一例外 real_seq_0031 仍 0.0017（reloc 已触发但不崩溃，接受阈值/外观失配救不回）。
- **全量数字更新估计**：7 条被救回 sim 使 130 序列均值从 0.4873 升约 +0.019 → **≈0.506**（下界估计；reloc ON 也会影响其余丢帧序列，精确值需全量重跑 ~11 小时 GPU）。

## 六、复现命令

```powershell
# GPU OSTrack 全量（.venv-cuda 已备好 torch 2.13.0+cu130）
.venv-cuda\Scripts\python.exe tools\evaluate_train_ostrack.py --train-root train --splits real sim --no-relocalize --device cuda --output results\train_bfov_ostrack_full.json --csv results\train_bfov_ostrack_full.csv --resume
# A/B 对比
.venv\Scripts\python.exe tools\_compare_ostrack_hand.py
```

## 七、配置调优（2026-08-21，已接入 submission）

12 序列消融（real/sim 各 0001-0006，reloc ON 基线）+ 11 条敏感序列扫描 + real 漂移序列风险验证：

| 配置 | 12 序列 mean AUC | 结论 |
|---|---|---|
| reloc ON, win=1.0, accept=0.30（原提交） | 0.5820 | 基线 |
| + template_update_interval=200 | 0.5210 | 拒绝（模板污染，-0.061） |
| + window_influence=0.257 | **0.6316** | **采纳（+0.050 vs 基线；sim +0.128 / real -0.029，仅 real_0002 受损）** |
| + window_influence=0.5 | 0.6147 | 次优 |
| 输出框宽 ×0.9/0.85 | 0.580/… | 拒绝（序列相关，无全局收益） |
| relocalize_accept_score=0.5 | 11 条敏感序列 +0.011、零回退 | **采纳**（0.65 开始丢恢复，拒绝） |

- 风险验证（win0.257+accept0.5）：real_0007 0.133→0.410、real_0032 0.083→0.156、real_0013 不变；仅 real_0002 0.741→0.571（消融内已知）。
- **全量估算 ≈0.51**（0.4873 + 已测 22 条唯一序列 Δ 合计 +3.11 / 130 ≈ +0.024；未测 108 条假定不变——多数健康序列 reloc/win 不触发，成立；未测边界序列存在 ±0.5 不确定性）。精确值需全量 reloc+win0.257 复测（GPU ~11h）。
- submission 已更新并推送（提交 95861af）：`tracker_kwargs={"relocalize_enabled": True, "relocalize_accept_score": 0.5, "window_influence": 0.257}`。
