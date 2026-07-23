# PanoSOT-WHU

面向 360° 全景视频实时单目标跟踪的轻量级基线实现。

当前仓库已经补齐了赛题中“核心算法”部分，围绕以下难点做了对应设计：

1. `ERP 畸变`：不直接在等距柱状图上裁剪模板，而是在球面中心附近提取 `tangent patch`，把局部区域拉回近似透视视图，减轻高纬畸变。
2. `经线边界穿越`：内部状态使用 `(lon, lat)` 球面坐标建模，水平位置始终按 `[-pi, pi)` 环绕，不会在最左/最右边界断开。
3. `极区尺度变化`：跟踪状态保存 `equatorial_width`，回投到 ERP 时使用 `1 / cos(lat)` 做宽度补偿，让目标靠近两极时能自动变宽。
4. `长期跟踪与找回`：先做局部球面搜索；低置信度时自动切换到一次粗粒度 `360°` 重定位，再局部细化。

## 目录结构

```text
panosot/
  __init__.py
  geometry.py
  io.py
  metrics.py
  tracker.py
scripts/
  run_tracker.py
  evaluate_otb.py
requirements.txt
```

## 核心算法

### 1. 初始化

输入首帧和初始框 `x,y,w,h`：

- 把初始框中心转换到球面坐标 `(lon, lat)`
- 记录目标的 `equatorial_width` 和 `angular_height`
- 在目标中心提取一个局部切平面模板
- 用灰度 + 梯度构造轻量描述子

### 2. 局部球面搜索

对每一帧：

- 用上一帧速度预测当前球面中心
- 围绕预测位置在球面上做小网格搜索
- 对多个尺度候选提取 `tangent patch`
- 使用归一化相关性匹配模板和候选 patch

### 3. 丢失检测与全局找回

若局部最好分数较低：

- 在全景球面上做粗粒度经纬网格扫描
- 取 top-k 候选
- 对 top-k 候选再做一轮局部细化
- 用最高分结果恢复跟踪

### 4. 输出 ERP 跟踪框

最终输出 ERP 图像上的 `x,y,w,h`：

- `x` 支持跨越左右边界时自动环绕
- `w` 会根据目标所在纬度自动做极区宽度修正
- `y,h` 维持标准 ERP 像素坐标表示

## 依赖

当前基线仅依赖：

- `numpy`
- `Pillow`

安装：

```bash
pip install -r requirements.txt
```

## 运行方式

假设数据组织如下：

```text
sequence/
  00000001.jpg
  00000002.jpg
  ...
init.txt
gt.txt
```

其中 `init.txt` 第一行是首帧初始框：

```text
123.0,245.0,54.0,80.0
```

运行跟踪：

```bash
python scripts/run_tracker.py --sequence sequence --init-box init.txt --output pred.txt
```

评测 OTB 指标：

```bash
python scripts/evaluate_otb.py --pred pred.txt --gt gt.txt --first-frame sequence/00000001.jpg
```

输出示例：

```json
{
  "success_rate": 0.63,
  "auc": 0.49,
  "mean_iou": 0.46
}
```

## 后续增强建议

这个版本的目标是先把赛题关键逻辑搭完整，适合作为 Docker 提交方案的基础骨架。若要继续冲更高精度，建议下一步：

1. 用轻量骨干网络替换当前手工描述子，例如 `RepViT / MobileViT / EfficientViT`。
2. 把局部匹配替换为 `template-search` 双分支相关头。
3. 在重定位阶段增加全景候选提议器，例如稀疏多尺度扫描或小目标检测头。
4. 增加遮挡判别与模板库，减少长期更新造成的漂移。
5. 面向决赛补 `Dockerfile`、批量评测接口和 FPS profiling。
