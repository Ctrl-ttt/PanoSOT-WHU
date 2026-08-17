# PanoSOT-WHU 提交镜像
# 构建（在项目根目录执行）:
#   docker build -f submission/Dockerfile -t <镜像名> .
# 评测运行: docker run --rm --gpus all -v <测试集>:/mnt/dataset:ro -v <输出>:/mnt/result <镜像>
# 评测环境断网(--network none), 镜像须自包含全部依赖与模型权重。
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖(利用构建缓存), 再拷代码
COPY submission/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY panosot/ panosot/
COPY submission/weights/ weights/
COPY submission/track.py track.py

# 数据集/结果/权重目录; 权重随镜像打包, 断网可加载
ENV DATASET_DIR=/mnt/dataset \
    RESULT_DIR=/mnt/result \
    WEIGHTS_DIR=/app/weights \
    TORCH_HOME=/app/weights

# 容器启动即自动跑完全部序列, 无需额外参数
ENTRYPOINT ["python3", "track.py"]
