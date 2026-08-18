# PanoSOT-WHU 提交镜像
# 构建（在项目根目录执行）:
#   docker build -f submission/Dockerfile -t <镜像名> .
# 评测运行: docker run --rm --gpus all -v <测试集>:/mnt/dataset:ro -v <输出>:/mnt/result <镜像>
# 评测环境断网(--network none), 镜像须自包含全部依赖与模型权重。
# Microsoft-maintained Ubuntu 22.04 base.  The pinned CUDA-enabled PyTorch
# wheel below bundles the CUDA/cuDNN runtime, so the image does not need to
# depend on Docker Hub's NVIDIA registry at build time.
FROM mcr.microsoft.com/dotnet/runtime-deps:8.0-jammy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖(利用构建缓存), 再拷代码
COPY submission/requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt \
 && pip3 uninstall -y \
        nvidia-nccl-cu12 nvidia-nvshmem-cu12 nvidia-cusparselt-cu12 \
        cuda-toolkit cuda-bindings cuda-pathfinder \
    || true
# Torch loads NCCL during import even for single-GPU inference. Keep the
# version required by the CUDA 12.9 torch wheel (2.27.3); overriding it with
# a newer wheel can make torch report an inconsistent CUDA dependency set.
RUN pip3 install --no-cache-dir nvidia-nccl-cu12==2.27.3
RUN pip3 install --no-cache-dir nvidia-cusparselt-cu12==0.7.1
RUN pip3 install --no-cache-dir nvidia-nvshmem-cu12==3.4.5

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
