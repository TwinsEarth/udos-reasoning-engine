# UDOS 推演引擎推理服务镜像 (CPU)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 版本单一来源: 与 udos/__init__.py / pyproject.toml 同步
LABEL org.opencontainers.image.version="5.5.5"

# 先装 CPU 版 torch (体积更小), 再装其余依赖
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# 应用代码 (开源镜像不内置任何上游源码; CTM/D2L 适配器在显式启用时经 huggingface_hub 运行时拉取)
COPY udos ./udos
COPY demos ./demos
# 预置训练好的物理预测器, 容器开箱即带多步推演
COPY checkpoints ./checkpoints

EXPOSE 8000

# 容器级健康检查: 调 /health
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import json,urllib.request,sys; \
sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4))['status']=='ok' else 1)"

CMD ["python", "-m", "udos.server", "--host", "0.0.0.0", "--port", "8000", "--preset", "small", "--checkpoint", "checkpoints/predictor_v4.3.9.pt"]
