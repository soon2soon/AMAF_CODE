FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglfw3 libosmesa6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e ".[gym]"
COPY configs ./configs
COPY experiments ./experiments
COPY scripts ./scripts
ENTRYPOINT ["python", "scripts/run.py"]
