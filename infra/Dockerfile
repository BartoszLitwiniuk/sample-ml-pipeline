# -------- Base --------
FROM python:3.12-slim AS base

# Install system dependencies required for LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# -------- Test --------
FROM base AS test

RUN uv sync --group dev

COPY src ./src
COPY tests ./tests
COPY config ./config

CMD ["pytest", "-v"]

# -------- Runtime (optional) --------
FROM base AS runtime

COPY src ./src
COPY config ./config

ENTRYPOINT ["python", "src/main.py"]
CMD ["config/base.yaml"]