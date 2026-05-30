# ── Stage 1: Install Python dependencies ─────────────────
FROM python:3.11-slim AS base

# System deps for numpy, scipy, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt requirements_phase5.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements_phase5.txt

# ── Stage 2: Application ──────────────────────────────────
FROM base AS app

WORKDIR /app

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/processed/ ./data/processed/
COPY outputs/ ./outputs/
COPY .env.example .env ./

# Ensure outputs directories exist
RUN mkdir -p outputs/reports

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "asyncio", "--log-level", "info"]
