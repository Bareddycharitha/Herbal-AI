# ==========================================================
# Herbal-AI Backend - Production Docker Image
# ==========================================================

FROM python:3.11-slim

# ----------------------------------------------------------
# Environment
# ----------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

# ----------------------------------------------------------
# Working directory
# ----------------------------------------------------------

WORKDIR /app

# ----------------------------------------------------------
# System dependencies
# ----------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Install Python dependencies
# ----------------------------------------------------------

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ----------------------------------------------------------
# Copy application code
# ----------------------------------------------------------

COPY ai ./ai
COPY backend ./backend

# ----------------------------------------------------------
# Create runtime directories
# ----------------------------------------------------------

RUN mkdir -p \
    /app/ai/results \
    /app/ai/logs \
    /app/ai/image_classifier/results \
    /app/ai/image_classifier/logs

# ----------------------------------------------------------
# Verify required runtime knowledge base files
# ----------------------------------------------------------

RUN test -f /app/ai/datasets/knowledge_base/herbal_knowledge_base.json && \
    test -f /app/ai/datasets/knowledge_base/disease_knowledge_base.json && \
    test -f /app/ai/datasets/knowledge_base/class_to_kb_mapping.json

# ----------------------------------------------------------
# Application configuration
# ----------------------------------------------------------

EXPOSE 8000

# ----------------------------------------------------------
# Health check
# ----------------------------------------------------------

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ----------------------------------------------------------
# Start application
# ----------------------------------------------------------

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]