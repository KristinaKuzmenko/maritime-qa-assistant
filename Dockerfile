# Multi-stage build for Maritime QA Assistant
# Combines backend (FastAPI) and frontend (Streamlit) in one container
# Stages: base -> test -> production
# For CI/CD: build test stage first, then production if tests pass

# =============================================================================
# Base stage: system dependencies and Python packages
# =============================================================================
FROM python:3.10-slim as base

# Install system dependencies and update security packages
# apt-get upgrade ensures all system packages (including GnuPG) are patched
RUN apt-get -o Acquire::Retries=3 update && \
    apt-get upgrade -y && \
    apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install production requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Test stage: validate code with tests (for CI/CD pipeline)
# =============================================================================
FROM base as test

# Install test dependencies
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy application code for testing
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Set Python path
ENV PYTHONPATH=/app/frontend:/app/backend:$PYTHONPATH

# Set dummy environment variables for tests (required by Settings validation)
ENV NEO4J_PASSWORD=test_password \
    OPENAI_API_KEY=test_key \
    AUTH_SECRET_KEY=test_secret_key_minimum_32_characters_long

# Run tests (unit tests only, skip integration tests that need external services)
# Exclude evaluation tests (require ragas from requirements-eval.txt)
RUN python -m pytest backend/tests/ -v --tb=short -m "not integration" \
    --ignore=backend/tests/test_evaluation_metrics.py \
    --deselect=backend/tests/test_workflow.py::TestLLMInstance::test_groq_missing_key_raises || \
    (echo "❌ Tests failed! Build stopped." && exit 1)

# =============================================================================
# Production stage: clean image without test dependencies
# =============================================================================
FROM base as production

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create necessary directories (only temp and logs, files stored in S3)
RUN mkdir -p /app/data/temp \
    /app/logs

# Set environment variables
# Frontend first to avoid utils/ conflicts between backend and frontend
ENV PYTHONPATH=/app/frontend:/app/backend:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Expose ports
# 8000 - Backend (FastAPI)
# 8501 - Frontend (Streamlit)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Copy startup script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Default command (can be overridden)
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# =============================================================================
# CI/CD Build Pipeline:
#   1. Run tests:    docker build --target test -t maritime-qa-app:test .
#   2. Build prod:   docker build --target production -t maritime-qa-app:latest .
#   3. Push to ECR:  docker push 930953062641.dkr.ecr.us-east-1.amazonaws.com/maritime-qa-app:latest
#
# Local Development:
#   With tests:      docker build --target test .
#   Production only: docker build --target production -t maritime-qa-app:latest .
# =============================================================================
# =============================================================================
