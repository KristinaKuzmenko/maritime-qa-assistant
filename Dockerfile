# Multi-stage build for Maritime QA Assistant
# Combines backend (FastAPI) and frontend (Streamlit) in one container
# Stages: base -> test -> production
# For CI/CD: build test stage first, then production if tests pass

# =============================================================================
# Base stage: system dependencies and Python packages
# =============================================================================
FROM python:3.10-slim-bookworm as base

# Install system dependencies and update security packages
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true && \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list 2>/dev/null || true && \
    printf 'Acquire::Retries "10";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
    > /etc/apt/apt.conf.d/80retries && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

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

# Run tests (unit tests only, skip integration/benchmark tests that need external services)
# Exclude evaluation tests (require ragas from requirements-eval.txt)
# Exclude benchmark tests (expensive, need real API calls and documents)
RUN python -m pytest backend/tests/ -v --tb=short -m "not integration and not benchmark" \
    --ignore=backend/tests/test_evaluation_metrics.py \
    --deselect=backend/tests/test_workflow.py::TestLLMInstance::test_groq_missing_key_raises || \
    (echo "❌ Tests failed! Build stopped." && exit 1)

# =============================================================================
# Benchmark stage: for performance testing in staging (not for production!)
# =============================================================================
FROM base as benchmark

# Install test dependencies (includes pytest-benchmark)
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY pytest.ini .

# Set Python path
ENV PYTHONPATH=/app/frontend:/app/backend:$PYTHONPATH

# Create data directory for test documents
RUN mkdir -p /app/data/test_docs

# Note: Mount real test documents at runtime:
#   docker run -v ./data/test_docs:/app/data/test_docs maritime-qa:benchmark
#   pytest backend/tests/benchmark_real.py -v --benchmark-only

# Default command: run benchmarks
CMD ["pytest", "backend/tests/benchmark_real.py", "-v", "--benchmark-only"]

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
#   1. Run tests:      docker build --target test -t maritime-qa-app:test .
#   2. Build prod:     docker build --target production -t maritime-qa-app:latest .
#   3. Run benchmarks: docker build --target benchmark -t maritime-qa-app:benchmark .
#   3. Push to ECR:  docker push 930953062641.dkr.ecr.us-east-1.amazonaws.com/maritime-qa-app:latest
#
# Local Development:
#   With tests:      docker build --target test .
#   Production only: docker build --target production -t maritime-qa-app:latest .
# =============================================================================
# =============================================================================
