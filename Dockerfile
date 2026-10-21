# Dockerfile for 24/7 Free Cloud Deployment on Hugging Face Spaces / Render
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies (build-essential, git, curl, ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for Hugging Face Spaces security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Copy dependency files
COPY --chown=user:user requirements-webapp.txt .

# Install Python dependencies with pre-built llama-cpp-python CPU wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-webapp.txt && \
    pip install --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python

# Copy application source code
COPY --chown=user:user . .

# Ensure generated directories exist with proper write permissions
RUN mkdir -p webapp/static/generated data outputs models && \
    chmod -R 777 webapp/static/generated data outputs models

EXPOSE 7860

# Start Uvicorn production server on port 7860
CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
