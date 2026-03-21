# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Production image for the AI Bootcamp Starter
#
# Build:   docker build -t ai-bootcamp .
# Run:     docker run -p 8000:8000 --env-file .env ai-bootcamp
# ─────────────────────────────────────────────────────────────────────────────

# Use a slim Python image to keep the image size small
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching: only rebuilds if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user for security (never run as root in production)
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Expose the port FastAPI will listen on
EXPOSE 8000

# Health check: Docker will restart the container if /health fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
