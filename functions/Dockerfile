# Use Python 3.10 slim image (compatible with Cloud Functions)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY inference.py .
COPY preprocessing.py .
COPY utils/ ./utils/

# Copy model files (these should be in your functions/models/ folder)
COPY models/ ./models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application with Gunicorn
# 1 worker (to avoid loading models multiple times)
# 4 threads (handle concurrent requests)
# 60s timeout (for model loading and inference)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 500 --worker-class uvicorn.workers.UvicornWorker main:app