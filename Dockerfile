# Use Python 3.11 lightweight base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY . /app/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI Server with Uvicorn
CMD ["uvicorn", "local_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
