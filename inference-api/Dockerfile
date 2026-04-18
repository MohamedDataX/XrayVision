# Dockerfile - Microservice Inference API (Python + FastAPI)


FROM python:3.11-slim

WORKDIR /app

# dependencies système
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

#copy requirements
COPY requirements.txt .

# dependency python
RUN pip install --no-cache-dir -r requirements.txt

#code source 
COPY src/ ./src/

# SSD-CNN-256
RUN mkdir -p /app/models /app/logs

#environnement variables
ENV MODEL_PATH=/app/models/ssd_cnn_256_best.pt
ENV DEVICE=cpu
ENV CONFIDENCE_THRESHOLD=0.25
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# entrypoint
WORKDIR /app/src
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
