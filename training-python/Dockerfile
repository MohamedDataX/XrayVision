# =============================================================================
# Dockerfile - Microservice Training (Python + PyTorch)
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/

# Créer les répertoires
RUN mkdir -p /app/data/preprocessed /app/models /app/logs

# Variables d'environnement
ENV DATA_PATH=/app/data/preprocessed
ENV MODEL_OUTPUT_PATH=/app/models
ENV LOGS_PATH=/app/logs
ENV BATCH_SIZE=32
ENV EPOCHS=100
ENV LEARNING_RATE=0.001
ENV DEVICE=cpu
ENV NUM_WORKERS=4

# Point d'entrée
ENTRYPOINT ["python", "src/train.py"]
