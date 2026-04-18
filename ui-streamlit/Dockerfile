# =============================================================================
# Dockerfile - Microservice UI Streamlit
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Copier les requirements
COPY requirements_microservices.txt requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY app_microservices.py app.py

# Variables d'environnement
ENV INFERENCE_API_URL=http://inference-api:8000
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Exposer le port
EXPOSE 8501

# Point d'entrée
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
