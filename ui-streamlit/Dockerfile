# Dockerfile - Microservice UI Streamlit

FROM python:3.11-slim

WORKDIR /app

# copy requirements
COPY requirements_microservices.txt requirements.txt

# installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

#copy the app
COPY app_microservices.py app.py

# environment variable
ENV INFERENCE_API_URL=http://inference-api:8000
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

#port expose
EXPOSE 8501

#input
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
