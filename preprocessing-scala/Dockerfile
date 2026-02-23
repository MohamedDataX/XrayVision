# =============================================================================
# Dockerfile - Microservice Preprocessing (Scala + Spark)
# =============================================================================

FROM eclipse-temurin:11-jdk as builder

# Installer sbt
RUN apt-get update && apt-get install -y curl gnupg2 && \
    echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list && \
    curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x99E82A75642AC823" | apt-key add && \
    apt-get update && apt-get install -y sbt

WORKDIR /build

# Copier les fichiers de build
COPY build.sbt .
COPY project/ project/

# Télécharger les dépendances
RUN sbt update

# Copier le code source
COPY src/ src/

# Compiler le projet
RUN sbt assembly

# =============================================================================
# Image finale
# =============================================================================
FROM eclipse-temurin:11-jre

WORKDIR /app

# Copier le JAR assemblé
COPY --from=builder /build/target/scala-2.12/*.jar /app/preprocessing.jar

# Créer les répertoires
RUN mkdir -p /app/data/raw /app/data/preprocessed /app/logs

# Variables d'environnement par défaut
ENV RAW_DATA_PATH=/app/data/raw
ENV OUTPUT_PATH=/app/data/preprocessed
ENV TRAIN_RATIO=0.7
ENV VAL_RATIO=0.15
ENV TEST_RATIO=0.15
ENV IMAGE_SIZE=256
ENV GRAYSCALE=true
ENV SPARK_DRIVER_MEMORY=4g

# Point d'entrée
ENTRYPOINT ["java", "-Xmx8g", "-jar", "/app/preprocessing.jar"]
