#!/bin/bash

# Script pour exécuter le pipeline complet
# Usage: ./run_pipeline.sh [preprocess|train|api|ui|all]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_RAW="$PROJECT_ROOT/data/RawData"
DATA_PROCESSED="$PROJECT_ROOT/data/processed"



# functions

check_java() {
    if ! command -v java &> /dev/null; then
        echo -e "${RED}❌ Java non trouvé. Installez Java 11+${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Java trouvé${NC}"
}

check_sbt() {
    if ! command -v sbt &> /dev/null; then
        echo -e "${RED}❌ sbt non trouvé. Installez sbt${NC}"
        echo "   brew install sbt"
        exit 1
    fi
    echo -e "${GREEN}✅ sbt trouvé${NC}"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 non trouvé${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Python3 trouvé${NC}"
}

# scala spark
run_preprocessing() {
    echo -e "\n${YELLOW}═══ PREPROCESSING (Scala + Spark) ═══${NC}\n"
    
    check_java
    check_sbt
    
    cd "$PROJECT_ROOT/preprocessing-scala"
    
    echo "📦 Compilation du projet Scala..."
    sbt clean compile
    
    echo "🚀 Exécution du preprocessing..."
    sbt "run $DATA_RAW $DATA_PROCESSED"
    
    echo -e "\n${GREEN}✅ Preprocessing terminé!${NC}"
    echo "   Output: $DATA_PROCESSED"
}

# =============================================================================
# TRAINING (Python + PyTorch)
# =============================================================================
run_training() {
    echo -e "\n${YELLOW}═══ TRAINING (Python + PyTorch) ═══${NC}\n"
    
    check_python
    
    cd "$PROJECT_ROOT/training-python"
    
    # Create venv if needed
    if [ ! -d "venv" ]; then
        echo "Création de l'environnement virtuel..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    echo "📦 Installation des dépendances..."
    pip install -q -r requirements.txt
    
    echo "lancement de l'entraînement..."
    python src/train.py \
        --data-root "$DATA_PROCESSED" \
        --output-dir "./models" \
        --epochs 15 \
        --batch-size 16
    
    deactivate
    
    echo -e "\n${GREEN} Training terminé!${NC}"
    echo "   Modèle: $PROJECT_ROOT/training-python/models/best_model.pt"
}


# inference API (FastAPI)

run_api() {
    echo -e "\n${YELLOW}═══ INFERENCE API (FastAPI) ═══${NC}\n"
    
    check_python
    
    cd "$PROJECT_ROOT/inference-api"
    
    # Create venv if needed
    if [ ! -d "venv" ]; then
        echo "📦 Création de l'environnement virtuel..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    echo "📦 Installation des dépendances..."
    pip install -q -r requirements.txt
    
    # Copy model if exists
    MODEL_SRC="$PROJECT_ROOT/training-python/models/best_model.pt"
    MODEL_DST="$PROJECT_ROOT/inference-api/models/best_model.pt"
    
    if [ -f "$MODEL_SRC" ]; then
        echo "📋 Copie du modèle..."
        mkdir -p "$(dirname "$MODEL_DST")"
        cp "$MODEL_SRC" "$MODEL_DST"
    fi
    
    echo "🚀 Démarrage de l'API..."
    echo -e "${BLUE}   URL: http://localhost:8000${NC}"
    echo -e "${BLUE}   Docs: http://localhost:8000/docs${NC}"
    
    python3 src/main.py
}

# ui streamlit
run_ui() {
    echo -e "\n${YELLOW}═══ UI (Streamlit) ═══${NC}\n"
    
    check_python
    
    cd "$PROJECT_ROOT/ui-streamlit"
    
    # Create venv if needed
    if [ ! -d "venv" ]; then
        echo "📦 Création de l'environnement virtuel..."
        python -m venv venv
    fi
    
    source venv/Scripts/activate
    
    echo "📦 Installation des dépendances..."
    pip install -q -r requirements.txt
    
    echo "🚀 Démarrage de l'UI..."
    echo -e "${BLUE}   URL: http://localhost:8501${NC}"
    
    streamlit run app.py
}

# main
case "$1" in
    preprocess)
        run_preprocessing
        ;;
    train)
        run_training
        ;;
    api)
        run_api
        ;;
    ui)
        run_ui
        ;;
    all)
        run_preprocessing
        run_training
        echo -e "\n${GREEN}✅ Pipeline complet terminé!${NC}"
        echo -e "${YELLOW}Pour démarrer les services:${NC}"
        echo "   Terminal 1: ./run_pipeline.sh api"
        echo "   Terminal 2: ./run_pipeline.sh ui"
        ;;
    *)
        echo "Usage: $0 {preprocess|train|api|ui|all}"
        echo ""
        echo "Commands:"
        echo "  preprocess  - Traite les données brutes (Scala + Spark)"
        echo "  train       - Entraîne le modèle SSD-CNN-256 (Python)"
        echo "  api         - Démarre l'API FastAPI (port 8000)"
        echo "  ui          - Démarre l'interface Streamlit (port 8501)"
        echo "  all         - Exécute preprocess + train"
        exit 1
        ;;
esac
