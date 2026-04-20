#!/bin/bash

# Pipeline runner
# Usage: ./run_pipeline.sh [preprocess|train|api|ui|all]

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_RAW="$PROJECT_ROOT/data/RawData"
DATA_PROCESSED="$PROJECT_ROOT/data/processed"

# Check dependencies
check_java() { command -v java &> /dev/null || exit 1; }
check_sbt() { command -v sbt &> /dev/null || exit 1; }
check_python() { command -v python3 &> /dev/null || exit 1; }

# Setup python environment
setup_venv() {
    [ -d "venv" ] || python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
}

# Preprocessing
run_preprocessing() {
    echo "Running preprocessing..."
    check_java && check_sbt
    cd "$PROJECT_ROOT/preprocessing-scala"
    sbt -error clean compile "run $DATA_RAW $DATA_PROCESSED"
}

# Training
run_training() {
    echo "Running training..."
    check_python
    cd "$PROJECT_ROOT/training-python"
    setup_venv
    python src/train.py --data-root "$DATA_PROCESSED" --output-dir "./models" --epochs 15 --batch-size 16
}

# Scala API
run_api() {
    echo "Starting Scala Inference API..."
    check_java && check_sbt
    # Force jump into directory and run
    cd "$PROJECT_ROOT/inference-scala"
    sbt run
}

# Streamlit UI
run_ui() {
    echo "Starting UI..."
    check_python
    cd "$PROJECT_ROOT/ui-streamlit"
    setup_venv
    streamlit run app.py
}

# Main logic
case "$1" in
    preprocess) run_preprocessing ;;
    train)      run_training ;;
    api)        run_api ;;
    ui)         run_ui ;;
    all)
        run_preprocessing
        run_training
        ;;
    *)
        echo "Usage: $0 {preprocess|train|api|ui|all}"
        exit 1
        ;;
esac