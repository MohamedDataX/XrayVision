# XrayVision - Détection d'Objets X-ray

**Projet de détection d'objets dangereux dans les images X-ray**

Architecture microservices avec preprocessing distribué Apache Spark, optimisée pour Mac CPU.

## 🏗️ Architecture

```
XrayVision/
├── preprocessing-scala/    # Preprocessing Spark (Scala) - Parallélisé
├── training-python/        # Training SSD-CNN-256 (PyTorch)
├── inference-api/          # API FastAPI
├── ui-streamlit/           # Interface Streamlit
└── data/                   # Données
    ├── RawData/            # Images brutes + JSON
    └── processed/          # Images traitées + YOLO labels
```

## 🎯 18 Classes de Détection (17 objets + Background)

| ID | Classe | Dangereux |
|----|--------|-----------|
| 0 | Background | - |
| 1 | Gun | ✅ |
| 2 | Knife | ✅ |
| 3 | Scissors | ❌ |
| 4 | Bullet | ✅ |
| 5 | Razor_blade | ✅ |
| 6 | Shuriken | ❌ |
| 7 | Lighter | ❌ |
| 8 | Pressure_vessel | ❌ |
| 9 | Wrench | ❌ |
| 10 | Pliers | ❌ |
| 11 | Hammer | ❌ |
| 12 | Screwdriver | ❌ |
| 13 | Battery | ❌ |
| 14 | Bat | ❌ |
| 15 | Saw_blade | ❌ |
| 16 | Fireworks | ❌ |
| 17 | Dart | ❌ |

## 🚀 Démarrage Rapide

### Prérequis

- Java 11+ et sbt (pour preprocessing Scala/Spark)
- Python 3.9+
- 4GB RAM
- Apache Spark 3.5+ (inclus dans les dépendances sbt)

### Pipeline Complet

```bash
# Rendre le script exécutable
chmod +x run_pipeline.sh

# 1. Preprocessing des données
./run_pipeline.sh preprocess

# 2. Entraînement du modèle
./run_pipeline.sh train

# 3. Démarrer l'API (Terminal 1)
./run_pipeline.sh api

# 4. Démarrer l'UI (Terminal 2)
./run_pipeline.sh ui
```

### Ou tout en une fois

```bash
./run_pipeline.sh all
# Puis démarrer api et ui séparément
```

## 📁 Structure des Données

### Entrée (RawData)

```text
data/RawData/
├── Positive_Samples/
│   ├── images/
│   │   ├── P00000_OL.png    # Overlay image
│   │   └── P00000_SD.png    # Shadow image
│   └── labels/
│       └── P00000.json      # {"objects": [{"label": "Gun", "ol_bb": [x1,y1,x2,y2]}]}
└── Negative_Samples/
    └── images/
```

### Sortie (processed)

```text
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   └── P00000.txt       # class x_center y_center width height (YOLO format)
│   ├── val/
│   └── test/
└── data.yaml
```

## 🔧 Microservices

### 1. Preprocessing (Scala + Apache Spark)

- Conversion grayscale (RGB → Gray)
- Resize 256×256
- Conversion bboxes → format YOLO normalisé
- Split train/val/test (70/15/15)
- **Traitement parallélisé avec Spark RDD** (utilise tous les cœurs CPU)

```bash
cd preprocessing-scala
sbt run
```

> **Note** : Les chemins des données sont configurés dans `PreprocessingMain.scala` :
>
> - Input : `../data/RawData`
> - Output : `../data/processed`

### 2. Training (Python + PyTorch)

- Modèle: SSD-CNN-256 (~3M paramètres)
- Loss: Focal Loss + Smooth L1
- Input: 1×256×256 grayscale
- Output: 18 classes (17 objets + background) + bounding boxes

```bash
cd training-python
pip install -r requirements.txt
python src/train.py --epochs 50
```

### 3. Inference API (FastAPI)

- Endpoint: `POST /predict`
- Endpoint: `GET /health`
- Input: Image file (PNG, JPG)
- Output: JSON avec détections
- Seuils: Confidence 0.7, NMS 0.3

```bash
cd inference-api
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000
# API sur http://localhost:8000
```

**Exemple de réponse:**

```json
{
  "detections": [
    {
      "class": "Knife",
      "confidence": 0.87,
      "bbox": [0.2, 0.3, 0.5, 0.6],
      "bbox_pixels": [51, 77, 128, 153],
      "is_dangerous": true
    }
  ],
  "has_dangerous": true,
  "summary": "1 objet(s) détecté(s), ALERTE: Knife"
}
```

### 4. UI (Streamlit)

- Upload d'image
- Affichage des bounding boxes
- Alertes objets dangereux

```bash
cd ui-streamlit
pip install -r requirements.txt
streamlit run app.py
# UI sur http://localhost:8501
```

## 📊 Modèle SSD-CNN-256

### Architecture

```text
Input (1×256×256)
    │
    ▼
Backbone CNN (6 conv blocks)
    │
    ├── Feature Map 1 (32×32, 128ch) → Head 1 (3 anchors)
    ├── Feature Map 2 (16×16, 256ch) → Head 2 (3 anchors)
    ├── Feature Map 3 (8×8, 512ch)   → Head 3 (3 anchors)
    └── Feature Map 4 (4×4, 512ch)   → Head 4 (3 anchors)
    │
    ▼
Classification (18 classes) + Regression (bbox)
```

### Paramètres

- ~3M paramètres
- Anchors: 3 par location × 4 feature maps
- Total anchors: 4356
- Classes: 18 (17 objets + 1 background)

## ⚠️ Notes

- Optimisé pour Mac CPU (pas de GPU requis)
- Preprocessing parallélisé avec Apache Spark (local[*])
- Batch size: 16
- Temps d'entraînement: ~5-10 min/epoch sur CPU
- Scala 2.12.18 / Spark 3.5.0

## 📄 License

MIT License
