# XrayVision
![Scala](https://img.shields.io/badge/scala-%23E23237.svg?style=flat-square&logo=scala&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-%23E25A1C.svg?style=flat-square&logo=apachespark&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)
![ONNX](https://img.shields.io/badge/ONNX-005BE0?style=flat-square&logo=onnx&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)

Système de détection d'objets dangereux dans les images X-ray. Le projet utilise un pipeline hybride : preprocessing distribué (Spark), entraînement Deep Learning (PyTorch) et un service d'inférence haute performance (Scala + ONNX).

## Architecture

```text
XrayVision/
├── preprocessing-scala/   # Nettoyage et normalisation des données (Spark)
├── parsing-scala/         # Parsing des données brutes
├── training-python/       # Entraînement SSD-CNN-256 + export ONNX
│   └── src/
│       └── train.py
├── inference-scala/       # Service d'inférence http4s + ONNX Runtime
├── ui-streamlit/          # Interface de visualisation des détections
├── plots/                 # Scripts de visualisation des métriques
├── data/                  # Images brutes et données traitées
└── run_pipeline.sh
```

* **preprocessing-scala** : Nettoyage et normalisation des données avec Apache Spark.
* **training-python** : Entraînement du modèle SSD-CNN-256 et export vers ONNX.
* **inference-scala** : Service d'inférence performant utilisant http4s et ONNX Runtime.
* **ui-streamlit** : Interface utilisateur pour la visualisation des détections.

---

## Guide d'exécution

Le projet est piloté par un script centralisé `run_pipeline.sh`.

### Prérequis

* Java 11+ & sbt
* Python 3.9+
* Modèle ONNX exporté (généré après la phase train)

### Lancement du pipeline

```bash
chmod +x run_pipeline.sh

./run_pipeline.sh preprocess   # 1. Préparer les données (Spark)
./run_pipeline.sh train        # 2. Entraîner le modèle (PyTorch)
./run_pipeline.sh api          # 3. Lancer le service d'inférence (Scala/ONNX)
./run_pipeline.sh ui           # 4. Lancer l'interface (Streamlit)
```