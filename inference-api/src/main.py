"""
=============================================================================
XRAYVISION - Inference API
=============================================================================
API FastAPI pour détection d'objets dans les images X-ray
- POST /predict : retourne les détections avec bboxes
- GET /health : vérification du service
"""

import io
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================
# Chemin absolu basé sur la racine du projet
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Priorité 1: Modèle fine-tuné (meilleur)
_FINETUNED_MODEL = _PROJECT_ROOT / "training-python" / "models_fine_tuned" / "best_model.pt"
# Priorité 2: Modèle local dans inference-api/models/
_LOCAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "best_model.pt"

# Choisir le modèle fine-tuné en priorité s'il existe
if _FINETUNED_MODEL.exists():
    MODEL_PATH = _FINETUNED_MODEL
    print(f"📦 Utilisation du modèle FINE-TUNÉ: {_FINETUNED_MODEL}")
elif _LOCAL_MODEL.exists():
    MODEL_PATH = _LOCAL_MODEL
    print(f"📦 Utilisation du modèle LOCAL: {_LOCAL_MODEL}")
else:
    MODEL_PATH = _FINETUNED_MODEL  # Fallback (erreur affichée au chargement)

IMAGE_SIZE = 256
NUM_CLASSES = 18  # 17 objets + 1 background (index 0)

# Index 0 = Background, Index 1-17 = Objets
CLASS_NAMES = [
    "Background",  # Index 0 - classe background
    "Gun", "Knife", "Scissors", "Bullet", "Razor_blade", "Shuriken",
    "Lighter", "Pressure_vessel", "Wrench", "Pliers", "Hammer",
    "Screwdriver", "Battery", "Bat", "Saw_blade", "Fireworks", "Dart"
]

DANGEROUS_CLASSES = {"Gun", "Knife", "Bullet", "Razor_blade"}

# Confidence thresholds
CONF_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3
MAX_DETECTIONS = 20


# ============================================================================
# SSD-CNN-256 MODEL (copie pour inférence standalone)
# ============================================================================
class SSDBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        fm1 = self.conv3(x)
        fm2 = self.conv4(fm1)
        fm3 = self.conv5(fm2)
        fm4 = self.conv6(fm3)
        return [fm1, fm2, fm3, fm4]


class SSDHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
    
    def forward(self, x):
        batch_size = x.size(0)
        cls_pred = self.cls_head(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
        reg_pred = self.reg_head(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        return cls_pred, reg_pred


class SSDCNN256(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = SSDBackbone()
        self.num_anchors = [3, 3, 3, 3]
        self.fm_channels = [128, 256, 512, 512]
        self.heads = nn.ModuleList([
            SSDHead(ch, na, num_classes)
            for ch, na in zip(self.fm_channels, self.num_anchors)
        ])
        self.register_buffer('anchors', self._generate_anchors())
    
    def _generate_anchors(self):
        anchors = []
        feature_sizes = [32, 16, 8, 4]
        anchor_scales = [[0.1, 0.15, 0.2], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]]
        for fm_idx, fm_size in enumerate(feature_sizes):
            scales = anchor_scales[fm_idx]
            for i in range(fm_size):
                for j in range(fm_size):
                    cx, cy = (j + 0.5) / fm_size, (i + 0.5) / fm_size
                    for scale in scales:
                        anchors.append([cx, cy, scale, scale])
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, x):
        feature_maps = self.backbone(x)
        cls_preds, reg_preds = [], []
        for fm, head in zip(feature_maps, self.heads):
            cls, reg = head(fm)
            cls_preds.append(cls)
            reg_preds.append(reg)
        return torch.cat(cls_preds, dim=1), torch.cat(reg_preds, dim=1)


# ============================================================================
# INFERENCE HELPERS
# ============================================================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Prétraite l'image pour le modèle"""
    img = image.convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)


def decode_predictions(
    cls_preds: torch.Tensor,
    reg_preds: torch.Tensor,
    anchors: torch.Tensor,
    conf_threshold: float = CONF_THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD
) -> List[Dict[str, Any]]:
    """Décode les prédictions du modèle en détections"""
    cls_probs = F.softmax(cls_preds[0], dim=-1)
    
    # Ignorer la classe 0 (Background) en prenant le max sur les classes 1-17
    # On garde seulement les probabilités des classes d'objets réels
    object_probs = cls_probs[:, 1:]  # Exclure la classe background (index 0)
    scores, labels = object_probs.max(dim=-1)
    labels = labels + 1  # Remettre l'offset car on a exclu l'index 0
    
    mask = scores > conf_threshold
    if mask.sum() == 0:
        return []
    
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]
    filtered_regs = reg_preds[0][mask]
    filtered_anchors = anchors[mask]
    
    # Decode boxes
    cx = filtered_anchors[:, 0] + filtered_regs[:, 0] * filtered_anchors[:, 2]
    cy = filtered_anchors[:, 1] + filtered_regs[:, 1] * filtered_anchors[:, 3]
    w = filtered_anchors[:, 2] * torch.exp(filtered_regs[:, 2].clamp(max=4))
    h = filtered_anchors[:, 3] * torch.exp(filtered_regs[:, 3].clamp(max=4))
    
    x1 = (cx - w / 2).clamp(0, 1)
    y1 = (cy - h / 2).clamp(0, 1)
    x2 = (cx + w / 2).clamp(0, 1)
    y2 = (cy + h / 2).clamp(0, 1)
    
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    
    # Simple NMS
    keep = simple_nms(boxes, filtered_scores, nms_threshold)
    
    # Limit detections
    keep = keep[:MAX_DETECTIONS]
    
    # Build results
    results = []
    for idx in keep:
        label_id = filtered_labels[idx].item()
        class_name = CLASS_NAMES[label_id]
        box = boxes[idx].tolist()
        
        results.append({
            "class": class_name,
            "class_id": label_id,
            "confidence": round(filtered_scores[idx].item(), 3),
            "bbox": [round(x, 4) for x in box],  # [x1, y1, x2, y2] normalized
            "is_dangerous": class_name in DANGEROUS_CLASSES
        })
    
    return results


def simple_nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> List[int]:
    """Non-Maximum Suppression simple"""
    if len(boxes) == 0:
        return []
    
    _, order = scores.sort(descending=True)
    keep = []
    
    while len(order) > 0:
        i = order[0].item()
        keep.append(i)
        
        if len(order) == 1:
            break
        
        order = order[1:]
        
        # Compute IoU
        xx1 = torch.max(boxes[i, 0], boxes[order, 0])
        yy1 = torch.max(boxes[i, 1], boxes[order, 1])
        xx2 = torch.min(boxes[i, 2], boxes[order, 2])
        yy2 = torch.min(boxes[i, 3], boxes[order, 3])
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order, 2] - boxes[order, 0]) * (boxes[order, 3] - boxes[order, 1])
        union = area_i + area_j - inter
        iou = inter / union.clamp(min=1e-6)
        
        mask = iou < threshold
        order = order[mask]
    
    return keep


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="XrayVision API",
    description="API de détection d'objets dangereux dans les images X-ray",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
model = None
device = "cpu"


@app.on_event("startup")
async def load_model():
    """Charge le modèle au démarrage"""
    global model, device
    
    device = "cpu"
    print(f"🖥️ Using device: {device}")
    
    model = SSDCNN256(num_classes=NUM_CLASSES)
    
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print(f"✅ Modèle chargé: {model_path}")
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle: {e}")
    else:
        print(f"⚠️ Modèle non trouvé: {model_path}")
        print("   L'API fonctionne mais sans modèle entraîné")


@app.get("/health")
def health():
    """Vérification de santé du service"""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None,
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: float = CONF_THRESHOLD
) -> Dict[str, Any]:
    """
    Détection d'objets dans une image X-ray
    
    Returns:
        - detections: liste des objets détectés avec classe, confidence, bbox
        - has_dangerous: True si un objet dangereux détecté
        - summary: résumé des détections
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        original_size = image.size
        
        # Preprocess
        img_tensor = preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            cls_preds, reg_preds = model(img_tensor)
            detections = decode_predictions(
                cls_preds, reg_preds, model.anchors,
                conf_threshold=confidence_threshold
            )
        
        # Convert normalized bbox to pixel coordinates
        for det in detections:
            bbox_norm = det["bbox"]
            det["bbox_pixels"] = [
                int(bbox_norm[0] * original_size[0]),
                int(bbox_norm[1] * original_size[1]),
                int(bbox_norm[2] * original_size[0]),
                int(bbox_norm[3] * original_size[1])
            ]
        
        # Summary
        has_dangerous = any(d["is_dangerous"] for d in detections)
        dangerous_items = [d["class"] for d in detections if d["is_dangerous"]]
        
        return {
            "filename": file.filename,
            "image_size": list(original_size),
            "detections": detections,
            "num_detections": len(detections),
            "has_dangerous": has_dangerous,
            "dangerous_items": dangerous_items,
            "summary": f"{len(detections)} objet(s) détecté(s)" + (
                f", ALERTE: {', '.join(dangerous_items)}" if has_dangerous else ""
            )
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     XRAYVISION - INFERENCE API                                ║
    ║     FastAPI + SSD-CNN-256 Object Detection                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
