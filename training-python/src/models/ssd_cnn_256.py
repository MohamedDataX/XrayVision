"""
=============================================================================
XRAYVISION - SSD-CNN-256 Model
=============================================================================
Modèle SSD simplifié pour détection d'objets dans les images X-ray
- Input: 1×256×256 (grayscale)
- Backbone: CNN léger (6 couches conv)
- Feature maps multi-scales
- Anchor-based detection
- Output: 17 classes + bounding boxes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_CLASSES = 18  # 17 objets + 1 background (index 0)
IMAGE_SIZE = 256

# Index 0 = Background, Index 1-17 = Objets
CLASS_NAMES = [
    "Background",  # Index 0 - classe background pour anchors négatifs
    "Gun", "Knife", "Scissors", "Bullet", "Razor_blade", "Shuriken",
    "Lighter", "Pressure_vessel", "Wrench", "Pliers", "Hammer",
    "Screwdriver", "Battery", "Bat", "Saw_blade", "Fireworks", "Dart"
]

DANGEROUS_CLASSES = {"Gun", "Knife", "Bullet", "Razor_blade"}


# ============================================================================
# SSD-CNN-256 BACKBONE
# ============================================================================
class SSDBackbone(nn.Module):
    """
    Backbone CNN léger pour extraction de features multi-scales
    Input: 1×256×256 → Feature maps à différentes résolutions
    """
    def __init__(self):
        super().__init__()
        
        # Conv Block 1: 256 -> 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 128×128
        )
        
        # Conv Block 2: 128 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64×64
        )
        
        # Conv Block 3: 64 -> 32 (Feature Map 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 32×32
        )
        
        # Conv Block 4: 32 -> 16 (Feature Map 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 16×16
        )
        
        # Conv Block 5: 16 -> 8 (Feature Map 3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 8×8
        )
        
        # Conv Block 6: 8 -> 4 (Feature Map 4)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 4×4
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Retourne les feature maps à différentes échelles
        """
        x = self.conv1(x)  # 128
        x = self.conv2(x)  # 64
        
        fm1 = self.conv3(x)   # 32×32, 128 channels
        fm2 = self.conv4(fm1) # 16×16, 256 channels
        fm3 = self.conv5(fm2) # 8×8, 512 channels
        fm4 = self.conv6(fm3) # 4×4, 512 channels
        
        return [fm1, fm2, fm3, fm4]


# ============================================================================
# SSD PREDICTION HEADS
# ============================================================================
class SSDHead(nn.Module):
    """
    Têtes de prédiction pour classification + régression bbox
    """
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        
        # Classification head: num_anchors × num_classes
        self.cls_head = nn.Conv2d(
            in_channels, 
            num_anchors * num_classes, 
            kernel_size=3, 
            padding=1
        )
        
        # Regression head: num_anchors × 4 (dx, dy, dw, dh)
        self.reg_head = nn.Conv2d(
            in_channels, 
            num_anchors * 4, 
            kernel_size=3, 
            padding=1
        )
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_pred: (batch, num_anchors * H * W, num_classes)
            reg_pred: (batch, num_anchors * H * W, 4)
        """
        batch_size = x.size(0)
        
        cls_pred = self.cls_head(x)  # (B, A*C, H, W)
        reg_pred = self.reg_head(x)  # (B, A*4, H, W)
        
        # Reshape pour faciliter le traitement
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
        cls_pred = cls_pred.view(batch_size, -1, self.num_classes)
        
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous()
        reg_pred = reg_pred.view(batch_size, -1, 4)
        
        return cls_pred, reg_pred


# ============================================================================
# SSD-CNN-256 COMPLETE MODEL
# ============================================================================
class SSDCNN256(nn.Module):
    """
    Modèle SSD-CNN-256 complet pour détection d'objets X-ray
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = SSDBackbone()
        
        # Nombre d'anchors par feature map location
        # Anchors simples: 3 par location (petit, moyen, grand)
        self.num_anchors = [3, 3, 3, 3]
        
        # Feature map channels
        self.fm_channels = [128, 256, 512, 512]
        
        # Prediction heads pour chaque feature map
        self.heads = nn.ModuleList([
            SSDHead(ch, na, num_classes)
            for ch, na in zip(self.fm_channels, self.num_anchors)
        ])
        
        # Générer les anchors de base (normalisés)
        self.register_buffer('anchors', self._generate_anchors())
        
        # Initialisation
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation des poids"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _generate_anchors(self) -> torch.Tensor:
        """
        Génère les anchors pour toutes les feature maps
        Format: (cx, cy, w, h) normalisé [0, 1]
        """
        anchors = []
        feature_sizes = [32, 16, 8, 4]  # Tailles des feature maps
        anchor_scales = [
            [0.1, 0.15, 0.2],   # FM1: petits objets
            [0.2, 0.3, 0.4],    # FM2: moyens objets
            [0.4, 0.5, 0.6],    # FM3: grands objets
            [0.6, 0.7, 0.8]     # FM4: très grands objets
        ]
        
        for fm_idx, fm_size in enumerate(feature_sizes):
            scales = anchor_scales[fm_idx]
            
            for i in range(fm_size):
                for j in range(fm_size):
                    # Centre de la cellule (normalisé)
                    cx = (j + 0.5) / fm_size
                    cy = (i + 0.5) / fm_size
                    
                    # Anchors avec différentes tailles
                    for scale in scales:
                        anchors.append([cx, cy, scale, scale])
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input image (B, 1, 256, 256)
        
        Returns:
            cls_preds: (B, num_anchors_total, num_classes)
            reg_preds: (B, num_anchors_total, 4)
        """
        # Extraire les feature maps
        feature_maps = self.backbone(x)
        
        # Prédictions de chaque head
        cls_preds = []
        reg_preds = []
        
        for fm, head in zip(feature_maps, self.heads):
            cls, reg = head(fm)
            cls_preds.append(cls)
            reg_preds.append(reg)
        
        # Concaténer toutes les prédictions
        cls_preds = torch.cat(cls_preds, dim=1)  # (B, total_anchors, C)
        reg_preds = torch.cat(reg_preds, dim=1)  # (B, total_anchors, 4)
        
        return cls_preds, reg_preds
    
    def decode_predictions(
        self, 
        cls_preds: torch.Tensor, 
        reg_preds: torch.Tensor,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ) -> List[dict]:
        """
        Décode les prédictions en détections
        
        Returns:
            Liste de dicts avec 'boxes', 'scores', 'labels' pour chaque image
        """
        batch_size = cls_preds.size(0)
        results = []
        
        # Softmax pour les scores de classe
        cls_probs = F.softmax(cls_preds, dim=-1)
        
        for b in range(batch_size):
            # Ignorer la classe 0 (Background) - prendre max sur classes 1-17
            object_probs = cls_probs[b][:, 1:]  # Exclure background
            scores, labels = object_probs.max(dim=-1)
            labels = labels + 1  # Remettre l'offset (classes 1-17)
            
            # Filtrer par confiance
            mask = scores > conf_threshold
            
            if mask.sum() == 0:
                results.append({
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long)
                })
                continue
            
            # Appliquer le masque
            filtered_scores = scores[mask]
            filtered_labels = labels[mask]
            filtered_regs = reg_preds[b][mask]
            filtered_anchors = self.anchors[mask]
            
            # Décoder les boxes (format: cx, cy, w, h → x1, y1, x2, y2)
            boxes = self._decode_boxes(filtered_regs, filtered_anchors)
            
            # NMS simple
            keep = self._nms(boxes, filtered_scores, nms_threshold)
            
            results.append({
                'boxes': boxes[keep],
                'scores': filtered_scores[keep],
                'labels': filtered_labels[keep]
            })
        
        return results
    
    def _decode_boxes(
        self, 
        reg_preds: torch.Tensor, 
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Décode les prédictions de régression en boxes
        """
        # reg_preds: (N, 4) - offsets (dx, dy, dw, dh)
        # anchors: (N, 4) - (cx, cy, w, h)
        
        cx = anchors[:, 0] + reg_preds[:, 0] * anchors[:, 2]
        cy = anchors[:, 1] + reg_preds[:, 1] * anchors[:, 3]
        w = anchors[:, 2] * torch.exp(reg_preds[:, 2].clamp(max=4))
        h = anchors[:, 3] * torch.exp(reg_preds[:, 3].clamp(max=4))
        
        # Convertir en x1, y1, x2, y2
        x1 = (cx - w / 2).clamp(0, 1)
        y1 = (cy - h / 2).clamp(0, 1)
        x2 = (cx + w / 2).clamp(0, 1)
        y2 = (cy + h / 2).clamp(0, 1)
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _nms(
        self, 
        boxes: torch.Tensor, 
        scores: torch.Tensor, 
        threshold: float
    ) -> torch.Tensor:
        """
        Non-Maximum Suppression simple
        """
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Trier par score décroissant
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            
            if order.numel() == 1:
                break
            
            # Calculer IoU avec les boxes restantes
            order = order[1:]
            ious = self._compute_iou(boxes[i].unsqueeze(0), boxes[order])
            
            # Garder les boxes avec IoU < threshold
            mask = ious.squeeze() < threshold
            order = order[mask]
        
        return torch.tensor(keep, dtype=torch.long)
    
    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'IoU entre box1 et box2
        """
        # Intersection
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter = inter_w * inter_h
        
        # Union
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1 + area2 - inter
        
        return inter / union.clamp(min=1e-6)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def count_parameters(model: nn.Module) -> int:
    """Compte le nombre de paramètres du modèle"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_classes: int = NUM_CLASSES) -> SSDCNN256:
    """Factory function pour créer le modèle"""
    model = SSDCNN256(num_classes=num_classes)
    print(f"✅ Modèle SSD-CNN-256 créé ({count_parameters(model):,} paramètres)")
    return model


if __name__ == "__main__":
    # Test du modèle
    print("🧪 Test du modèle SSD-CNN-256...")
    
    model = create_model()
    
    # Input test
    x = torch.randn(2, 1, 256, 256)
    
    # Forward pass
    cls_preds, reg_preds = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Cls predictions: {cls_preds.shape}")
    print(f"   Reg predictions: {reg_preds.shape}")
    print(f"   Total anchors: {model.anchors.shape[0]}")
    
    # Test decode
    results = model.decode_predictions(cls_preds, reg_preds, conf_threshold=0.1)
    print(f"   Détections: {[len(r['boxes']) for r in results]}")
    
    print("✅ Test réussi!")
