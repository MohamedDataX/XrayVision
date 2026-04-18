"""
=============================================================================
XRAYVISION - DataLoader
=============================================================================
DataLoader pour images X-ray avec annotations YOLO format
- Charge images grayscale 256×256
- Charge annotations YOLO (class x_center y_center width height)
- Assigne les anchors aux ground truth boxes
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import random


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


# ============================================================================
# ANCHOR GENERATION (doit correspondre au modèle)
# ============================================================================
def generate_anchors() -> torch.Tensor:
    """
    Génère les anchors identiques à ceux du modèle
    """
    anchors = []
    feature_sizes = [32, 16, 8, 4]
    anchor_scales = [
        [0.1, 0.15, 0.2],
        [0.2, 0.3, 0.4],
        [0.4, 0.5, 0.6],
        [0.6, 0.7, 0.8]
    ]
    
    for fm_idx, fm_size in enumerate(feature_sizes):
        scales = anchor_scales[fm_idx]
        
        for i in range(fm_size):
            for j in range(fm_size):
                cx = (j + 0.5) / fm_size
                cy = (i + 0.5) / fm_size
                
                for scale in scales:
                    anchors.append([cx, cy, scale, scale])
    
    return torch.tensor(anchors, dtype=torch.float32)


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calcule l'IoU entre boxes (format: cx, cy, w, h)
    box1: (N, 4)
    box2: (M, 4)
    Returns: (N, M)
    """
    # Convertir en x1, y1, x2, y2
    box1_x1 = box1[:, 0] - box1[:, 2] / 2
    box1_y1 = box1[:, 1] - box1[:, 3] / 2
    box1_x2 = box1[:, 0] + box1[:, 2] / 2
    box1_y2 = box1[:, 1] + box1[:, 3] / 2
    
    box2_x1 = box2[:, 0] - box2[:, 2] / 2
    box2_y1 = box2[:, 1] - box2[:, 3] / 2
    box2_x2 = box2[:, 0] + box2[:, 2] / 2
    box2_y2 = box2[:, 1] + box2[:, 3] / 2
    
    # Intersection
    inter_x1 = torch.max(box1_x1.unsqueeze(1), box2_x1.unsqueeze(0))
    inter_y1 = torch.max(box1_y1.unsqueeze(1), box2_y1.unsqueeze(0))
    inter_x2 = torch.min(box1_x2.unsqueeze(1), box2_x2.unsqueeze(0))
    inter_y2 = torch.min(box1_y2.unsqueeze(1), box2_y2.unsqueeze(0))
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Areas
    area1 = box1[:, 2] * box1[:, 3]
    area2 = box2[:, 2] * box2[:, 3]
    
    # Union
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    
    return inter_area / union.clamp(min=1e-6)


def encode_boxes(
    gt_boxes: torch.Tensor,  # (M, 4) cx, cy, w, h
    anchors: torch.Tensor    # (N, 4) cx, cy, w, h
) -> torch.Tensor:
    """
    Encode les GT boxes en offsets par rapport aux anchors
    """
    # Offsets
    dx = (gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
    dy = (gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
    dw = torch.log(gt_boxes[:, 2] / anchors[:, 2])
    dh = torch.log(gt_boxes[:, 3] / anchors[:, 3])
    
    return torch.stack([dx, dy, dw, dh], dim=1)


# ============================================================================
# XRAY DATASET
# ============================================================================
class XRayDataset(Dataset):
    """
    Dataset pour images X-ray avec annotations YOLO
    """
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        anchors: Optional[torch.Tensor] = None,
        iou_threshold: float = 0.5,
        augment: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.augment = augment
        self.iou_threshold = iou_threshold
        
        # Générer les anchors si non fournis
        self.anchors = anchors if anchors is not None else generate_anchors()
        self.num_anchors = len(self.anchors)
        
        # Lister les images
        self.image_files = sorted([
            f for f in self.images_dir.glob("*.png")
        ])
        
        # Filtrer pour avoir seulement les images avec labels existants
        self.samples = []
        for img_path in self.image_files:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                self.samples.append((img_path, label_path))
            else:
                # Image sans label = pas d'objets (background)
                self.samples.append((img_path, None))
        
        print(f"   📊 Loaded {len(self.samples)} samples from {images_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, label_path = self.samples[idx]
        
        # Charger l'image
        image = self._load_image(img_path)
        
        # Charger les annotations
        gt_boxes, gt_labels = self._load_labels(label_path)
        
        # Augmentation
        if self.augment:
            image, gt_boxes = self._augment(image, gt_boxes)
        
        # Assigner les anchors aux GT boxes
        cls_targets, reg_targets, pos_mask = self._assign_anchors(gt_boxes, gt_labels)
        
        return image, cls_targets, reg_targets, pos_mask
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """Charge et prétraite une image"""
        img = Image.open(path).convert('L')  # Grayscale
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        
        # Convertir en tensor et normaliser
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
        
        return img_tensor
    
    def _load_labels(self, path: Optional[Path]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Charge les labels au format YOLO
        Format: class x_center y_center width height (normalisé 0-1)
        
        IMPORTANT: Les class_id du fichier YOLO (0-16) sont décalés de +1
        pour laisser l'index 0 au background (anchors négatifs)
        """
        if path is None or not path.exists():
            return torch.empty(0, 4), torch.empty(0, dtype=torch.long)
        
        boxes = []
        labels = []
        
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validation (class_id original 0-16, sera décalé à 1-17)
                    if 0 <= class_id < 17 and width > 0 and height > 0:
                        boxes.append([x_center, y_center, width, height])
                        # DÉCALAGE: class_id + 1 pour réserver l'index 0 au background
                        labels.append(class_id + 1)
        
        if len(boxes) == 0:
            return torch.empty(0, 4), torch.empty(0, dtype=torch.long)
        
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    
    def _augment(
        self, 
        image: torch.Tensor, 
        boxes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augmentation simple"""
        # Flip horizontal (50% chance)
        if random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip x_center
        
        # Variation de luminosité
        if random.random() > 0.5:
            factor = 0.8 + random.random() * 0.4  # 0.8 - 1.2
            image = (image * factor).clamp(0, 1)
        
        # Bruit léger
        if random.random() > 0.7:
            noise = torch.randn_like(image) * 0.02
            image = (image + noise).clamp(0, 1)
        
        return image, boxes
    
    def _assign_anchors(
        self,
        gt_boxes: torch.Tensor,  # (M, 4) cx, cy, w, h
        gt_labels: torch.Tensor  # (M,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assigne chaque anchor à un GT box (ou background)
        
        Returns:
            cls_targets: (num_anchors,) class targets
            reg_targets: (num_anchors, 4) regression targets
            pos_mask: (num_anchors,) positive anchor mask
        """
        num_anchors = self.num_anchors
        
        # Initialize targets
        cls_targets = torch.zeros(num_anchors, dtype=torch.long)
        reg_targets = torch.zeros(num_anchors, 4)
        pos_mask = torch.zeros(num_anchors, dtype=torch.bool)
        
        if len(gt_boxes) == 0:
            # Pas de GT boxes, tous les anchors sont négatifs
            return cls_targets, reg_targets, pos_mask
        
        # Calculer IoU entre anchors et GT boxes
        ious = compute_iou(self.anchors, gt_boxes)  # (num_anchors, M)
        
        # Pour chaque anchor, trouver le GT box avec max IoU
        max_ious, max_idx = ious.max(dim=1)  # (num_anchors,)
        
        # Anchors positifs: IoU > threshold
        pos_mask = max_ious > self.iou_threshold
        
        # Assigner les labels et boxes aux positifs
        cls_targets[pos_mask] = gt_labels[max_idx[pos_mask]]
        
        # Encoder les regression targets pour les positifs
        assigned_gt_boxes = gt_boxes[max_idx[pos_mask]]
        assigned_anchors = self.anchors[pos_mask]
        reg_targets[pos_mask] = encode_boxes(assigned_gt_boxes, assigned_anchors)
        
        # S'assurer qu'au moins un anchor est assigné à chaque GT
        for gt_idx in range(len(gt_boxes)):
            best_anchor_idx = ious[:, gt_idx].argmax()
            pos_mask[best_anchor_idx] = True
            cls_targets[best_anchor_idx] = gt_labels[gt_idx]
            
            gt_box = gt_boxes[gt_idx:gt_idx+1]
            anchor = self.anchors[best_anchor_idx:best_anchor_idx+1]
            reg_targets[best_anchor_idx] = encode_boxes(gt_box, anchor).squeeze(0)
        
        return cls_targets, reg_targets, pos_mask


# ============================================================================
# DATALOADER FACTORY
# ============================================================================
def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les dataloaders pour train, val, test
    
    Args:
        data_root: Chemin vers le dossier contenant images/ et labels/
        batch_size: Taille des batches
        num_workers: Nombre de workers pour le chargement
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_path = Path(data_root)
    
    # Générer les anchors une fois
    anchors = generate_anchors()
    
    print(f"📂 Loading data from {data_root}")
    
    # Train dataset
    train_ds = XRayDataset(
        images_dir=data_path / "images" / "train",
        labels_dir=data_path / "labels" / "train",
        anchors=anchors,
        augment=True
    )
    
    # Val dataset
    val_ds = XRayDataset(
        images_dir=data_path / "images" / "val",
        labels_dir=data_path / "labels" / "val",
        anchors=anchors,
        augment=False
    )
    
    # Test dataset
    test_ds = XRayDataset(
        images_dir=data_path / "images" / "test",
        labels_dir=data_path / "labels" / "test",
        anchors=anchors,
        augment=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"   ✅ Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"   ✅ Val: {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"   ✅ Test: {len(test_ds)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test du dataloader
    print("🧪 Test du DataLoader...")
    
    # Test avec des données simulées
    anchors = generate_anchors()
    print(f"   Anchors générés: {anchors.shape}")
    
    # Simuler des GT boxes
    gt_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],  # Centre, 20% width, 30% height
        [0.3, 0.7, 0.15, 0.15]
    ])
    gt_labels = torch.tensor([0, 1])  # Gun, Knife
    
    # Test IoU
    iou = compute_iou(anchors[:5], gt_boxes)
    print(f"   IoU shape: {iou.shape}")
    
    # Test encoding
    encoded = encode_boxes(gt_boxes, anchors[:2])
    print(f"   Encoded boxes: {encoded.shape}")
    
    print("✅ Test réussi!")
