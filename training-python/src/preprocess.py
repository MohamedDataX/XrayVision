#!/usr/bin/env python3
"""
=============================================================================
XRAYVISION - Preprocessing Python (Alternative rapide)
=============================================================================
Preprocessing des données X-ray sans Spark
- Conversion grayscale 256×256
- Conversion JSON → YOLO format
- Split train/val/test (70/20/10)
"""

import os
import json
import random
import shutil
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_SIZE = 256

# Mapping des labels vers IDs
LABEL_TO_ID = {
    "Gun": 0,
    "Knife": 1,
    "Scissors": 2,
    "Bullet": 3,
    "Razor_blade": 4,
    "Shuriken": 5,
    "Lighter": 6,
    "Pressure_vessel": 7,
    "Wrench": 8,
    "Pliers": 9,
    "Hammer": 10,
    "Screwdriver": 11,
    "Battery": 12,
    "Bat": 13,
    "Saw_blade": 14,
    "Fireworks": 15,
    "Dart": 16
}

CLASS_NAMES = list(LABEL_TO_ID.keys())


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================
def parse_json_annotation(json_path: Path) -> List[Dict]:
    """Parse un fichier JSON d'annotation"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        objects = []
        for obj in data.get('objects', []):
            label = obj.get('label', '')
            bbox = obj.get('ol_bb', [])  # overlay bounding box
            
            if label in LABEL_TO_ID and len(bbox) == 4:
                objects.append({
                    'label': label,
                    'class_id': LABEL_TO_ID[label],
                    'bbox': bbox  # [x_min, y_min, x_max, y_max]
                })
        
        return objects
    except Exception as e:
        print(f"   ⚠️ Erreur parsing {json_path}: {e}")
        return []


def convert_bbox_to_yolo(bbox: List[int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convertit bbox [x_min, y_min, x_max, y_max] en format YOLO [x_center, y_center, width, height]
    Coordonnées normalisées (0-1)
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Calculer centre et dimensions
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Clamp to valid range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0.001, min(1, width))
    height = max(0.001, min(1, height))
    
    return x_center, y_center, width, height


def process_image(
    img_path: Path,
    json_path: Path,
    output_img_path: Path,
    output_label_path: Path
) -> bool:
    """Traite une image: resize + convert to grayscale + create YOLO label"""
    try:
        # Load and get original size
        img = Image.open(img_path)
        orig_width, orig_height = img.size
        
        # Convert to grayscale and resize
        img = img.convert('L')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        
        # Save processed image
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_img_path)
        
        # Parse annotations
        objects = parse_json_annotation(json_path) if json_path.exists() else []
        
        # Create YOLO label file
        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_label_path, 'w') as f:
            for obj in objects:
                x_c, y_c, w, h = convert_bbox_to_yolo(
                    obj['bbox'], orig_width, orig_height
                )
                # YOLO format: class x_center y_center width height
                f.write(f"{obj['class_id']} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        
        return True
    except Exception as e:
        print(f"   ⚠️ Erreur {img_path.name}: {e}")
        return False


# ============================================================================
# MAIN PREPROCESSING
# ============================================================================
def preprocess_data(
    raw_data_path: str,
    output_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    max_samples: int = None
):
    """
    Preprocessing complet des données
    """
    raw_path = Path(raw_data_path)
    out_path = Path(output_path)
    
    print("=" * 60)
    print("🔄 XRAYVISION - Preprocessing")
    print("=" * 60)
    
    # Collect all samples
    positive_path = raw_path / "Positive_Samples"
    negative_path = raw_path / "Negative_Samples"
    
    samples = []
    
    # Positive samples (with annotations)
    if positive_path.exists():
        images_dir = positive_path / "images"
        labels_dir = positive_path / "labels"
        
        # Get unique sample IDs (without _OL/_SD suffix)
        image_files = list(images_dir.glob("*_OL.png"))
        
        for img_file in image_files:
            # Extract sample ID (P00000)
            sample_id = img_file.stem.replace("_OL", "")
            json_file = labels_dir / f"{sample_id}.json"
            
            samples.append({
                'id': sample_id,
                'image': img_file,
                'label': json_file,
                'has_objects': json_file.exists()
            })
    
    # Negative samples (no annotations)
    if negative_path.exists():
        images_dir = negative_path / "images"
        
        for img_file in images_dir.glob("*_OL.png"):
            sample_id = img_file.stem.replace("_OL", "")
            
            samples.append({
                'id': sample_id,
                'image': img_file,
                'label': None,
                'has_objects': False
            })
    
    print(f"📊 Found {len(samples)} samples")
    
    if max_samples:
        samples = samples[:max_samples]
        print(f"   Limited to {max_samples} samples")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)
    
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    print(f"   Train: {len(train_samples)}")
    print(f"   Val: {len(val_samples)}")
    print(f"   Test: {len(test_samples)}")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    def process_split(split_samples: List[Dict], split_name: str):
        print(f"\n📁 Processing {split_name}...")
        success = 0
        
        for i, sample in enumerate(split_samples):
            out_img = out_path / "images" / split_name / f"{sample['id']}_OL.png"
            out_lbl = out_path / "labels" / split_name / f"{sample['id']}_OL.txt"
            
            # Use empty path for label if no annotation
            label_path = sample['label'] if sample['label'] else Path("/nonexistent")
            
            if process_image(sample['image'], label_path, out_img, out_lbl):
                success += 1
            
            if (i + 1) % 500 == 0:
                print(f"   Processed {i + 1}/{len(split_samples)}")
        
        print(f"   ✅ {split_name}: {success}/{len(split_samples)} processed")
        return success
    
    total = 0
    total += process_split(train_samples, 'train')
    total += process_split(val_samples, 'val')
    total += process_split(test_samples, 'test')
    
    # Create data.yaml
    yaml_content = f"""# XrayVision Dataset
path: {out_path.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    
    with open(out_path / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "=" * 60)
    print(f"✅ Preprocessing terminé!")
    print(f"   Total: {total} images traitées")
    print(f"   Output: {out_path}")
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess X-ray data')
    parser.add_argument('--input', type=str, default='../data/RawData',
                        help='Path to raw data')
    parser.add_argument('--output', type=str, default='../data/processed',
                        help='Path to output processed data')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    preprocess_data(
        raw_data_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples
    )
