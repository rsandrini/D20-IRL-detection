#!/usr/bin/env python3
"""
Convert Pascal VOC XML annotations to YOLO format for YOLOv8 training.

Usage:
    python scripts/convert_voc_to_yolo.py

Output:
    data/
      images/train/  images/val/
      labels/train/  labels/val/
      data.yaml

Train after running this:
    pip install ultralytics
    yolo train model=yolov8n.pt data=data/data.yaml epochs=150 imgsz=640 batch=16

Export to TFLite after training:
    yolo export model=runs/detect/train/weights/best.pt format=tflite imgsz=640 int8=True

Then copy the exported model:
    cp runs/detect/train/weights/best_saved_model/best_int8.tflite tflite1/custom_model_lite/detect.tflite
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

DICE_TRAINING_DIR = Path("dice_training")
OUTPUT_DIR = Path("data")
VAL_SPLIT = 0.2
SEED = 42

CLASSES = [str(i) for i in range(1, 21)]  # 1..20 → class IDs 0..19


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASSES:
            print(f"  WARNING: unknown class '{name}' in {xml_path.name}, skipping")
            continue

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # YOLO format: normalized cx, cy, w, h
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        class_id = CLASSES.index(name)
        objects.append((class_id, cx, cy, w, h))

    return objects


def main():
    xml_files = sorted([
        f for f in DICE_TRAINING_DIR.glob("*.xml")
        if "Zone.Identifier" not in f.name
    ])

    pairs = []
    for xml_path in xml_files:
        img_path = xml_path.with_suffix(".jpg")
        if img_path.exists():
            pairs.append((img_path, xml_path))
        else:
            print(f"WARNING: image not found for {xml_path.name}")

    print(f"Found {len(pairs)} image/annotation pairs")

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(pairs)
    val_size = int(len(pairs) * VAL_SPLIT)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    print(f"Split → train: {len(train_pairs)}, val: {len(val_pairs)}")

    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    label_counts = defaultdict(int)
    skipped = 0

    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for img_path, xml_path in split_pairs:
            objects = parse_xml(xml_path)
            if not objects:
                skipped += 1
                continue

            shutil.copy(img_path, OUTPUT_DIR / "images" / split / img_path.name)

            label_file = OUTPUT_DIR / "labels" / split / (img_path.stem + ".txt")
            with open(label_file, "w") as f:
                for class_id, cx, cy, w, h in objects:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    label_counts[CLASSES[class_id]] += 1

    # Write data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    print(f"\nLabel distribution (total instances per face):")
    total = 0
    for cls in CLASSES:
        count = label_counts[cls]
        bar = "█" * (count // 2)
        print(f"  Face {int(cls):>2}: {count:>3}  {bar}")
        total += count
    print(f"\n  Total instances : {total}")
    print(f"  Skipped (no obj): {skipped}")
    print(f"\ndata.yaml → {yaml_path}")
    print("\n--- Next steps ---")
    print("pip install ultralytics")
    print(f"yolo train model=yolov8n.pt data={yaml_path.absolute()} epochs=150 imgsz=640 batch=16 patience=30")
    print("\nAfter training:")
    print("yolo export model=runs/detect/train/weights/best.pt format=tflite imgsz=640 int8=True")
    print("cp runs/detect/train/weights/best_saved_model/best_int8.tflite tflite1/custom_model_lite/detect.tflite")


if __name__ == "__main__":
    main()
