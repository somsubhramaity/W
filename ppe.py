import os
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

# Ultralytics YOLO
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
#DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
ROOT = Path("datasets")
DATA_DIR = ROOT / "construction-ppe"
YAML_PATH = Path("construction-ppe.yaml")  # put your YAML next to this script (or change path)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Class names (must match YAML order)
NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}

# -----------------------------
# Utils
# -----------------------------


def yolo_to_xyxy(box, w, h):
    # box = [cls, x_center, y_center, width, height] normalized
    _, xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return int(x1), int(y1), int(x2), int(y2)

def draw_boxes(img, labels_path, names=NAMES):
    h, w = img.shape[:2]
    if not labels_path.exists():
        return img

    with open(labels_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for ln in lines:
        parts = ln.split()
        cls = int(parts[0])
        vals = list(map(float, parts))
        x1, y1, x2, y2 = yolo_to_xyxy(vals, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            names.get(cls, str(cls)),
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return img

def dataset_stats(split="train"):
    img_dir = DATA_DIR / "images" / split
    lbl_dir = DATA_DIR / "labels" / split
    imgs = sorted(img_dir.glob("*.*"))
    counts = {k: 0 for k in NAMES.keys()}
    n_empty = 0

    for img_path in imgs:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            n_empty += 1
            continue
        with open(lbl_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            n_empty += 1
            continue
        for ln in lines:
            cls = int(ln.split()[0])
            counts[cls] = counts.get(cls, 0) + 1

    total_boxes = sum(counts.values())
    print(f"\n--- Split: {split} ---")
    print(f"Images: {len(imgs)}")
    print(f"Empty/missing label files: {n_empty}")
    print(f"Total boxes: {total_boxes}")
    print("Boxes per class:")
    for k in sorted(counts.keys()):
        print(f"  {k:>2} {NAMES[k]:<10}: {counts[k]}")
    return counts

def save_label_visualizations(split="train", n=12):
    img_dir = DATA_DIR / "images" / split
    lbl_dir = DATA_DIR / "labels" / split
    imgs = sorted(img_dir.glob("*.*"))
    random.shuffle(imgs)
    imgs = imgs[:n]

    vis_dir = OUT_DIR / f"label_vis_{split}"
    vis_dir.mkdir(exist_ok=True)

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        drawn = draw_boxes(img.copy(), lbl_path)
        out_path = vis_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), drawn)
    print(f"[+] Saved {len(imgs)} label visualizations to: {vis_dir}")

# -----------------------------
# Training + Evaluation
# -----------------------------
def train_baseline():
    # Small model for quick training; swap to yolov8s.pt if you have GPU
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(YAML_PATH),
        imgsz=640,
        epochs=50,
        batch=16,
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else None,
        project="runs",
        name="ppe_baseline",
        exist_ok=True,
    )
    return results

def validate_and_predict():
    model_path = Path("runs/detect/ppe_baseline/weights/best.pt")
    if not model_path.exists():
        print(f"[!] Missing trained model: {model_path}")
        return

    model = YOLO(str(model_path))

    print("\n[+] Running validation...")
    model.val(data=str(YAML_PATH))

    print("\n[+] Running prediction on a few test images...")
    test_dir = DATA_DIR / "images" / "test"
    pred_out = OUT_DIR / "pred_examples"
    pred_out.mkdir(exist_ok=True)

    # Save predictions (Ultralytics saves to runs/ by default; we also copy a few images)
    model.predict(source=str(test_dir), save=True, conf=0.25, iou=0.5)

    print("[+] Predictions saved under runs/detect/ (Ultralytics default).")

def main():

    # Basic dataset sanity checks / EDA
    dataset_stats("train")
    dataset_stats("val")
    dataset_stats("test")

    # Show work: label visualization
    save_label_visualizations("train", n=12)
    save_label_visualizations("val", n=8)

    # Train baseline
    print("\n[+] Training baseline...")
    train_baseline()

    # Evaluate + inference
    validate_and_predict()

    print("\nDone ✅")

if __name__ == "__main__":
    main()
