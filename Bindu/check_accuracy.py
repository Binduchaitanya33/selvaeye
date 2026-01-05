"""
check_accuracy.py
Accurate validation script for YOLOv11 models.

Usage:
    python check_accuracy.py
    python check_accuracy.py --weights runs/detect/train/weights/best.pt
    python check_accuracy.py --data css-data.yaml
"""

import argparse
import os
import sys
import yaml

from ultralytics import YOLO

# ✅ MUST MATCH TRAINING
DEFAULT_WEIGHTS = "runs/detect/train/weights/best.pt"
DEFAULT_DATA = "ppe_data.yaml"


def print_separator(title: str = ""):
    width = 60
    if title:
        padding = (width - len(title) - 2) // 2
        print("\n" + "=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Accuracy Checker")
    parser.add_argument("--weights", "-w", default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", "-d", default=DEFAULT_DATA)
    parser.add_argument("--split", "-s", default="val", choices=["val", "test"])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # ----------------- File checks -----------------
    if not os.path.exists(args.weights):
        print(f"❌ Weights not found: {args.weights}")
        return 1

    if not os.path.exists(args.data):
        print(f"❌ Data YAML not found: {args.data}")
        return 1

    # ----------------- Load YAML -----------------
    with open(args.data, "r") as f:
        data_yaml = yaml.safe_load(f)

    yaml_names = data_yaml.get("names", [])
    # Handle both dict and list formats
    if isinstance(yaml_names, dict):
        yaml_names = [yaml_names[i] for i in sorted(yaml_names.keys())]
    num_classes = len(yaml_names)

    print_separator("YOLOv11 Model Accuracy Check")
    print(f"📦 Weights: {args.weights}")
    print(f"📄 Dataset: {args.data}")
    print(f"📊 Split:   {args.split}")

    print("\n📋 Classes from YAML:")
    for i, n in enumerate(yaml_names):
        print(f"   {i}: {n}")

    # ----------------- Load model -----------------
    print("\n⏳ Loading model...")
    model = YOLO(args.weights)
    print("✅ Model loaded")

    print("\n📋 Classes from Model:")
    for i, n in model.names.items():
        print(f"   {i}: {n}")

    # ----------------- Validation -----------------
    print_separator("Running Validation")
    print("⏳ Validating...")

    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=640,
        conf=0.25,
        iou=0.6,
        verbose=args.verbose
    )

    # ----------------- Metrics -----------------
    print_separator("Results")

    box = results.box

    print("\n🎯 Overall Metrics:")
    print(f"   • Precision:   {pct(box.mp)}")
    print(f"   • Recall:      {pct(box.mr)}")
    print(f"   • mAP@50:      {pct(box.map50)}")
    print(f"   • mAP@50-95:   {pct(box.map)}")

    # ----------------- Per-class AP -----------------
    print("\n📊 Per-Class AP@50:")
    ap50 = box.ap50.tolist()

    for i in range(num_classes):
        name = yaml_names[i]
        ap = ap50[i]
        bar_len = int(ap * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        if ap >= 0.75:
            status = "✅"
        elif ap >= 0.5:
            status = "⚠️"
        else:
            status = "❌"

        print(f"   {status} {name:20} {bar} {pct(ap)}")

    # ----------------- Final verdict -----------------
    print_separator("Final Verdict")

    if box.map50 >= 0.8:
        print(f"✅ Model is EXCELLENT (mAP@50 = {pct(box.map50)})")
        print("   Ready for deployment or demo.")
    elif box.map50 >= 0.6:
        print(f"⚠️ Model is GOOD (mAP@50 = {pct(box.map50)})")
        print("   Minor gains possible with more data.")
    else:
        print(f"❌ Model accuracy is LOW (mAP@50 = {pct(box.map50)})")
        print("   Check dataset, labels, or class balance.")

    print_separator()
    print("ℹ️ Notes:")
    print(" • Always validate using the SAME YAML used for training")
    print(" • Use best.pt, not last.pt")
    print(" • More epochs alone will NOT fix data issues")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
