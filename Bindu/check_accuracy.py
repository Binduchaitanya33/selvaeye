"""
check_accuracy.py
Quick script to check model accuracy from command line.

Usage:
    python check_accuracy.py
    python check_accuracy.py --split test
    python check_accuracy.py --weights path/to/weights.pt
"""

import argparse
import os
import sys

from ultralytics import YOLO

# Default paths
DEFAULT_WEIGHTS = "runs/detect/train/weights/best.pt"
DEFAULT_DATA = "ppe_data.yaml"


def print_separator(title: str = ""):
    """Print a formatted separator line."""
    width = 60
    if title:
        padding = (width - len(title) - 2) // 2
        print("\n" + "=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Check SafetyEye model accuracy")
    parser.add_argument(
        "--weights", "-w",
        default=DEFAULT_WEIGHTS,
        help=f"Path to model weights (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--data", "-d",
        default=DEFAULT_DATA,
        help=f"Path to data YAML (default: {DEFAULT_DATA})"
    )
    parser.add_argument(
        "--split", "-s",
        choices=["val", "test"],
        default="val",
        help="Dataset split to validate on (default: val)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.weights):
        print(f"❌ Error: Weights file not found: {args.weights}")
        print("   Train your model first using: python train_yolo.py")
        return 1
    
    if not os.path.exists(args.data):
        print(f"❌ Error: Data YAML not found: {args.data}")
        return 1
    
    print_separator("SafetyEye Model Accuracy Check")
    print(f"\n📂 Weights: {args.weights}")
    print(f"📂 Data:    {args.data}")
    print(f"📂 Split:   {args.split}")
    
    # Load model
    print("\n⏳ Loading model...")
    try:
        model = YOLO(args.weights)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Show class names
    if hasattr(model, 'names'):
        print(f"\n📋 Classes ({len(model.names)}):")
        for idx, name in model.names.items():
            print(f"   {idx}: {name}")
    
    # Run validation
    print_separator("Running Validation")
    print("⏳ This may take a moment...")
    
    try:
        results = model.val(data=args.data, split=args.split, verbose=args.verbose)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1
    
    # Display results
    print_separator("Results")
    
    if hasattr(results, 'box'):
        print(f"\n🎯 Overall Metrics:")
        print(f"   • mAP@50:      {format_percentage(results.box.map50)}")
        print(f"   • mAP@50-95:   {format_percentage(results.box.map)}")
        print(f"   • Precision:   {format_percentage(results.box.mp)}")
        print(f"   • Recall:      {format_percentage(results.box.mr)}")
        
        # Per-class metrics
        if hasattr(results.box, 'ap50') and hasattr(model, 'names'):
            print(f"\n📊 Per-Class AP@50:")
            ap50_list = results.box.ap50.tolist()
            for idx, (class_name, ap) in enumerate(zip(model.names.values(), ap50_list)):
                bar_length = int(ap * 20)  # 20 char max bar
                bar = "█" * bar_length + "░" * (20 - bar_length)
                status = "✅" if ap >= 0.7 else "⚠️" if ap >= 0.5 else "❌"
                print(f"   {status} {class_name:20} {bar} {format_percentage(ap)}")
        
        # Summary
        print_separator("Summary")
        avg_ap = results.box.map50
        if avg_ap >= 0.7:
            print(f"\n✅ Model accuracy is GOOD (mAP@50: {format_percentage(avg_ap)})")
        elif avg_ap >= 0.5:
            print(f"\n⚠️ Model accuracy is MODERATE (mAP@50: {format_percentage(avg_ap)})")
            print("   Consider training for more epochs to improve accuracy.")
        else:
            print(f"\n❌ Model accuracy is LOW (mAP@50: {format_percentage(avg_ap)})")
            print("   Recommended: Train for more epochs or check your dataset.")
    else:
        print("⚠️ Could not extract detailed metrics from validation results.")
    
    print_separator()
    print("\n💡 Tips to improve accuracy:")
    print("   1. Train for more epochs (current training may be insufficient)")
    print("   2. Increase dataset size with more diverse images")
    print("   3. Apply data augmentation during training")
    print("   4. Balance class distribution in your dataset")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
