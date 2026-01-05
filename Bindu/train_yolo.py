"""
train_yolo.py
Train YOLOv11-s for PPE detection
(YAML is at project ROOT, not inside css-data)
"""

# ===================== FIX CUDA / CuBLAS WARNING =====================
import torch

torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
# =====================================================================

from ultralytics import YOLO

# ---------------- Configuration ----------------
DATA_YAML = "ppe_data.yaml"   # ✅ YAML AT ROOT
BASE_MODEL = "yolo11s.pt"
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16               # Safe for RTX 4050 (6GB)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    print("=" * 60)
    print("SafetyEye - YOLOv11 Training")
    print("=" * 60)

    print(f"\n📂 Dataset YAML: {DATA_YAML}")
    print(f"📂 Base model:   {BASE_MODEL}")
    print(f"🔢 Epochs:       {EPOCHS}")
    print(f"📐 Image size:   {IMAGE_SIZE}")
    print(f"📦 Batch size:   {BATCH_SIZE}")
    print(f"💻 Device:       {DEVICE}")
    print("\n⏳ Training started...\n")

    # Load model
    model = YOLO(BASE_MODEL)

    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,

        # Augmentation (balanced for speed + accuracy)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,

        # Training control
        patience=30,
        cache="disk",
        workers=8,
        save=True,

        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Misc
        close_mosaic=15,
        amp=True,
        val=True,

        # Output
        project="runs/detect",
        name="train",
        exist_ok=True,

        verbose=True,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print("\n📂 Best weights:")
    print("   runs/detect/train/weights/best.pt")
    print("\n📊 Check accuracy:")
    print("   python check_accuracy.py")

    return results


# ✅ TRAIN RUNS ONLY ONCE
if __name__ == "__main__":
    train()
