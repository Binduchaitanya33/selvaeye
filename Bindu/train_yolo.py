"""
train_yolo.py
Train YOLOv8 model for PPE detection with optimized parameters.

Target: Achieve >70% mAP@50 accuracy for SafetyEye PPE detection.
"""

from ultralytics import YOLO
import os

# Configuration
DATA_YAML = "ppe_data.yaml"
BASE_MODEL = "yolo11x.pt"  # Upgraded to Extreme model for maximum accuracy
EPOCHS = 100  # More epochs for better accuracy
IMAGE_SIZE = 640
BATCH_SIZE = 2  # Reduced for yolo11x to avoid memory errors on CPU


def train():
    """Train the YOLO model with optimized settings."""
    
    print("=" * 60)
    print("SafetyEye - YOLOv8 Training")
    print("=" * 60)
    print(f"\n📂 Data config: {DATA_YAML}")
    print(f"📂 Base model:  {BASE_MODEL}")
    print(f"🔢 Epochs:      {EPOCHS}")
    print(f"📐 Image size:  {IMAGE_SIZE}")
    print(f"📦 Batch size:  {BATCH_SIZE}")
    print("\n⏳ Training will take some time. Please wait...\n")
    
    # Load base model
    model = YOLO(BASE_MODEL)
    
    # Train with optimized hyperparameters
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        
        # Data augmentation for better generalization
        augment=True,
        mosaic=1.0,        # Mosaic augmentation
        mixup=0.1,         # Mixup augmentation
        copy_paste=0.1,    # Copy-paste augmentation
        
        # Training settings
        patience=20,       # Early stopping patience
        save=True,         # Save checkpoints
        save_period=10,    # Save every 10 epochs
        
        # Optimization
        optimizer="AdamW", # Better optimizer
        lr0=0.01,          # Initial learning rate
        lrf=0.01,          # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,   # Warmup epochs
        
        # Output
        project="runs/detect",
        name="train",
        exist_ok=True,     # Overwrite existing
        
        # Display
        verbose=True,
        plots=True,        # Generate training plots
    )
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\n📂 Weights saved to: runs/detect/train/weights/best.pt")
    print("\n📊 To check accuracy, run:")
    print("   python check_accuracy.py")
    print("\n🌐 Or view in Streamlit app:")
    print("   Click '📊 Accuracy' in the sidebar")
    
    return results


if __name__ == "__main__":
    train()
