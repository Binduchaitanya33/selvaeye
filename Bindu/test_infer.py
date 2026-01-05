from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# Path to your trained model weights
weights = "runs/detect/train/weights/last.pt"   # use best.pt if available

# Load model
model = YOLO(weights)

# Pick one test image automatically
test_folder = Path(r"C:\Users\bindu\Desktop\archive\css-data\test\images")
img = str(next(test_folder.glob("*.*")))   # get first image

# Run inference
res = model(img)[0]
out = res.plot()

# Save output image
out_file = "inference_out.jpg"
Image.fromarray(out).save(out_file)

print(f"✔ Saved output → {out_file}")
