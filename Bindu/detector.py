from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import io
import streamlit as st
from typing import Tuple, List, Dict, Optional
import argparse
import os
from datetime import datetime

# Path to weights (change if your path differs)
WEIGHTS = "runs/detect/train/weights/best.pt"

# Simple set of class names (optional) that indicate a violation.
# Adjust these names to match exactly what your model reports (model.names).
VIOLATION_KEYWORDS = {
    "NO_", "NO-", "NO", "No", "no"  # generic prefixes/keywords to catch classes like NO_Safety_Vest
}
# If you know exact class strings for violations, list them explicitly:
EXPLICIT_VIOLATIONS = {
    "NO_Safety_Vest", "NO-Hardhat", "NO_Mask", "NO-Safety Vest", "NO_SafetyVest"
}


@st.cache_resource(show_spinner=False)
def load_model(path: str = WEIGHTS) -> YOLO:
    """
    Load YOLO model and cache it for Streamlit sessions or CLI runs.
    """
    model = YOLO(path)
    return model


def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    """
    Convert PIL RGB -> OpenCV BGR numpy array.
    """
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """
    Convert OpenCV BGR -> PIL Image (RGB).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def _tensor_to_scalar(x):
    """
    Helper: convert a torch tensor/np array scalar to Python float/int safely.
    """
    try:
        # works for torch tensors on CPU
        return float(x.cpu().numpy().item())
    except Exception:
        try:
            return float(np.array(x).item())
        except Exception:
            return float(x)


def infer_pil(
    img_pil: Image.Image,
    model: Optional[YOLO] = None,
    conf_thresh: float = 0.3,
    return_bytes: bool = True
) -> Tuple[Image.Image, Optional[io.BytesIO], List[Dict]]:
    """
    Run YOLO inference on a PIL image.
    Args:
      img_pil: input PIL image (RGB)
      model: optional YOLO model object. If None, it will be loaded with load_model().
      conf_thresh: filter detections below this confidence
      return_bytes: if True, also return annotated image as BytesIO (jpeg)
    Returns:
      annotated_pil: PIL.Image with boxes/labels drawn
      annotated_bytes: io.BytesIO (jpeg) or None
      detections: list of dicts:
          {
            'class_id': int,
            'class_name': str,
            'conf': float,
            'xyxy': [x1,y1,x2,y2]
          }
    """
    if model is None:
        model = load_model()

    # convert PIL -> numpy (RGB) for Ultralytics
    arr = np.array(img_pil.convert("RGB"))
    res = model(arr)[0]  # first (and only) result

    # annotated numpy array (RGB) from ultralytics
    annotated_np = res.plot()  # often returns RGB numpy array
    annotated_pil = Image.fromarray(annotated_np)

    # collect detections
    dets: List[Dict] = []
    # res.boxes may contain tensors; iterate defensively
    try:
        for box in res.boxes:
            # class id
            try:
                cls_raw = box.cls[0]
            except Exception:
                cls_raw = box.cls
            cls_id = int(_tensor_to_scalar(cls_raw))

            # confidence
            try:
                conf_raw = box.conf[0]
            except Exception:
                conf_raw = box.conf
            conf = float(_tensor_to_scalar(conf_raw))

            # xyxy
            try:
                xyxy_raw = box.xyxy[0]
            except Exception:
                xyxy_raw = box.xyxy
            xyxy_arr = np.array(xyxy_raw).astype(float)
            xyxy = [float(round(x, 2)) for x in xyxy_arr.tolist()]

            # class name from model
            cls_name = model.names[cls_id] if hasattr(model, "names") and cls_id in model.names else str(cls_id)

            # apply confidence threshold filter
            if conf < conf_thresh:
                continue

            dets.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": round(conf, 4),
                "xyxy": xyxy
            })
    except Exception:
        # If res.boxes is not iterable as expected, attempt other fallbacks.
        # Keep dets empty in that case.
        pass

    # annotated image -> bytes (for download)
    annotated_bytes = None
    if return_bytes:
        buf = io.BytesIO()
        annotated_pil.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        annotated_bytes = buf

    return annotated_pil, annotated_bytes, dets


def analyze_detections(detections: List[Dict]) -> Dict:
    """
    Given the filtered detection list, decide violations, counts and a simple summary.
    Returns:
      {
        'counts': {class_name: count, ...},
        'violations': [ {timestamp(optional), class_name, conf, xyxy, message}, ... ],
        'total_people': int (if 'Person' class exists),
      }
    """
    counts = {}
    violations = []
    total_people = 0

    for d in detections:
        name = d.get("class_name", "Unknown")
        counts[name] = counts.get(name, 0) + 1
        if name.lower().startswith("person"):
            total_people += 1

        # violation logic:
        is_violation = False
        if name in EXPLICIT_VIOLATIONS:
            is_violation = True
        else:
            # check keywords
            for kw in VIOLATION_KEYWORDS:
                if kw in name:
                    is_violation = True
                    break

        if is_violation:
            violations.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "class": name,
                "conf": d.get("conf"),
                "xyxy": d.get("xyxy"),
                "message": f"{name} detected — possible violation"
            })

    return {
        "counts": counts,
        "violations": violations,
        "total_people": total_people
    }


# ---------------------------
# CLI / executable behaviour
# ---------------------------
def _make_output_filename(input_path: str, suffix: str = "_annotated.jpg") -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return f"{base}{suffix}"


def _cli_main():
    p = argparse.ArgumentParser(description="Run YOLO detector (ultralytics) on an input image and save annotated output.")
    p.add_argument("--image", "-i", required=True, help="Path to input image")
    p.add_argument("--weights", "-w", default=WEIGHTS, help="Path to model weights (pt file)")
    p.add_argument("--conf", "-c", type=float, default=0.3, help="Confidence threshold (0-1)")
    p.add_argument("--out", "-o", default=None, help="Output annotated filename (optional)")
    args = p.parse_args()

    img_path = args.image
    weights = args.weights
    conf = args.conf
    out = args.out

    if not os.path.exists(img_path):
        print(f"ERROR: input image not found: {img_path}")
        return 2

    if not os.path.exists(weights):
        print(f"ERROR: weights file not found: {weights}")
        return 3

    print("Loading model:", weights)
    model = YOLO(weights)

    # open image
    img = Image.open(img_path).convert("RGB")
    print("Running inference on:", img_path)
    annotated_pil, annotated_bytes, dets = infer_pil(img, model=model, conf_thresh=conf, return_bytes=True)

    # analyze
    summary = analyze_detections(dets)
    print("Detections summary:")
    for k, v in summary["counts"].items():
        print(f"  {k}: {v}")
    print(f"Total people (Person class): {summary['total_people']}")
    if summary["violations"]:
        print("Violations found:")
        for viol in summary["violations"]:
            print(f" - {viol['timestamp']} | {viol['class']} | conf={viol['conf']} | {viol['message']}")
    else:
        print("No violations found (by configured keywords).")

    # save annotated image
    if out is None:
        out = _make_output_filename(img_path)
    with open(out, "wb") as f:
        f.write(annotated_bytes.getbuffer())
    print("Saved annotated output to:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
