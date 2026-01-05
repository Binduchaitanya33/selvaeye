"""
accuracy_utils.py
Utilities to evaluate model accuracy and display metrics in SafetyEye.

Functions:
- validate_model: Run validation on test/val dataset
- load_training_results: Read training history CSV
- get_model_metrics: Get mAP, precision, recall etc.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from ultralytics import YOLO
import streamlit as st

# Default paths
WEIGHTS_PATH = "runs/detect/train/weights/best.pt"
RESULTS_CSV = "runs/detect/train/results.csv"
DATA_YAML = "ppe_data.yaml"


def load_training_results(results_path: str = RESULTS_CSV) -> Optional[pd.DataFrame]:
    """
    Load training results CSV from YOLO training.
    Returns DataFrame with training metrics or None if not found.
    """
    if not os.path.exists(results_path):
        return None
    
    try:
        df = pd.read_csv(results_path)
        # Clean column names (remove leading spaces)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error loading results CSV: {e}")
        return None


def get_training_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract key metrics from training results DataFrame.
    """
    if df is None or df.empty:
        return {}
    
    last_row = df.iloc[-1]
    
    summary = {
        "epochs_completed": int(last_row.get("epoch", 0)),
        "precision": float(last_row.get("metrics/precision(B)", 0)),
        "recall": float(last_row.get("metrics/recall(B)", 0)),
        "mAP50": float(last_row.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(last_row.get("metrics/mAP50-95(B)", 0)),
        "box_loss": float(last_row.get("train/box_loss", 0)),
        "cls_loss": float(last_row.get("train/cls_loss", 0)),
        "val_box_loss": float(last_row.get("val/box_loss", 0)),
        "val_cls_loss": float(last_row.get("val/cls_loss", 0)),
    }
    
    return summary


@st.cache_resource(show_spinner=False)
def load_model_for_eval(weights_path: str = WEIGHTS_PATH) -> Optional[YOLO]:
    """
    Load YOLO model for evaluation.
    """
    if not os.path.exists(weights_path):
        return None
    try:
        return YOLO(weights_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def validate_model(
    model: Optional[YOLO] = None,
    data_yaml: str = DATA_YAML,
    weights_path: str = WEIGHTS_PATH,
    split: str = "val"
) -> Optional[Dict[str, Any]]:
    """
    Run validation on the model and return metrics.
    
    Args:
        model: YOLO model object (if None, will load from weights_path)
        data_yaml: Path to data YAML file
        weights_path: Path to model weights
        split: Dataset split to validate on ('val' or 'test')
    
    Returns:
        Dictionary with validation metrics or None on error
    """
    if model is None:
        model = load_model_for_eval(weights_path)
    
    if model is None:
        return None
    
    if not os.path.exists(data_yaml):
        return None
    
    try:
        # Run validation
        results = model.val(data=data_yaml, split=split, verbose=False)
        
        # Extract metrics
        metrics = {
            # Box detection metrics
            "mAP50": float(results.box.map50) if hasattr(results, 'box') else 0.0,
            "mAP50_95": float(results.box.map) if hasattr(results, 'box') else 0.0,
            "precision": float(results.box.mp) if hasattr(results, 'box') else 0.0,
            "recall": float(results.box.mr) if hasattr(results, 'box') else 0.0,
            
            # Per-class metrics
            "per_class_ap50": dict(zip(
                model.names.values() if hasattr(model, 'names') else [],
                results.box.ap50.tolist() if hasattr(results, 'box') else []
            )),
            "per_class_ap": dict(zip(
                model.names.values() if hasattr(model, 'names') else [],
                results.box.ap.tolist() if hasattr(results, 'box') else []
            )),
        }
        
        return metrics
    
    except Exception as e:
        print(f"Validation error: {e}")
        return None


def get_class_names(weights_path: str = WEIGHTS_PATH) -> list:
    """
    Get class names from the trained model.
    """
    try:
        model = load_model_for_eval(weights_path)
        if model and hasattr(model, 'names'):
            return list(model.names.values())
    except Exception:
        pass
    return []


def format_percentage(value: float) -> str:
    """Format a 0-1 value as percentage string."""
    return f"{value * 100:.2f}%"


def create_metrics_card(title: str, value: float, description: str = "") -> str:
    """
    Create HTML for a styled metrics card.
    """
    percentage = format_percentage(value) if value <= 1 else f"{value:.4f}"
    
    # Color based on value (green for good, yellow for ok, red for poor)
    if value >= 0.7:
        color = "#16a34a"  # green
        bg = "#dcfce7"
    elif value >= 0.5:
        color = "#ca8a04"  # yellow
        bg = "#fef9c3"
    else:
        color = "#dc2626"  # red
        bg = "#fee2e2"
    
    return f"""
    <div style="
        background: {bg};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid {color}22;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    ">
        <div style="font-size: 14px; color: #666; margin-bottom: 8px;">{title}</div>
        <div style="font-size: 32px; font-weight: 700; color: {color};">{percentage}</div>
        {f'<div style="font-size: 12px; color: #888; margin-top: 8px;">{description}</div>' if description else ''}
    </div>
    """


def create_progress_bar(label: str, value: float, max_val: float = 1.0) -> str:
    """
    Create HTML for a styled progress bar.
    """
    percentage = (value / max_val) * 100
    color = "#3b82f6" if percentage >= 50 else "#f59e0b" if percentage >= 30 else "#ef4444"
    
    return f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-size: 13px; font-weight: 500;">{label}</span>
            <span style="font-size: 13px; color: #666;">{value:.2%}</span>
        </div>
        <div style="background: #e5e7eb; border-radius: 8px; height: 10px; overflow: hidden;">
            <div style="background: {color}; height: 100%; width: {percentage}%; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """
