"""
ui_accuracy.py
Streamlit page to display model accuracy metrics for SafetyEye.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def _import_accuracy_utils():
    """Import accuracy utilities lazily to avoid breaking app startup."""
    from accuracy_utils import (
        load_training_results,
        get_training_summary,
        validate_model,
        load_model_for_eval,
        get_class_names,
        create_metrics_card,
        create_progress_bar,
        format_percentage,
        WEIGHTS_PATH,
        RESULTS_CSV,
        DATA_YAML,
    )

    return {
        "load_training_results": load_training_results,
        "get_training_summary": get_training_summary,
        "validate_model": validate_model,
        "load_model_for_eval": load_model_for_eval,
        "get_class_names": get_class_names,
        "create_metrics_card": create_metrics_card,
        "create_progress_bar": create_progress_bar,
        "format_percentage": format_percentage,
        "WEIGHTS_PATH": WEIGHTS_PATH,
        "RESULTS_CSV": RESULTS_CSV,
        "DATA_YAML": DATA_YAML,
    }

def show_accuracy(go):
    """
    Main function to display the accuracy page.
    
    Args:
        go: Navigation function to switch pages
    """
    st.title("Model Accuracy")
    st.caption("Training metrics, validation results, and per-class performance.")

    try:
        au = _import_accuracy_utils()
    except Exception as e:
        st.error("Accuracy page can't load because required packages are missing in the current Python environment.")
        st.code(str(e))
        st.caption("Current Python interpreter used by this Streamlit server:")
        st.code(sys.executable)
        st.markdown(
            "If you started Streamlit before activating the `gpu` environment, stop the server (Ctrl+C) and start it again using one of these commands:"
        )
        st.markdown("Run the app using the `gpu` conda environment (where `ultralytics` is installed):")
        st.code("conda activate gpu\nstreamlit run app.py")
        st.markdown("Or use the launcher:")
        st.code("python run_app.py")
        st.markdown("Or force it (no activation needed):")
        st.code("conda run -n gpu --no-capture-output python -m streamlit run app.py")
        return
    
    WEIGHTS_PATH = au["WEIGHTS_PATH"]
    DATA_YAML = au["DATA_YAML"]
    load_training_results = au["load_training_results"]
    get_training_summary = au["get_training_summary"]
    validate_model = au["validate_model"]
    get_class_names = au["get_class_names"]

    # Check if model weights exist
    weights_exist = os.path.exists(WEIGHTS_PATH)
    
    if not weights_exist:
        st.warning(f"Model weights not found at: {WEIGHTS_PATH}")
        st.info("Train your model first (train_yolo.py) to see accuracy metrics.")
        return
    
    # ========================
    # Section 1: Training History
    # ========================
    st.subheader("Training History")
    
    training_df = load_training_results()
    
    if training_df is not None and not training_df.empty:
        summary = get_training_summary(training_df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Epochs", int(summary.get("epochs_completed", 0)))
        col2.metric("Final mAP50", f"{summary.get('mAP50', 0):.1%}")
        col3.metric("Precision", f"{summary.get('precision', 0):.1%}")
        col4.metric("Recall", f"{summary.get('recall', 0):.1%}")
        
        # Show training curves if multiple epochs
        if len(training_df) > 1:
            st.markdown("#### Training Progress")
            
            tab1, tab2 = st.tabs(["Loss Curves", "Metrics Curves"])
            
            with tab1:
                loss_cols = [c for c in training_df.columns if 'loss' in c.lower()]
                if loss_cols:
                    chart_data = training_df[['epoch'] + loss_cols].set_index('epoch')
                    st.line_chart(chart_data)
            
            with tab2:
                metric_cols = [c for c in training_df.columns if 'metrics' in c.lower() or 'mAP' in c or 'precision' in c.lower() or 'recall' in c.lower()]
                if metric_cols:
                    chart_data = training_df[['epoch'] + metric_cols].set_index('epoch')
                    st.line_chart(chart_data)
        
        # Show raw data option
        with st.expander("View raw training data"):
            st.dataframe(training_df, use_container_width=True)
    else:
        st.info("No training history found. Run training to generate metrics.")
    
    # ========================
    # Section 2: Live Validation
    # ========================
    st.subheader("Model Validation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Run validation on your dataset to get the latest accuracy metrics.")
    
    with col2:
        validate_split = st.selectbox("Dataset Split", ["val", "test"], index=0)
    
    if st.button("Run Validation"):
        if not os.path.exists(DATA_YAML):
            st.error(f"Data YAML not found: {DATA_YAML}")
        else:
            with st.spinner("Running validation... This may take a moment."):
                metrics = validate_model(split=validate_split)
                
            if metrics:
                st.success("✅ Validation complete!")
                
                # Store in session for display
                st.session_state['validation_metrics'] = metrics
                st.session_state['validation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                st.error("❌ Validation failed. Check model weights and data paths.")
    
    # Display cached validation results
    if 'validation_metrics' in st.session_state:
        metrics = st.session_state['validation_metrics']
        val_time = st.session_state.get('validation_time', 'Unknown')
        
        st.caption(f"Last validated: {val_time}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("mAP@50", f"{metrics.get('mAP50', 0):.3f}")
        col2.metric("mAP@50-95", f"{metrics.get('mAP50_95', 0):.3f}")
        col3.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        col4.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        
        # Per-class breakdown
        st.markdown("#### Per-class performance (AP@50)")
        
        per_class = metrics.get('per_class_ap50', {})
        if per_class:
            per_class_df = pd.DataFrame(
                [{"class": name, "ap50": ap} for name, ap in per_class.items()]
            ).sort_values("ap50", ascending=False)
            st.dataframe(per_class_df, use_container_width=True)
        else:
            st.info("Per-class metrics not available.")
    
    # ========================
    # Section 3: Model Info
    # ========================
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Path:**")
        st.code(WEIGHTS_PATH)
        
        st.markdown("**Data Configuration:**")
        st.code(DATA_YAML)
    
    with col2:
        class_names = get_class_names()
        if class_names:
            st.markdown("**Classes:**")
            for i, name in enumerate(class_names):
                st.markdown(f"• {i}: {name}")
        else:
            st.info("Unable to load class names")
    
    st.divider()
    st.caption("Tip: run more epochs to improve accuracy. Higher mAP50 typically indicates better detection performance.")


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="SafetyEye Accuracy", layout="wide")
    show_accuracy(lambda x: None)
