"""
ui_accuracy.py
Streamlit page to display model accuracy metrics for SafetyEye.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import accuracy utilities
try:
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
        DATA_YAML
    )
    ACCURACY_UTILS_AVAILABLE = True
except Exception as e:
    ACCURACY_UTILS_AVAILABLE = False
    print(f"Error importing accuracy_utils: {e}")

# Try to import style utilities
try:
    from style_utils import apply_global_styles
except Exception:
    def apply_global_styles():
        pass


def _inject_accuracy_styles():
    """Inject CSS styles for the accuracy page."""
    st.markdown("""
    <style>
    .accuracy-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .accuracy-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }
    .accuracy-header p {
        margin: 10px 0 0 0;
        opacity: 0.9;
        font-size: 14px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .accuracy-section {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 2px solid #f3f4f6;
    }
    .class-metrics-row {
        display: flex;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #f3f4f6;
    }
    .class-metrics-row:last-child {
        border-bottom: none;
    }
    .class-name {
        flex: 1;
        font-weight: 500;
        color: #374151;
    }
    .class-score {
        font-family: monospace;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 14px;
    }
    .score-good { background: #dcfce7; color: #16a34a; }
    .score-ok { background: #fef9c3; color: #ca8a04; }
    .score-poor { background: #fee2e2; color: #dc2626; }
    .training-info {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
    }
    .training-stat {
        background: #f9fafb;
        padding: 12px 16px;
        border-radius: 8px;
        text-align: center;
    }
    .training-stat-label {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 4px;
    }
    .training-stat-value {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
    }
    </style>
    """, unsafe_allow_html=True)


def show_accuracy(go):
    """
    Main function to display the accuracy page.
    
    Args:
        go: Navigation function to switch pages
    """
    apply_global_styles()
    _inject_accuracy_styles()
    
    # Header
    st.markdown("""
    <div class="accuracy-header">
        <h1>📊 Model Accuracy Dashboard</h1>
        <p>View training metrics, validation results, and per-class performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("← Home"):
            go("home")
    with col2:
        if st.button("← Dashboard"):
            go("dashboard")
    
    st.markdown("---")
    
    if not ACCURACY_UTILS_AVAILABLE:
        st.error("❌ Accuracy utilities not available. Please check accuracy_utils.py")
        return
    
    # Check if model weights exist
    weights_exist = os.path.exists(WEIGHTS_PATH)
    
    if not weights_exist:
        st.warning(f"⚠️ Model weights not found at `{WEIGHTS_PATH}`")
        st.info("Train your model first using `train_yolo.py` to see accuracy metrics.")
        return
    
    # ========================
    # Section 1: Training History
    # ========================
    st.markdown('<div class="accuracy-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Training History</div>', unsafe_allow_html=True)
    
    training_df = load_training_results()
    
    if training_df is not None and not training_df.empty:
        summary = get_training_summary(training_df)
        
        # Training stats
        st.markdown('<div class="training-info">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="training-stat">
                <div class="training-stat-label">Epochs</div>
                <div class="training-stat-value">{summary.get('epochs_completed', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="training-stat">
                <div class="training-stat-label">Final mAP50</div>
                <div class="training-stat-value">{summary.get('mAP50', 0):.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="training-stat">
                <div class="training-stat-label">Precision</div>
                <div class="training-stat-value">{summary.get('precision', 0):.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="training-stat">
                <div class="training-stat-label">Recall</div>
                <div class="training-stat-value">{summary.get('recall', 0):.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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
        with st.expander("📋 View Raw Training Data"):
            st.dataframe(training_df, use_container_width=True)
    else:
        st.info("No training history found. Run training to generate metrics.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================
    # Section 2: Live Validation
    # ========================
    st.markdown('<div class="accuracy-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Model Validation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Run validation on your dataset to get the latest accuracy metrics.")
    
    with col2:
        validate_split = st.selectbox("Dataset Split", ["val", "test"], index=0)
    
    if st.button("🔄 Run Validation", type="primary"):
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
        
        # Key metrics grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metrics_card(
                "mAP@50",
                metrics.get('mAP50', 0),
                "Mean Average Precision at IoU=0.5"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metrics_card(
                "mAP@50-95",
                metrics.get('mAP50_95', 0),
                "Average over IoU thresholds"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metrics_card(
                "Precision",
                metrics.get('precision', 0),
                "True positives / Predictions"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_metrics_card(
                "Recall",
                metrics.get('recall', 0),
                "True positives / Ground truths"
            ), unsafe_allow_html=True)
        
        # Per-class breakdown
        st.markdown("#### Per-Class Performance (AP@50)")
        
        per_class = metrics.get('per_class_ap50', {})
        if per_class:
            for class_name, ap in per_class.items():
                score_class = "score-good" if ap >= 0.7 else "score-ok" if ap >= 0.5 else "score-poor"
                st.markdown(f"""
                <div class="class-metrics-row">
                    <span class="class-name">{class_name}</span>
                    <span class="class-score {score_class}">{ap:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Per-class metrics not available.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================
    # Section 3: Model Info
    # ========================
    st.markdown('<div class="accuracy-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔧 Model Information</div>', unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("💡 **Tip:** Run more training epochs to improve accuracy. Higher mAP50 indicates better detection performance.")


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="SafetyEye Accuracy", layout="wide")
    show_accuracy(lambda x: None)
