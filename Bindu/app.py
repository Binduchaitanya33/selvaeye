"""
app.py
Lightweight router for the refactored SafetyEye Streamlit app.

Responsibilities:
- Page routing (home <-> dashboard)
- Minimal global session-state initialization
- Friendly error handling if a UI module fails to import
- Small status info (model weights existence) in the sidebar (non-intrusive)
"""

import streamlit as st
import os
import traceback

st.set_page_config(page_title="SafetyEye Dashboard", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Minimal session-state defaults
# -------------------------
st.session_state.setdefault("page", "home")
st.session_state.setdefault("violation_log", [])   # persisted violations across session
st.session_state.setdefault("sim_logs", [])        # simulator logs
st.session_state.setdefault("sim_total", 0)        # simulator total violations

# -------------------------
# Helper: simple model-file status
# (Do NOT import detector or load model here — keep the router lightweight)
# -------------------------
def model_weights_exist(path: str = "runs/detect/train/weights/best.pt") -> bool:
    """Return True if the typical trained weights file exists on disk."""
    try:
        return os.path.exists(path)
    except Exception:
        return False


# -------------------------
# Navigation helper
# -------------------------
def go(page_name: str):
    """Set the current page and let Streamlit rerun to render the target view."""
    st.session_state["page"] = page_name


# -------------------------
# Top-level sidebar / quick status
# -------------------------
with st.sidebar:
    st.markdown("## SafetyEye — Status")
    weights_found = model_weights_exist()
    if weights_found:
        st.success("Model weights found (best.pt)")
    else:
        st.info("Model weights not found — training or model integration pending")
        st.caption("Expected path: runs/detect/train/weights/best.pt")

    st.markdown("---")
    st.markdown("Navigation")
    if st.button("Home"):
        go("home")
    if st.button("Dashboard"):
        go("dashboard")
    if st.button("📊 Accuracy"):
        go("accuracy")

# -------------------------
# Main router
# -------------------------
def main():
    page = st.session_state.get("page", "home")

    try:
        if page == "home":
            # load and show home view
            from ui_home import show_home  # dynamic import so app.py stays lightweight
            show_home(go)
        elif page == "accuracy":
            # load and show accuracy metrics view
            from ui_accuracy import show_accuracy
            show_accuracy(go)
        else:
            # load and show dashboard view
            from ui_dashboard import show_dashboard
            show_dashboard(go)

    except Exception as e:
        # Friendly error page: show stacktrace and guidance
        st.title("SafetyEye — Error")
        st.error("An error occurred while loading the requested page.")
        st.markdown("**Error:**")
        st.code(str(e))
        st.markdown("**Traceback (for debugging):**")
        tb = traceback.format_exc()
        st.code(tb)
        st.markdown(
            "If this is caused by a missing module (for example `detector.py` or model weights), "
            "please ensure the file exists in the project root and that required packages are installed "
            "(e.g. `ultralytics`, `streamlit`)."
        )


if __name__ == "__main__":
    main()
