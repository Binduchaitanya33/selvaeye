"""Streamlit entrypoint for SafetyEye.

This file intentionally keeps UI routing simple:
- A basic sidebar navigation (Home / Dashboard / Accuracy)
- Minimal session-state initialization
- Friendly error display if a page fails to load
"""

import traceback

import streamlit as st

st.set_page_config(page_title="SafetyEye", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Minimal session-state defaults
# -------------------------
st.session_state.setdefault("page", "Home")
st.session_state.setdefault("violation_log", [])   # persisted violations across session
st.session_state.setdefault("sim_logs", [])        # simulator logs
st.session_state.setdefault("sim_total", 0)        # simulator total violations


# -------------------------
# Navigation helper
# -------------------------
def go(page_name: str):
    """Set the current page and let Streamlit rerun to render the target view."""
    st.session_state["page"] = page_name


# -------------------------
# Sidebar navigation (simple)
# -------------------------
with st.sidebar:
    st.title("SafetyEye")
    selection = st.radio(
        "Menu",
        ["Home", "Dashboard", "Accuracy"],
        index=["Home", "Dashboard", "Accuracy"].index(st.session_state.get("page", "Home"))
        if st.session_state.get("page", "Home") in ["Home", "Dashboard", "Accuracy"]
        else 0,
    )
    if selection != st.session_state.get("page"):
        go(selection)

# -------------------------
# Main router
# -------------------------
def main():
    page = st.session_state.get("page", "Home")

    try:
        if page == "Home":
            from ui_home import show_home  # dynamic import so app.py stays lightweight
            show_home(go)
        elif page == "Accuracy":
            from ui_accuracy import show_accuracy
            show_accuracy(go)
        else:
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
