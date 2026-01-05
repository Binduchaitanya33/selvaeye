import streamlit as st


def show_home(go):
    st.title("SafetyEye")
    st.caption("Use the sidebar to navigate: Home, Dashboard, Accuracy.")

    st.write(
        "This app provides a simple Streamlit interface for occupancy monitoring and model metrics. "
        "No custom navbar is used so Streamlit's default top bar remains unchanged."
    )
