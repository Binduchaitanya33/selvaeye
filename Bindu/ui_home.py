import streamlit as st

from style_utils import apply_global_styles


def show_home(go):
    apply_global_styles()
    st.markdown(
        """
        <style>
        .home-actions {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        .home-img {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--se-border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="se-header">
            <div class="se-header-title">SafetyEye</div>
            <div class="se-header-subtitle">Occupancy monitoring, PPE compliance checks, and simulator preview</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Choose an action")
        st.caption("Use the sidebar anytime to switch pages.")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Open Dashboard", key="enter_dashboard"):
                go("dashboard")
        with c2:
            if st.button("View Accuracy", key="enter_accuracy"):
                go("accuracy")
        with c3:
            if st.button("Open Simulator", key="enter_sim"):
                go("simulator")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Overview")
        st.caption("Lightweight Streamlit UI for demos and monitoring.")
        st.markdown('<div class="home-img">', unsafe_allow_html=True)
        st.image(
            "https://images.unsplash.com/photo-1522071820081-009f0129c71c?q=80&w=400&auto=format&fit=crop&ixlib=rb-4.0.3&s=6a5a2bbf5d6c6c3f6e9a2f3a1d6a7b87",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()
