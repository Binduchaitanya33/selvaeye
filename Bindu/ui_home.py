import streamlit as st


def show_home(go):
    st.markdown(
        """
        <style>
        .landing { display:flex; align-items:center; justify-content:space-between; gap:20px; padding:28px; background: linear-gradient(90deg,#eef2ff 0%, #ffffff 100%); border-radius:12px; }
        .landing-left { max-width:64%; }
        .hero { font-size:28px; font-weight:800; color:#0f172a; }
        .sub { color:#475569; font-size:15px; margin-top:8px }
        .action { display:flex; gap:12px; margin-top:18px }
        .big-btn { background:#2563eb; color:white; padding:12px 20px; border-radius:10px; text-decoration:none; font-weight:700 }
        .alt-btn { background:#10b981; color:white; padding:12px 20px; border-radius:10px; text-decoration:none; font-weight:700 }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="landing">', unsafe_allow_html=True)
        st.markdown('<div class="landing-left">', unsafe_allow_html=True)
        st.markdown('<div class="hero">👁️‍🗨️ Welcome to SafetyEye</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">Quickly inspect occupancy, monitor safety compliance, and simulate worker scenes. Choose how you want to proceed below.</div>', unsafe_allow_html=True)
        st.markdown('<div class="action">', unsafe_allow_html=True)
        if st.button('Enter Dashboard — Web UI', key='enter_dashboard'):
            go('dashboard')
        if st.button('📊 View Model Accuracy', key='enter_accuracy'):
            go('accuracy')
        if st.button('Open Simulator — Desktop View', key='enter_sim'):
            go('simulator')
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.image("https://images.unsplash.com/photo-1522071820081-009f0129c71c?q=80&w=400&auto=format&fit=crop&ixlib=rb-4.0.3&s=6a5a2bbf5d6c6c3f6e9a2f3a1d6a7b87", width=300)

    st.stop()
