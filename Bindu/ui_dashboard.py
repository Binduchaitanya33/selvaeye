# ui_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from io import BytesIO

from style_utils import apply_global_styles
from data_utils import make_sample, safe_load_uploaded_csv, default_thresholds, generate_violations
from avatar_utils import make_flat_avatar
from simulator import create_worker_scene, init_sim_state

# Detector helper (must exist in project root)
# detector.py should provide: load_model(), infer_pil(pil_image, model=None), analyze_detections()
try:
    from detector import load_model, infer_pil, analyze_detections
    DETECTOR_AVAILABLE = True
except Exception:
    # don't crash if detector.py or dependencies are missing
    DETECTOR_AVAILABLE = False


def _inject_local_overrides():
    """
    Inject CSS overrides for:
      - Alerts / violation placeholders (light bg, black text)
      - Buttons (background, hover)
      - Improve general contrast for faint content seen previously
    This keeps sidebar styling intact while improving readability in the main area.
    """
    css = """
    <style>
    /* Alerts area: lighter background, black text */
    .alerts-area {
        background: #fff8e6 !important;  /* very light warm tone */
        border: 1px solid rgba(0,0,0,0.04) !important;
        color: #0b0b0b !important;
        padding: 14px !important;
        border-radius: 10px !important;
    }
    /* Violation placeholder box (single-line hint) */
    .violation-placeholder {
        background: #fffaf0;
        border-radius: 8px;
        padding: 12px 16px;
        color: #0b0b0b;
        border: 1px solid rgba(11, 18, 32, 0.04);
        font-weight: 500;
    }

    /* Button styling: subtle glossy pill buttons with hover */
    button.stButton > button {
        background: linear-gradient(180deg, #3578f6 0%, #0d64d6 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        box-shadow: 0 6px 18px rgba(10, 60, 200, 0.12) !important;
        border: none !important;
        transition: transform .12s ease, box-shadow .12s ease, opacity .12s ease;
    }
    button.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(10,60,200,0.16) !important;
        opacity: 0.98;
    }

    /* Secondary (outline) style for disabled / less-prominent actions */
    .stButton button[disabled] {
        background: linear-gradient(180deg,#e6eefb,#f7f9ff) !important;
        color: #7b7f86 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    /* Improve faint / low-contrast text in content area */
    .stApp .stText, .stApp .stMarkdown, .stApp .stDataFrame {
        color: #0b1220 !important;
    }

    /* Metric / KPI cards contrast improvement */
    .card { box-shadow: 0 12px 40px rgba(11,18,32,0.06) !important; }

    /* Make the dashboard headings more visible */
    .stApp .stMarkdown h1, .stApp .stMarkdown h2, .stApp .stMarkdown h3 {
        text-shadow: 0 2px 8px rgba(124,58,237,0.04);
    }

    /* Keep sidebar text colors untouched (so sidebar remains same) */
    [data-testid="stSidebar"] * { color: inherit !important; }

    /* Small responsive tweaks for images and simulated scene area */
    img { max-width: 100% !important; height: auto !important; }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def show_dashboard(go):
    apply_global_styles()
    # then apply the small local overrides to enforce alert colors & button styling
    _inject_local_overrides()

    # Header
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("👁️‍🗨️ SafetyEye — Occupancy & Safety Monitor (MVP)")
        st.markdown("_Demo dashboard using simulated logs or uploaded CSV. No model needed._")
    # # Right header column: use a lightweight text badge instead of external image
    # with col2:
    #     st.markdown("<div style='text-align:right; padding-top:8px; font-weight:700; color:#444;'></div>", unsafe_allow_html=True)

    # st.divider()

    # Sidebar inputs
    st.sidebar.header("Controls & Inputs")
    mode = st.sidebar.radio("Mode", ["Simulated Stream", "Upload CSV / Logs", "Single Image (preview)"])

    st.sidebar.subheader("Zone Thresholds (people)")
    thresholds = default_thresholds()
    for z in list(thresholds.keys()):
        thresholds[z] = st.sidebar.number_input(f"Threshold — {z}", min_value=1, value=thresholds[z], step=1)

    layout_img = st.sidebar.file_uploader("Optional: Upload floor plan / zone map", type=["png","jpg","jpeg"])

    # Model checkbox (you had this; keep it)
    use_model = st.sidebar.checkbox("Enable CV model (placeholder)", value=False)
    if use_model:
        if DETECTOR_AVAILABLE:
            st.sidebar.success("Detector module found — detection enabled when you upload an image below.")
        else:
            st.sidebar.error("Detector module not found or failed to import. Place detector.py and weights in project root.")

    # Confidence slider for detections
    conf_thresh = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1, max_value=0.9, value=0.35, step=0.05
    )

    # If user enabled model and detector exists, offer detection uploader
    detection_uploaded = None
    run_detection = False
    model = None
    if use_model and DETECTOR_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔎 Detection (YOLO)")
        detection_uploaded = st.sidebar.file_uploader("Upload image for detection", type=["jpg", "jpeg", "png"])
        if st.sidebar.button("Run Detection"):
            run_detection = True
        # load model lazily (only when user enabled)
        try:
            model = load_model()
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
            model = None

    # Avatar / people view controls
    st.sidebar.markdown("---")
    avatar_style = st.sidebar.selectbox("Person style", ["Stylized Icons", "Generated Avatars (more realistic)", "Photo thumbnails (upload)"], index=1)
    avatar_size = st.sidebar.slider("Avatar size", min_value=48, max_value=160, value=96, step=8)
    per_row_default = 8
    per_row = st.sidebar.number_input("Avatars per row", min_value=3, max_value=12, value=per_row_default)
    upload_people = None
    if avatar_style == "Photo thumbnails (upload)":
        upload_people = st.sidebar.file_uploader("Upload person images (multiple)", accept_multiple_files=True, type=["png","jpg","jpeg"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("Download a sample CSV to test upload:")
    sample_df = make_sample(180)
    sample_csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download sample CSV", data=sample_csv_bytes, file_name="sample_occupancy.csv", mime="text/csv")

    # ---- Load Data ----
    if mode == "Upload CSV / Logs":
        uploaded = st.file_uploader("Upload CSV (timestamp, zone, people_count)", type=["csv"])
        if uploaded:
            df = safe_load_uploaded_csv(uploaded)
            if not {'timestamp', 'zone', 'people_count'}.issubset(df.columns):
                st.error("CSV must contain columns: timestamp, zone, people_count")
                df = pd.DataFrame(columns=['timestamp', 'zone', 'people_count'])
        else:
            st.info("No file uploaded — using sample data.")
            df = sample_df.copy()
    elif mode == "Single Image (preview)":
        uploaded_img = st.file_uploader("Upload an image (preview only)", type=["png","jpg","jpeg"])
        if uploaded_img:
            try:
                img = Image.open(uploaded_img)
                st.image(img, caption="Preview image", use_container_width=True)
                # If user enabled model and wants detection on this preview, allow button below
                if use_model and DETECTOR_AVAILABLE and model is not None:
                    if st.button("Run detection on preview image"):
                        with st.spinner("Running detection on preview..."):
                            annotated_pil, annotated_bytes, dets = infer_pil(img, model=model, conf_thresh=conf_thresh, return_bytes=True)
                        st.image(annotated_pil, caption="Detection results", use_container_width=True)
                        if dets:
                            st.write("Detections:")
                            for d in dets:
                                st.write(f"{d['class_name']} — {d['conf']:.2f} — bbox: {d['xyxy']}")
                        else:
                            st.info("No detections.")
            except Exception as e:
                st.warning("Could not open image: " + str(e))
        df = make_sample(60)
    else:
        df = sample_df.copy()

    # Clean / enforce types
    if not df.empty:
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception:
                pass
        if 'people_count' in df.columns:
            df['people_count'] = pd.to_numeric(df['people_count'], errors='coerce').fillna(0).astype(int)
        else:
            df['people_count'] = 0
    else:
        df = pd.DataFrame(columns=['timestamp', 'zone', 'people_count'])

    # Zone selection
    zones = sorted(df['zone'].dropna().unique().tolist())
    selected_zones = st.multiselect("Zones to display", options=zones, default=zones if zones else [])
    if selected_zones:
        df = df[df['zone'].isin(selected_zones)].sort_values('timestamp')
    else:
        df = df.sort_values('timestamp')

    # ---- Main layout ----
    left, right = st.columns([2, 1])

    # Keep a place to show detection results on the right side (if any)
    detection_result_placeholder = right.empty()
    detection_info_placeholder = right.empty()

    with left:
        # KPIs
        k1, k2, k3 = st.columns(3)
        if not df.empty:
            latest = df.groupby('zone')['people_count'].last()
        else:
            latest = pd.Series(dtype='int64')

        total_people = int(latest.sum()) if not latest.empty else 0
        thresholds_series = pd.Series(thresholds).reindex(latest.index).fillna(999).astype(int) if not latest.empty else pd.Series(dtype='int64')
        total_violations = int((latest >= thresholds_series).sum()) if not latest.empty else 0
        compliance_rate = 100.0 if latest.empty else max(0.0, 100.0 - (total_violations / len(latest) * 100.0))

        with k1:
            st.markdown(f'<div class="card"><div class="kpi">{total_people} <span class="small">people now</span></div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="card"><div class="kpi">{total_violations} <span class="small">zones over limit</span></div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="card"><div class="kpi">{compliance_rate:.0f}% <span class="small">compliance</span></div></div>', unsafe_allow_html=True)

        st.markdown("### Occupancy — People View")

        if not latest.empty:
            color_palette = [
                (37,99,235), # blue
                (16,185,129), # green
                (234,88,12),  # orange
                (244,63,94),  # pink/red
                (99,102,241), # indigo
                (14,165,233), # cyan
            ]

            for zone in latest.index:
                cnt = int(latest[zone]) if zone in latest.index else 0
                with st.container():
                    st.markdown(f"<div class='zone-title'>{zone} — {cnt} people</div>", unsafe_allow_html=True)
                    if cnt <= 0:
                        st.write("No people detected in this zone.")
                        continue

                    avatars = []
                    for i in range(cnt):
                        if avatar_style == "Photo thumbnails (upload)" and upload_people:
                            f = upload_people[i % len(upload_people)]
                            try:
                                im = Image.open(f).convert('RGBA').resize((avatar_size, avatar_size))
                            except Exception:
                                im = make_flat_avatar(size=avatar_size, fill_color=color_palette[i % len(color_palette)])
                        else:
                            im = make_flat_avatar(size=avatar_size, fill_color=color_palette[i % len(color_palette)])
                        avatars.append(im)

                    for i in range(0, len(avatars), per_row):
                        row = avatars[i:i+per_row]
                        cols = st.columns(len(row))
                        for c, im in zip(cols, row):
                            c.image(im, width=avatar_size)
        else:
            st.info("No timeline / occupancy data available.")

        st.markdown("### Latest Zone Snapshot")
        if not latest.empty:
            snap = latest.reset_index().rename(columns={'people_count':'current_count'})
            snap['threshold'] = snap['zone'].map(thresholds).fillna(999).astype(int)
            snap['status'] = np.where(snap['current_count'] >= snap['threshold'], 'OVER LIMIT', 'OK')
            st.table(snap)
        else:
            st.info("No zone snapshot available (no data).")

        st.markdown("### Violation Log")
        violations_df = generate_violations(df, thresholds)
        if not violations_df.empty:
            st.dataframe(violations_df)
        else:
            # show a styled placeholder with black text (our CSS ensures color)
            st.markdown('<div class="violation-placeholder">No active violations detected.</div>', unsafe_allow_html=True)

    # ---- Right column (controls, live monitor, detection output) ----
    with right:
        if st.button('← Home', key='back_home'):
            go('home')
        st.markdown("### Live Monitor")

        # show layout image if uploaded
        if layout_img:
            try:
                img = Image.open(layout_img)
                st.image(img, caption='Floor plan / zone map', use_container_width=True)
            except Exception:
                st.warning('Could not open layout image')

        st.markdown("### Controls")
        if st.button("▶️ Refresh / Rerun", key='refresh_main'):
            pass

        st.markdown("---")
        # Alerts area with light background and dark text (CSS applied)
        st.markdown('<div class="alerts-area">', unsafe_allow_html=True)
        st.markdown("### Alerts")
        if not violations_df.empty:
            for _, r in violations_df.iterrows():
                # use st.markdown so color inherits alerts-area CSS (we still show st.error lines for severity)
                st.markdown(f"<div style='padding:6px 0; color:#0b0b0b;'>{r['timestamp']} — {r['zone']}: {r['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="violation-placeholder">All zones within thresholds ✅</div>', unsafe_allow_html=True)

        st.markdown("### Quick Actions")
        if not violations_df.empty:
            csv_bytes = violations_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download violation log CSV", data=csv_bytes, file_name="violations.csv", mime="text/csv")
        else:
            st.button("No violations to export", disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------------
        # Detection result area (if detection was requested)
        # -----------------------------
        if use_model and DETECTOR_AVAILABLE:
            # Priority: if user uploaded detection image and clicked Run Detection
            if detection_uploaded and run_detection and model is not None:
                try:
                    pil = Image.open(detection_uploaded).convert("RGB")
                    with st.spinner("Running detection..."):
                        annotated_pil, annotated_bytes, dets = infer_pil(pil, model=model, conf_thresh=conf_thresh, return_bytes=True)

                    # display annotated image and details
                    st.markdown("### Detection Output")
                    st.image(annotated_pil, use_container_width=True)

                    if dets:
                        st.markdown("**Detections**")
                        det_df = pd.DataFrame(dets)
                        st.dataframe(det_df[['class_name','conf','xyxy']])

                        # analyze and persist violations
                        try:
                            summary = analyze_detections(dets)
                        except Exception:
                            summary = {"counts": {}, "violations": [], "total_people": 0}

                        # People KPI for this image
                        st.metric("People (this image)", summary.get('total_people', 0))

                        # Violations (this image)
                        if summary.get('violations'):
                            st.markdown("### Violations (this image)")
                            vdf = pd.DataFrame(summary['violations'])
                            st.dataframe(vdf)

                            # persist to session log
                            st.session_state.setdefault('violation_log', [])
                            st.session_state['violation_log'].extend(summary['violations'])

                            for v in summary['violations']:
                                st.error(f"{v['timestamp']} — {v['class']}: {v['message']} (conf {v['conf']})")
                        else:
                            st.success("No active violations detected ✅")

                        # show full session violation log + download
                        if st.session_state.get('violation_log'):
                            st.markdown("### Full session violation log")
                            st.dataframe(pd.DataFrame(st.session_state['violation_log']))
                            csv_bytes = pd.DataFrame(st.session_state['violation_log']).to_csv(index=False).encode('utf-8')
                            st.download_button("Download violation log CSV", csv_bytes, file_name="violation_log.csv", mime="text/csv")

                        # annotated image download
                        if annotated_bytes:
                            st.download_button("Download annotated image", data=annotated_bytes.getvalue(), file_name="annotated.jpg", mime="image/jpeg")

                    else:
                        st.info("No detections above threshold.")
                except Exception as e:
                    st.error(f"Detection failed: {e}")

            # If user enabled model but hasn't run detection yet, show hint
            elif use_model and model is not None:
                st.info("Upload an image in the sidebar and press 'Run Detection' to see model output here.")

            # If model not loaded
            elif model is None:
                st.warning("Model not loaded. Check detector.py and weight file paths.")

    st.divider()
    st.markdown("**Notes:** This is an MVP dashboard that uses simulated logs or uploaded CSV. To add a CV model later, create an inference function and fill the placeholder `use_model` branch.")

    # ---- Worker Scene Simulator UI ----
    st.markdown("---")
    sim_enabled = st.checkbox("Enable worker scene simulator (preview)")
    if sim_enabled:
        init_sim_state()
        sim_show_ann = st.checkbox("Show annotations (labels & colours)", value=True)
        sim_bg = st.color_picker("Simulator background color", value="#f5f7fa")
        c1, c2, c3 = st.columns([1,1,2])
        st.markdown('<div class="sim-controls">', unsafe_allow_html=True)
        with c1:
            if st.button("Generate Scene"):
                img, cnt = create_worker_scene(show_annotations=sim_show_ann, bg_color=sim_bg)
                st.image(img, caption=f"Simulated scene — {cnt} workers", use_container_width=True)
        with c2:
            if st.button("Clear simulator logs"):
                st.session_state['sim_logs'] = []
                st.session_state['sim_total'] = 0
                st.success("Simulator logs cleared")
        with c3:
            st.metric("Total recorded violations", st.session_state.get('sim_total', 0))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Recent Simulator Logs")
        if st.session_state.get('sim_logs'):
            st.dataframe(pd.DataFrame(st.session_state['sim_logs']).sort_values('time', ascending=False).reset_index(drop=True))
        else:
            st.info("No simulator logs yet. Click 'Generate Scene' to produce activity.")
