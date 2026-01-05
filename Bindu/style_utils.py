import streamlit as st

def apply_global_styles():
    st.markdown(
        """
        <style>

        /* -------------------------------------- */
        /* ROOT COLOR SYSTEM (PREMIUM PALETTE)    */
        /* -------------------------------------- */
        :root {
            --bg: #f3f6fa;
            --sidebar: #111827;
            --card: #ffffff;
            --muted: #6b7280;
            --accent: #0f172a;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --teal: #0ea5a4;
            --purple: #7c3aed;
            --success: #10b981;
            --danger: #ef4444;
            --light-border: rgba(0,0,0,0.08);
        }

        /* Background for entire app */
        body, .stApp {
            font-family: Inter, ui-sans-serif, system-ui;
            background: linear-gradient(180deg, var(--bg), #ffffff);
            color: var(--accent);
        }

        /* Fix main + sidebar backgrounds */
        [data-testid="stAppViewContainer"],
        .stApp .main,
        .stApp .block-container {
            background: transparent !important;
        }
        
        /* ---------------------------- */
        /* PREMIUM SIDEBAR              */
        /* ---------------------------- */
        [data-testid="stSidebar"], .stSidebar {
            background: var(--sidebar) !important;
        }

        .stSidebar, .stSidebar p, .stSidebar span, .stSidebar label {
            color: #e6eef8 !important;
            font-weight: 500;
        }

        /* Sidebar buttons */
        .stSidebar button {
            background: var(--primary) !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            padding: 8px 16px !important;
            transition: 0.25s ease;
        }
        .stSidebar button:hover {
            background: var(--primary-hover) !important;
            transform: scale(1.03);
        }

        /* ---------------------------- */
        /* MAIN BUTTON STYLING          */
        /* ---------------------------- */
        button[kind="secondary"], button[kind="primary"], .stButton>button {
            background: var(--primary) !important;
            color: white !important;
            padding: 8px 18px !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
            transition: 0.25s ease !important;
        }
        button:hover {
            background: var(--primary-hover) !important;
            transform: scale(1.03) !important;
        }

        /* ---------------------------- */
        /* PREMIUM CARDS / KPI BLOCKS   */
        /* ---------------------------- */
        .card {
            background: var(--card);
            padding: 16px;
            border-radius: 14px;
            box-shadow: 0 8px 28px rgba(0,0,0,0.06);
            border: 1px solid var(--light-border);
        }

        .kpi {
            font-size: 24px;
            font-weight: 700;
            color: var(--primary);
        }

        .zone-title {
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 6px;
        }

        /* ---------------------------- */
        /* ALERT AREA (LIGHTER & BLACK) */
        /* ---------------------------- */
        .alerts-area {
            background: #f9fafb !important;
            border: 1px solid var(--light-border) !important;
            padding: 12px !important;
            border-radius: 12px !important;
        }
        .alerts-area, .alerts-area * {
            color: #000 !important;
        }

        /* ---------------------------- */
        /* VIOLATION LOG SECTION        */
        /* ---------------------------- */
        .violation-log-container {
            background: linear-gradient(180deg, #ffffff, #fff7ec) !important;
            border: 1px solid rgba(240,180,90,0.30) !important;
            padding: 14px !important;
            border-radius: 12px !important;
        }

        /* Make all nested text BLACK */
        .violation-log-container,
        .violation-log-container * {
            color: #000000 !important;
            font-weight: 500;
        }

        /* Improve tables inside violation block */
        .violation-log-container .stDataFrame,
        .violation-log-container table {
            color: black !important;
        }

        /* ---------------------------- */
        /* HEADINGS - PREMIUM LOOK      */
        /* ---------------------------- */
        h1, h2, h3 {
            color: var(--purple) !important;
            font-weight: 800 !important;
        }

        p, span, div {
            color: var(--accent);
        }

        /* ---------------------------- */
        /* Avatars & Grid               */
        /* ---------------------------- */
        .avatar-card {
            padding: 4px;
            border-radius: 10px;
        }

        /* ---------------------------- */
        /* Smooth Hover Effects          */
        /* ---------------------------- */
        .stButton>button:active {
            transform: scale(0.97) !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
