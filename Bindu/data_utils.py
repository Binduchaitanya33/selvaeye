import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@st.cache_data
def make_sample(periods=120):
    now = datetime.now().replace(second=0, microsecond=0)
    times = [now - timedelta(minutes=(periods - 1 - i)) for i in range(periods)]
    rows = []
    for t in times:
        rows.append({"timestamp": t, "zone": "Zone A", "people_count": max(0, 6 + int(3*np.random.randn()))})
        rows.append({"timestamp": t, "zone": "Zone B", "people_count": max(0, 3 + int(2*np.random.randn()))})
        rows.append({"timestamp": t, "zone": "Restricted", "people_count": int(np.random.binomial(1, 0.03))})
    return pd.DataFrame(rows)


def safe_load_uploaded_csv(uploaded_file):
    try:
        dfu = pd.read_csv(uploaded_file)
        if 'timestamp' in dfu.columns:
            dfu['timestamp'] = pd.to_datetime(dfu['timestamp'], errors='coerce')
        return dfu
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        return pd.DataFrame(columns=['timestamp', 'zone', 'people_count'])


def default_thresholds():
    return {"Zone A": 10, "Zone B": 8, "Restricted": 2}


def generate_violations(data, thr_map):
    rows = []
    if data.empty:
        return pd.DataFrame(rows)
    grouped = data.groupby('zone')['people_count'].last().reset_index()
    for _, r in grouped.iterrows():
        zone = r['zone']
        cnt = int(r['people_count'])
        thr = thr_map.get(zone, 999)
        if cnt >= thr:
            rows.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'zone': zone,
                'people': cnt,
                'message': f'Occupancy {cnt} >= threshold {thr}'
            })
    return pd.DataFrame(rows)
