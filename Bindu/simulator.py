from PIL import Image, ImageDraw, ImageColor
import numpy as np
from datetime import datetime
import streamlit as st


def init_sim_state():
    if 'sim_logs' not in st.session_state:
        st.session_state['sim_logs'] = []
    if 'sim_total' not in st.session_state:
        st.session_state['sim_total'] = 0


def _draw_worker(draw_obj, x, y, color="#2b8cff", vest=False, helmet=False, boots=False, scale=1.6):
    head_r = int(14 * scale)
    torso_w = int(12 * scale)
    torso_h = int(40 * scale)
    torso_top = y + int(12 * scale)
    torso_bottom = torso_top + torso_h

    shadow_w = int(26 * scale)
    draw_obj.ellipse([x - shadow_w, torso_bottom + int(28*scale), x + shadow_w, torso_bottom + int(38*scale)], fill=(220,220,220,160))

    outline = tuple(max(0, int(c) - 30) for c in ImageColor.getrgb(color)) if isinstance(color, str) else None
    if outline:
        draw_obj.ellipse([x - head_r - 2, y - head_r - 2, x + head_r + 2, y + head_r + 2], fill=outline)

    draw_obj.ellipse([x - head_r, y - head_r, x + head_r, y + head_r], fill=color)

    draw_obj.rectangle([x - torso_w//2, torso_top + int(4*scale), x + torso_w//2, torso_bottom], fill=color)
    draw_obj.ellipse([x - torso_w//2, torso_top, x + torso_w//2, torso_top + int(8*scale)], fill=color)

    draw_obj.line([(x - torso_w//2, torso_top + int(8*scale)), (x - int(32*scale), torso_top + int(26*scale))], fill=color, width=int(5*scale))
    draw_obj.line([(x + torso_w//2, torso_top + int(8*scale)), (x + int(32*scale), torso_top + int(26*scale))], fill=color, width=int(5*scale))

    draw_obj.line([(x - int(6*scale), torso_bottom), (x - int(16*scale), torso_bottom + int(48*scale))], fill=color, width=int(6*scale))
    draw_obj.line([(x + int(6*scale), torso_bottom), (x + int(16*scale), torso_bottom + int(48*scale))], fill=color, width=int(6*scale))

    if vest:
        vest_color = "#ffb703"
        draw_obj.polygon([
            (x - int(8*scale), torso_top + int(2*scale)),
            (x, torso_top + int(10*scale)),
            (x + int(8*scale), torso_top + int(2*scale)),
            (x + int(6*scale), torso_bottom - int(8*scale)),
            (x - int(6*scale), torso_bottom - int(8*scale)),
        ], fill=vest_color)

    if helmet:
        draw_obj.rectangle([x - int(18*scale), y - int(20*scale), x + int(18*scale), y - int(12*scale)], fill="#0f172a")

    if boots:
        boot_color = "#3e4657"
        draw_obj.rectangle([x - int(18*scale), torso_bottom + int(36*scale), x - int(6*scale), torso_bottom + int(48*scale)], fill=boot_color)
        draw_obj.rectangle([x + int(6*scale), torso_bottom + int(36*scale), x + int(18*scale), torso_bottom + int(48*scale)], fill=boot_color)


def create_worker_scene(show_annotations=True, canvas_size=(700, 450), bg_color="#f5f7fa"):
    init_sim_state()
    w, h = canvas_size
    try:
        bg_rgb = ImageColor.getrgb(bg_color) if isinstance(bg_color, str) else tuple(bg_color)
    except Exception:
        bg_rgb = (245, 247, 250)
    img = Image.new("RGB", canvas_size, bg_rgb)
    draw = ImageDraw.Draw(img)

    worker_count = int(np.random.randint(1, 8))

    for idx in range(worker_count):
        x = int(np.random.randint(40, w - 80))
        y = int(np.random.randint(60, h - 160))

        violation = np.random.choice(["None", "No Helmet", "No High-Vis", "No Boots", "Distracted"], p=[0.50, 0.22, 0.15, 0.08, 0.05])

        if violation == "None":
            severity = "Safe"
            main_color = "#10b981"
        elif violation == "No Helmet":
            severity = "Critical"
            main_color = "#ef4444"
        elif violation == "Distracted":
            severity = "Notice"
            main_color = "#f97316"
        else:
            severity = "Warning"
            main_color = "#f59e0b"

        has_helmet = (violation != "No Helmet")
        has_vest = (violation != "No High-Vis")
        has_boots = (violation != "No Boots")

        _draw_worker(draw, x, y, color=main_color, vest=has_vest, helmet=has_helmet, boots=has_boots, scale=1.0)

        if show_annotations:
            label = f"W{idx+1}: {severity}"
            draw.text((x - 18, y - 22), label, fill="#0f172a")

        if violation != "None":
            st.session_state['sim_logs'].append({
                "time": datetime.now().strftime('%H:%M:%S'),
                "worker": f"W{idx+1}",
                "violation": violation,
                "severity": severity
            })
            st.session_state['sim_total'] += 1

    return img, worker_count
