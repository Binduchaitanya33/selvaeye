import os
import sys
import random
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk, ImageColor
import numpy as np

# --- Simulator logic (self-contained copy of the worker-scene generator) ---
def _draw_worker(draw_obj, x, y, color="#2b8cff", vest=False, helmet=False, boots=False, scale=1.6):
    head_r = int(14 * scale)
    torso_w = int(12 * scale)
    torso_h = int(40 * scale)
    torso_top = y + int(12 * scale)
    torso_bottom = torso_top + torso_h
    # shadow
    shadow_w = int(26 * scale)
    draw_obj.ellipse([x - shadow_w, torso_bottom + int(28*scale), x + shadow_w, torso_bottom + int(38*scale)], fill=(220,220,220,160))
    # outline
    try:
        outline = tuple(max(0, int(c) - 30) for c in ImageColor.getrgb(color)) if isinstance(color, str) else None
    except Exception:
        outline = None
    if outline:
        draw_obj.ellipse([x - head_r - 2, y - head_r - 2, x + head_r + 2, y + head_r + 2], fill=outline)
    # head
    draw_obj.ellipse([x - head_r, y - head_r, x + head_r, y + head_r], fill=color)
    # torso
    draw_obj.rectangle([x - torso_w//2, torso_top + int(4*scale), x + torso_w//2, torso_bottom], fill=color)
    draw_obj.ellipse([x - torso_w//2, torso_top, x + torso_w//2, torso_top + int(8*scale)], fill=color)
    # arms
    draw_obj.line([(x - torso_w//2, torso_top + int(8*scale)), (x - int(32*scale), torso_top + int(26*scale))], fill=color, width=int(5*scale))
    draw_obj.line([(x + torso_w//2, torso_top + int(8*scale)), (x + int(32*scale), torso_top + int(26*scale))], fill=color, width=int(5*scale))
    # legs
    draw_obj.line([(x - int(6*scale), torso_bottom), (x - int(16*scale), torso_bottom + int(48*scale))], fill=color, width=int(6*scale))
    draw_obj.line([(x + int(6*scale), torso_bottom), (x + int(16*scale), torso_bottom + int(48*scale))], fill=color, width=int(6*scale))
    # vest
    if vest:
        vest_color = "#ffb703"
        draw_obj.polygon([
            (x - int(8*scale), torso_top + int(2*scale)),
            (x, torso_top + int(10*scale)),
            (x + int(8*scale), torso_top + int(2*scale)),
            (x + int(6*scale), torso_bottom - int(8*scale)),
            (x - int(6*scale), torso_bottom - int(8*scale)),
        ], fill=vest_color)
    # helmet & boots
    if helmet:
        draw_obj.rectangle([x - int(18*scale), y - int(20*scale), x + int(18*scale), y - int(12*scale)], fill="#0f172a")
    if boots:
        boot_color = "#0f172a"
        draw_obj.rectangle([x - int(18*scale), torso_bottom + int(36*scale), x - int(6*scale), torso_bottom + int(48*scale)], fill=boot_color)
        draw_obj.rectangle([x + int(6*scale), torso_bottom + int(36*scale), x + int(18*scale), torso_bottom + int(48*scale)], fill=boot_color)

def create_worker_scene(show_annotations=True, canvas_size=(900, 600)):
    w, h = canvas_size
    img = Image.new("RGB", canvas_size, (245, 247, 250))
    draw = ImageDraw.Draw(img)
    worker_count = int(np.random.randint(1, 8))
    logs = []
    for idx in range(worker_count):
        x = int(np.random.randint(60, w - 120))
        y = int(np.random.randint(80, h - 220))
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
        _draw_worker(draw, x, y, color=main_color, vest=has_vest, helmet=has_helmet, boots=has_boots, scale=1.6)
        if show_annotations:
            label = f"W{idx+1}: {severity}"
            draw.text((x - 22, y - 26), label, fill="#0f172a")
        if violation != "None":
            logs.append({"time": datetime.now().strftime('%H:%M:%S'), "worker": f"W{idx+1}", "violation": violation, "severity": severity})
    return img, worker_count, logs

 # --- Tkinter GUI ---
class SimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SafetyEye Simulator — Executable Demo")
        self.geometry("1100x700")
        self.resizable(False, False)
        self.logs = []

        # Controls
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)
        self.gen_btn = ttk.Button(control_frame, text="Generate Scene", command=self.generate_scene)
        self.gen_btn.pack(side=tk.LEFT)
        self.clear_btn = ttk.Button(control_frame, text="Clear Logs", command=self.clear_logs)
        self.clear_btn.pack(side=tk.LEFT, padx=6)
        self.count_label = ttk.Label(control_frame, text="Total violations: 0")
        self.count_label.pack(side=tk.LEFT, padx=12)

        # Canvas and logs
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.canvas_label = ttk.Label(main_frame)
        self.canvas_label.pack(side=tk.LEFT, padx=6)

        log_frame = ttk.Frame(main_frame)
        log_frame.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(log_frame, text="Recent Logs").pack()
        self.logbox = tk.Listbox(log_frame, width=40, height=30)
        self.logbox.pack(side=tk.TOP, fill=tk.Y, padx=4)

    def generate_scene(self):
        img, cnt, logs = create_worker_scene(show_annotations=True, canvas_size=(900,600))
        self.imgtk = ImageTk.PhotoImage(img)
        self.canvas_label.configure(image=self.imgtk)
        for l in logs:
            self.logs.append(l)
            self.logbox.insert(0, f"{l['time']} {l['worker']} {l['violation']} ({l['severity']})")
        self.count_label.config(text=f"Total violations: {len(self.logs)}")

    def clear_logs(self):
        self.logs = []
        self.logbox.delete(0, tk.END)
        self.count_label.config(text="Total violations: 0")

def main():
    app = SimulatorApp()
    app.mainloop()

if __name__ == '__main__':
    main()
