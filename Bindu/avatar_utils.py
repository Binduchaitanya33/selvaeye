import random
import numpy as np
from PIL import Image, ImageDraw


def make_flat_avatar(size=96, fill_color=(59,130,246)):
    img = Image.new('RGBA', (size, size), (255,255,255,0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 3
    head_r = int(size * 0.20)
    head_color = (249,250,251)
    draw.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r], fill=head_color)
    body_top = cy + head_r + int(size * 0.02)
    body_bottom = size - int(size * 0.06)
    body_w = int(size * 0.44)
    draw.rectangle([cx - body_w//2, body_top, cx + body_w//2, body_bottom], fill=fill_color)
    return img


def make_person_icon(size=80, color=(70,130,180)):
    return make_flat_avatar(size=size, fill_color=color)


def make_realistic_avatar(size=112, skin_tone=None, hair_color=None, shirt_color=None):
    img = Image.new('RGBA', (size, size), (255,255,255,0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 3
    head_r = int(size * 0.22)

    skin_options = [(244,194,157),(229,194,152),(198,134,66),(142,85,42),(98,60,34),(230,190,150),(210,170,120)]
    shirt_options = [(16,185,129),(59,130,246),(239,68,68),(234,179,8),(99,102,241),(236,72,153),(14,165,233),(34,197,94)]

    skin = tuple(skin_tone) if skin_tone else random.choice(skin_options)
    shirt = tuple(shirt_color) if shirt_color else random.choice(shirt_options)

    draw.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r], fill=skin)

    body_top = cy + head_r + int(size*0.02)
    body_bottom = size - int(size*0.06)
    body_w = int(size * 0.42)
    draw.rectangle([cx - body_w//2, body_top, cx + body_w//2, body_bottom], fill=shirt)

    return img


def random_choice(seq):
    return seq[int(np.random.randint(0, len(seq)))]
