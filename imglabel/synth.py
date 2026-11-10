"""Make synthetic grayscale images with bright blob 'defects' and ground-truth boxes."""
import json
import os
import random

import numpy as np
from PIL import Image

CLASSES = ["pore", "crack", "spatter"]


def make_image(seed, size=(160, 120)):
    rng = random.Random(seed)
    w, h = size
    img = np.full((h, w), 60, dtype=np.uint8)
    img += np.array([[rng.randint(0, 12) for _ in range(w)] for _ in range(h)], dtype=np.uint8)
    boxes = []
    for _ in range(rng.randint(1, 4)):
        bw, bh = rng.randint(8, 30), rng.randint(8, 30)
        x, y = rng.randint(0, w - bw - 1), rng.randint(0, h - bh - 1)
        img[y:y + bh, x:x + bw] = 220
        boxes.append({"class": rng.choice(CLASSES), "x": x, "y": y, "w": bw, "h": bh})
    return Image.fromarray(img), boxes


def generate(outdir, n=12, seed=0):
    os.makedirs(outdir, exist_ok=True)
    ann = {}
    for i in range(n):
        name = f"img_{i:03d}.png"
        im, boxes = make_image(seed * 1000 + i)
        im.save(os.path.join(outdir, name))
        ann[name] = boxes
    with open(os.path.join(outdir, "annotations.json"), "w") as f:
        json.dump(ann, f, indent=1)
    return ann
