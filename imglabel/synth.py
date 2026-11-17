"""Make synthetic grayscale images with bright blob 'defects' and ground-truth boxes.

Why synthetic data? Everything in this toolkit can be run and tested without real images.
Because we draw the defects ourselves, we know exactly where the correct boxes are.
"""
import json
import os
import random

import numpy as np
from PIL import Image

# Placeholder defect types. Change these to match your own data.
CLASSES = ["pore", "crack", "spatter"]


def make_image(seed, size=(160, 120)):
    """Build one image and its correct boxes. The same seed always gives the same image."""
    rng = random.Random(seed)
    w, h = size
    # Start with a flat grey background (value 60 out of 255).
    img = np.full((h, w), 60, dtype=np.uint8)
    # Add a little random noise so the image looks less perfectly flat.
    img += np.array([[rng.randint(0, 12) for _ in range(w)] for _ in range(h)], dtype=np.uint8)
    boxes = []
    for _ in range(rng.randint(1, 4)):                  # 1 to 4 "defects" per image
        bw, bh = rng.randint(8, 30), rng.randint(8, 30)  # random defect size
        x, y = rng.randint(0, w - bw - 1), rng.randint(0, h - bh - 1)  # random spot that stays inside the image
        img[y:y + bh, x:x + bw] = 220                    # paint a bright rectangle
        boxes.append({"class": rng.choice(CLASSES), "x": x, "y": y, "w": bw, "h": bh})
    return Image.fromarray(img), boxes


def generate(outdir, n=12, seed=0):
    """Write n images plus an annotations.json file with the correct boxes into outdir."""
    os.makedirs(outdir, exist_ok=True)
    ann = {}
    for i in range(n):
        name = f"img_{i:03d}.png"                      # img_000.png, img_001.png, ...
        im, boxes = make_image(seed * 1000 + i)        # different seed per image, but repeatable
        im.save(os.path.join(outdir, name))
        ann[name] = boxes
    with open(os.path.join(outdir, "annotations.json"), "w") as f:
        json.dump(ann, f, indent=1)
    return ann
