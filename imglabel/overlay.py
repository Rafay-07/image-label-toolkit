"""Draw boxes on an image so a reviewer can check labels by eye."""
from PIL import Image, ImageDraw

COLORS = {"pore": (255, 0, 0), "crack": (0, 200, 0), "spatter": (0, 120, 255)}


def draw(image_path, boxes, out_path):
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for b in boxes:
        c = COLORS.get(b["class"], (255, 255, 0))
        d.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=c, width=2)
        d.text((b["x"] + 2, max(0, b["y"] - 10)), b["class"], fill=c)
    im.save(out_path)
