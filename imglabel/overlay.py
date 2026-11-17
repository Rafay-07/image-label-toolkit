"""Draw boxes on an image so a reviewer can check labels by eye."""
from PIL import Image, ImageDraw

# One colour per class so different defect types are easy to tell apart. Values are (red, green, blue).
COLORS = {"pore": (255, 0, 0), "crack": (0, 200, 0), "spatter": (0, 120, 255)}


def draw(image_path, boxes, out_path):
    """Save a copy of the image with each labelled box outlined and its class name written above it."""
    im = Image.open(image_path).convert("RGB")   # RGB so coloured outlines work on grayscale images
    d = ImageDraw.Draw(im)
    for b in boxes:
        c = COLORS.get(b["class"], (255, 255, 0))   # unknown classes get yellow
        d.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=c, width=2)
        # max(0, ...) keeps the label text from being drawn off the top edge of the image.
        d.text((b["x"] + 2, max(0, b["y"] - 10)), b["class"], fill=c)
    im.save(out_path)
