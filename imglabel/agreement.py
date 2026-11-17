"""Labeller agreement: how well do two sets of boxes match? Uses IoU (intersection over union).

Why this matters: if two people label the same images, their boxes should mostly line up.
If they don't, the labels are unreliable and a model trained on them will learn the wrong thing.
"""

# A "box" in this project is a small dictionary:
#   {"class": "pore", "x": 10, "y": 12, "w": 20, "h": 18}
# x and y are the top-left corner in pixels, w and h are the width and height.


def iou(a, b):
    """Overlap score between two boxes, from 0.0 (no overlap) to 1.0 (identical).

    IoU = area where the boxes overlap / total area covered by both boxes.
    """
    # Right and bottom edges of each box (x + width, y + height).
    ax2, ay2, bx2, by2 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
    # Width and height of the overlapping rectangle. max(0, ...) means "no overlap" gives 0, not a negative number.
    iw = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih                                   # overlap area
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter  # both areas added, minus the overlap counted twice
    return inter / union if union else 0.0            # guard against dividing by zero


def match(boxes_a, boxes_b, threshold=0.5):
    """Greedy one-to-one matching of same-class boxes, best IoU first.
    Returns (matches, unmatched_a, unmatched_b) as index lists."""
    # Step 1: list every possible pair of boxes that have the same class and overlap enough.
    pairs = []
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            if a["class"] == b["class"]:
                v = iou(a, b)
                if v >= threshold:
                    pairs.append((v, i, j))
    # Step 2: sort so the best overlaps come first.
    pairs.sort(reverse=True)
    # Step 3: walk down the list and accept a pair only if neither box has already been matched.
    # This is what makes it "one-to-one": each box can be used at most once.
    used_a, used_b, matches = set(), set(), []
    for v, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); matches.append((i, j, v))
    # Whatever is left over has no partner: boxes only labeller A drew, and boxes only labeller B drew.
    return (matches,
            [i for i in range(len(boxes_a)) if i not in used_a],
            [j for j in range(len(boxes_b)) if j not in used_b])


def agreement(ann_a, ann_b, threshold=0.5):
    """Precision, recall and F1 of labeller B against labeller A, over all images."""
    # tp = matched boxes, fp = boxes only B drew, fn = boxes only A drew.
    tp = fp = fn = 0
    # set(ann_a) | set(ann_b) is every image name that either labeller touched.
    for name in set(ann_a) | set(ann_b):
        m, ua, ub = match(ann_a.get(name, []), ann_b.get(name, []), threshold)
        tp += len(m); fn += len(ua); fp += len(ub)
    # Precision: of the boxes B drew, how many did A also draw?
    p = tp / (tp + fp) if tp + fp else 0.0
    # Recall: of the boxes A drew, how many did B find?
    r = tp / (tp + fn) if tp + fn else 0.0
    # F1 is a single score that balances precision and recall.
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}


def class_counts(ann):
    """Count how many boxes there are of each class (useful for spotting a lopsided dataset)."""
    counts = {}
    for boxes in ann.values():
        for b in boxes:
            counts[b["class"]] = counts.get(b["class"], 0) + 1
    return counts
