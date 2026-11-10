"""Labeller agreement: how well do two sets of boxes match? Uses IoU (intersection over union)."""


def iou(a, b):
    ax2, ay2, bx2, by2 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0


def match(boxes_a, boxes_b, threshold=0.5):
    """Greedy one-to-one matching of same-class boxes, best IoU first.
    Returns (matches, unmatched_a, unmatched_b) as index lists."""
    pairs = []
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            if a["class"] == b["class"]:
                v = iou(a, b)
                if v >= threshold:
                    pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_a, used_b, matches = set(), set(), []
    for v, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); matches.append((i, j, v))
    return (matches,
            [i for i in range(len(boxes_a)) if i not in used_a],
            [j for j in range(len(boxes_b)) if j not in used_b])


def agreement(ann_a, ann_b, threshold=0.5):
    """Precision, recall and F1 of labeller B against labeller A, over all images."""
    tp = fp = fn = 0
    for name in set(ann_a) | set(ann_b):
        m, ua, ub = match(ann_a.get(name, []), ann_b.get(name, []), threshold)
        tp += len(m); fn += len(ua); fp += len(ub)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}


def class_counts(ann):
    counts = {}
    for boxes in ann.values():
        for b in boxes:
            counts[b["class"]] = counts.get(b["class"], 0) + 1
    return counts
