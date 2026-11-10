"""Dataset index in SQLite: file hash, size, duplicates, corrupt files, and annotation validation."""
import hashlib
import os
import sqlite3

from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Dataset:
    def __init__(self, db_path=":memory:"):
        self.db = sqlite3.connect(db_path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS images(
            name TEXT PRIMARY KEY, sha TEXT, width INT, height INT, ok INT)""")

    def index(self, folder):
        """Scan a folder. Returns (indexed, corrupt_names)."""
        corrupt = []
        count = 0
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                continue
            path = os.path.join(folder, name)
            try:
                with Image.open(path) as im:
                    im.verify()                      # catches truncated or damaged files
                with Image.open(path) as im:
                    w, h = im.size
                ok = 1
            except Exception:
                w = h = 0
                ok = 0
                corrupt.append(name)
            self.db.execute("INSERT OR REPLACE INTO images VALUES (?,?,?,?,?)",
                            (name, sha256(path), w, h, ok))
            count += 1
        self.db.commit()
        return count, corrupt

    def duplicates(self):
        """Groups of names that share identical file content."""
        rows = self.db.execute(
            "SELECT sha, GROUP_CONCAT(name) FROM images GROUP BY sha HAVING COUNT(*) > 1").fetchall()
        return [sorted(names.split(",")) for _, names in rows]

    def size_of(self, name):
        row = self.db.execute("SELECT width, height FROM images WHERE name=?", (name,)).fetchone()
        return row


def validate_annotations(ds, annotations, classes):
    """Return a list of problems: unknown image, unknown class, empty or out-of-bounds box."""
    problems = []
    for name, boxes in annotations.items():
        size = ds.size_of(name)
        if size is None:
            problems.append(f"{name}: image not in dataset")
            continue
        w, h = size
        for i, b in enumerate(boxes):
            if b["class"] not in classes:
                problems.append(f"{name} box {i}: unknown class '{b['class']}'")
            if b["w"] <= 0 or b["h"] <= 0:
                problems.append(f"{name} box {i}: non-positive size")
            elif b["x"] < 0 or b["y"] < 0 or b["x"] + b["w"] > w or b["y"] + b["h"] > h:
                problems.append(f"{name} box {i}: outside the {w}x{h} image")
    return problems
