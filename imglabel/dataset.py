"""Dataset index in SQLite: file hash, size, duplicates, corrupt files, and annotation validation.

The idea: before any model sees the data, check that the files are healthy and the labels make sense.
"""
import hashlib
import os
import sqlite3

from PIL import Image


def sha256(path):
    """Fingerprint of a file's contents. Two files with the same fingerprint are exact duplicates."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in 64 KB chunks so a very large image never has to fit in memory all at once.
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Dataset:
    """A small SQLite table with one row per image: name, fingerprint, size, and whether it opened fine."""

    def __init__(self, db_path=":memory:"):
        # ":memory:" keeps the database in RAM (good for tests). Pass a file path to keep it on disk.
        self.db = sqlite3.connect(db_path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS images(
            name TEXT PRIMARY KEY, sha TEXT, width INT, height INT, ok INT)""")

    def index(self, folder):
        """Scan a folder. Returns (indexed, corrupt_names)."""
        corrupt = []
        count = 0
        for name in sorted(os.listdir(folder)):
            # Skip anything that is not an image file.
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                continue
            path = os.path.join(folder, name)
            try:
                with Image.open(path) as im:
                    im.verify()                      # catches truncated or damaged files
                # verify() leaves the image unusable, so open it again to read the size.
                with Image.open(path) as im:
                    w, h = im.size
                ok = 1
            except Exception:
                # If the file cannot be opened it counts as corrupt. Record it instead of crashing.
                w = h = 0
                ok = 0
                corrupt.append(name)
            # INSERT OR REPLACE means re-running the scan updates old rows instead of duplicating them.
            self.db.execute("INSERT OR REPLACE INTO images VALUES (?,?,?,?,?)",
                            (name, sha256(path), w, h, ok))
            count += 1
        self.db.commit()
        return count, corrupt

    def duplicates(self):
        """Groups of names that share identical file content."""
        # GROUP BY the fingerprint; any fingerprint that appears more than once is a duplicate group.
        rows = self.db.execute(
            "SELECT sha, GROUP_CONCAT(name) FROM images GROUP BY sha HAVING COUNT(*) > 1").fetchall()
        return [sorted(names.split(",")) for _, names in rows]

    def size_of(self, name):
        """(width, height) of an image, or None if the image is not in the index."""
        row = self.db.execute("SELECT width, height FROM images WHERE name=?", (name,)).fetchone()
        return row


def validate_annotations(ds, annotations, classes):
    """Return a list of problems: unknown image, unknown class, empty or out-of-bounds box."""
    problems = []
    for name, boxes in annotations.items():
        size = ds.size_of(name)
        if size is None:
            # The label file mentions an image that is not in the folder.
            problems.append(f"{name}: image not in dataset")
            continue
        w, h = size
        for i, b in enumerate(boxes):
            # Check 1: the class name must be one we know about (catches typos like "proe").
            if b["class"] not in classes:
                problems.append(f"{name} box {i}: unknown class '{b['class']}'")
            # Check 2: a box with zero or negative width or height is not a real box.
            if b["w"] <= 0 or b["h"] <= 0:
                problems.append(f"{name} box {i}: non-positive size")
            # Check 3: the box must sit fully inside the image.
            elif b["x"] < 0 or b["y"] < 0 or b["x"] + b["w"] > w or b["y"] + b["h"] > h:
                problems.append(f"{name} box {i}: outside the {w}x{h} image")
    return problems
