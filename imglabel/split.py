"""Deterministic train/validation split. The same image always lands in the same set,
even when new images are added later, because the choice depends on a hash of the name.

Why not just shuffle randomly? A random shuffle would move old images between sets every time
new ones arrive, so a model could end up "seeing" validation images during training.
"""
import hashlib


def split(names, val_fraction=0.2):
    """Return (train_names, val_names). val_fraction=0.2 means about 20% go to validation."""
    train, val = [], []
    for n in sorted(names):
        # Turn the image name into a fixed number between 0.000 and 0.999.
        # The same name always gives the same number, so the decision never changes.
        bucket = int(hashlib.sha256(n.encode()).hexdigest(), 16) % 1000 / 1000
        # If the number is under the cutoff the image is validation, otherwise it is training.
        (val if bucket < val_fraction else train).append(n)
    return train, val
