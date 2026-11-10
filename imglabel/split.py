"""Deterministic train/validation split. The same image always lands in the same set,
even when new images are added later, because the choice depends on a hash of the name."""
import hashlib


def split(names, val_fraction=0.2):
    train, val = [], []
    for n in sorted(names):
        bucket = int(hashlib.sha256(n.encode()).hexdigest(), 16) % 1000 / 1000
        (val if bucket < val_fraction else train).append(n)
    return train, val
