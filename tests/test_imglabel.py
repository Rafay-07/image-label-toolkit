"""Tests for the image-label-toolkit. Run with: python -m pytest -q

Each test checks one behaviour. Where possible the expected answer is worked out by hand.
"""
import json, os, sys
# Let the tests import the imglabel package from the folder above this one.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from imglabel import agreement, cli, dataset, split, synth


def box(c, x, y, w, h):
    """Shortcut to build a box dictionary so the tests stay short."""
    return {"class": c, "x": x, "y": y, "w": w, "h": h}


def test_iou_known_values():
    # Same box twice: perfect overlap, so IoU is 1.0. Far apart: no overlap, so 0.0.
    assert agreement.iou(box("a", 0, 0, 10, 10), box("a", 0, 0, 10, 10)) == 1.0
    assert agreement.iou(box("a", 0, 0, 10, 10), box("a", 20, 20, 5, 5)) == 0.0
    # 10x10 and 10x10 shifted by 5 in x: intersection 50, union 150
    assert abs(agreement.iou(box("a", 0, 0, 10, 10), box("a", 5, 0, 10, 10)) - 1 / 3) < 1e-9


def test_match_is_one_to_one_and_class_aware():
    # B has a box with the wrong class on top of A's box, and a pore box slightly off.
    # The wrong-class box must be ignored and the pore box must be the match.
    a = [box("pore", 0, 0, 10, 10)]
    b = [box("crack", 0, 0, 10, 10), box("pore", 1, 1, 10, 10)]
    m, ua, ub = agreement.match(a, b)
    assert [(i, j) for i, j, _ in m] == [(0, 1)] and ua == [] and ub == [0]


def test_agreement_scores():
    # One box matches, one is only in A (missed by B), one is only in B (extra).
    # So tp=1, fp=1, fn=1, giving precision 0.5, recall 0.5 and F1 0.5.
    a = {"i": [box("pore", 0, 0, 10, 10), box("crack", 50, 50, 10, 10)]}
    b = {"i": [box("pore", 0, 0, 10, 10), box("spatter", 90, 90, 5, 5)]}
    r = agreement.agreement(a, b)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1) and r["f1"] == 0.5


def test_index_finds_duplicates_and_corrupt(tmp_path):
    # Make 3 real images, then add an exact copy and a fake "image" that is really just text.
    synth.generate(str(tmp_path), n=3)
    (tmp_path / "copy.png").write_bytes((tmp_path / "img_000.png").read_bytes())
    (tmp_path / "bad.png").write_bytes(b"not an image")
    ds = dataset.Dataset()
    n, corrupt = ds.index(str(tmp_path))
    assert n == 5 and corrupt == ["bad.png"]
    assert ds.duplicates() == [["copy.png", "img_000.png"]]


def test_validate_annotations(tmp_path):
    # Four deliberate mistakes: unknown class, zero width, box off the image edge, and a missing image.
    synth.generate(str(tmp_path), n=1)
    ds = dataset.Dataset(); ds.index(str(tmp_path))
    ann = {"img_000.png": [box("weird", 0, 0, 5, 5), box("pore", 0, 0, 0, 5), box("pore", 155, 0, 20, 5)],
           "ghost.png": []}
    probs = dataset.validate_annotations(ds, ann, synth.CLASSES)
    assert len(probs) == 4 and any("outside" in p for p in probs) and any("ghost" in p for p in probs)


def test_synthetic_annotations_are_valid(tmp_path):
    # The sample data we generate should itself pass every check.
    ann = synth.generate(str(tmp_path), n=8)
    ds = dataset.Dataset(); ds.index(str(tmp_path))
    assert dataset.validate_annotations(ds, ann, synth.CLASSES) == []


def test_split_is_stable_and_roughly_sized():
    names = [f"img_{i}.png" for i in range(500)]
    tr, va = split.split(names, 0.2)
    # Nothing is in both sets, and about 20% of 500 (roughly 100) land in validation.
    assert tr + va and set(tr).isdisjoint(va) and 70 < len(va) < 130
    tr2, va2 = split.split(names + ["new.png"], 0.2)
    assert set(va) <= set(va2)           # adding an image never moves an old one


def test_cli_roundtrip(tmp_path, capsys):
    # Run the real commands end to end: make data, check it, then draw overlays.
    cli.main(["synth", str(tmp_path), "-n", "4"])
    cli.main(["check", str(tmp_path), str(tmp_path / "annotations.json")])
    cli.main(["overlay", str(tmp_path), str(tmp_path / "annotations.json"), str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "0 label problems" in out and (tmp_path / "out" / "img_000.png").exists()
