import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from imglabel import agreement, cli, dataset, split, synth


def box(c, x, y, w, h):
    return {"class": c, "x": x, "y": y, "w": w, "h": h}


def test_iou_known_values():
    assert agreement.iou(box("a", 0, 0, 10, 10), box("a", 0, 0, 10, 10)) == 1.0
    assert agreement.iou(box("a", 0, 0, 10, 10), box("a", 20, 20, 5, 5)) == 0.0
    # 10x10 and 10x10 shifted by 5 in x: intersection 50, union 150
    assert abs(agreement.iou(box("a", 0, 0, 10, 10), box("a", 5, 0, 10, 10)) - 1 / 3) < 1e-9


def test_match_is_one_to_one_and_class_aware():
    a = [box("pore", 0, 0, 10, 10)]
    b = [box("crack", 0, 0, 10, 10), box("pore", 1, 1, 10, 10)]
    m, ua, ub = agreement.match(a, b)
    assert [(i, j) for i, j, _ in m] == [(0, 1)] and ua == [] and ub == [0]


def test_agreement_scores():
    a = {"i": [box("pore", 0, 0, 10, 10), box("crack", 50, 50, 10, 10)]}
    b = {"i": [box("pore", 0, 0, 10, 10), box("spatter", 90, 90, 5, 5)]}
    r = agreement.agreement(a, b)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1) and r["f1"] == 0.5


def test_index_finds_duplicates_and_corrupt(tmp_path):
    synth.generate(str(tmp_path), n=3)
    (tmp_path / "copy.png").write_bytes((tmp_path / "img_000.png").read_bytes())
    (tmp_path / "bad.png").write_bytes(b"not an image")
    ds = dataset.Dataset()
    n, corrupt = ds.index(str(tmp_path))
    assert n == 5 and corrupt == ["bad.png"]
    assert ds.duplicates() == [["copy.png", "img_000.png"]]


def test_validate_annotations(tmp_path):
    synth.generate(str(tmp_path), n=1)
    ds = dataset.Dataset(); ds.index(str(tmp_path))
    ann = {"img_000.png": [box("weird", 0, 0, 5, 5), box("pore", 0, 0, 0, 5), box("pore", 155, 0, 20, 5)],
           "ghost.png": []}
    probs = dataset.validate_annotations(ds, ann, synth.CLASSES)
    assert len(probs) == 4 and any("outside" in p for p in probs) and any("ghost" in p for p in probs)


def test_synthetic_annotations_are_valid(tmp_path):
    ann = synth.generate(str(tmp_path), n=8)
    ds = dataset.Dataset(); ds.index(str(tmp_path))
    assert dataset.validate_annotations(ds, ann, synth.CLASSES) == []


def test_split_is_stable_and_roughly_sized():
    names = [f"img_{i}.png" for i in range(500)]
    tr, va = split.split(names, 0.2)
    assert tr + va and set(tr).isdisjoint(va) and 70 < len(va) < 130
    tr2, va2 = split.split(names + ["new.png"], 0.2)
    assert set(va) <= set(va2)           # adding an image never moves an old one


def test_cli_roundtrip(tmp_path, capsys):
    cli.main(["synth", str(tmp_path), "-n", "4"])
    cli.main(["check", str(tmp_path), str(tmp_path / "annotations.json")])
    cli.main(["overlay", str(tmp_path), str(tmp_path / "annotations.json"), str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "0 label problems" in out and (tmp_path / "out" / "img_000.png").exists()
