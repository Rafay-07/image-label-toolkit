"""Command line: python -m imglabel.cli <command> ..."""
import argparse
import json
import os

from . import agreement, dataset, overlay, split, synth


def main(argv=None):
    ap = argparse.ArgumentParser(prog="imglabel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth"); s.add_argument("folder"); s.add_argument("-n", type=int, default=12)
    s = sub.add_parser("check"); s.add_argument("folder"); s.add_argument("annotations")
    s = sub.add_parser("agree"); s.add_argument("a"); s.add_argument("b"); s.add_argument("--iou", type=float, default=0.5)
    s = sub.add_parser("split"); s.add_argument("folder"); s.add_argument("--val", type=float, default=0.2)
    s = sub.add_parser("overlay"); s.add_argument("folder"); s.add_argument("annotations"); s.add_argument("outdir")
    a = ap.parse_args(argv)

    if a.cmd == "synth":
        synth.generate(a.folder, a.n)
        print(f"wrote {a.n} images and annotations.json to {a.folder}")
    elif a.cmd == "check":
        ds = dataset.Dataset()
        n, corrupt = ds.index(a.folder)
        ann = json.load(open(a.annotations))
        problems = dataset.validate_annotations(ds, ann, synth.CLASSES)
        print(f"{n} images, {len(corrupt)} corrupt, {len(ds.duplicates())} duplicate groups, {len(problems)} label problems")
        for p in problems:
            print(" -", p)
        print("class counts:", agreement.class_counts(ann))
    elif a.cmd == "agree":
        print(agreement.agreement(json.load(open(a.a)), json.load(open(a.b)), a.iou))
    elif a.cmd == "split":
        names = [f for f in os.listdir(a.folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        tr, va = split.split(names, a.val)
        print(f"train {len(tr)}, val {len(va)}")
    elif a.cmd == "overlay":
        os.makedirs(a.outdir, exist_ok=True)
        for name, boxes in json.load(open(a.annotations)).items():
            overlay.draw(os.path.join(a.folder, name), boxes, os.path.join(a.outdir, name))
        print("overlays written to", a.outdir)


if __name__ == "__main__":
    main()
