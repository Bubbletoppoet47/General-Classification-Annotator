"""
model_registry.py - where each trained model lives, and which ones exist.

Every model size has its OWN slot, one per kind:

    helper  (pretrained, boxes + classes)   ~/yolo_runs/helper_n  helper_s  helper_m  helper_l
    scratch (from scratch, + distance)      ~/yolo_runs/gimble_model_n  gimble_model_s  gimble_model_m  gimble_model_l

Training a size only ever continues / overwrites THAT size's slot; the other sizes are never touched.
The annotator lists every slot that has a trained model ("Medium helper", "Nano scratch", ...).

Older single-folder runs keep working: ~/yolo_runs/helper and ~/yolo_runs/gimble_model are used as the slot
of whatever size they actually are (read from the checkpoint once and remembered in model_info.json),
so nothing has to be moved or renamed.

Light on purpose: importing this pulls in no torch / ultralytics (only detecting the size of an old,
un-labelled folder does, and only once).
"""

import json
import os
import re

RUNS_DIR = os.path.join(os.path.expanduser("~"), "yolo_runs")
SIZES = ("n", "s", "m", "l")
SIZE_NAMES = {"n": "Nano", "s": "Small", "m": "Medium", "l": "Large", "x": "XLarge"}
DEFAULT_SIZE = {"scratch": "n", "helper": "m"}
PREFIX = {"scratch": "gimble_model", "helper": "helper"}        # folder = <prefix>_<size>; the bare prefix is the old single folder
INFO_FILE = "model_info.json"


def label(kind, size):
    return f"{SIZE_NAMES.get(size, size)} {kind}"


def best_path(run_dir):
    return os.path.join(run_dir, "weights", "best.pt")


def read_info(run_dir):
    try:
        with open(os.path.join(run_dir, INFO_FILE), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def write_info(run_dir, **fields):
    """Merges fields into run_dir/model_info.json (kind, size, ...)."""
    os.makedirs(run_dir, exist_ok=True)
    d = read_info(run_dir)
    d.update({k: v for k, v in fields.items() if v is not None})
    tmp = os.path.join(run_dir, INFO_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, os.path.join(run_dir, INFO_FILE))


def detect_size(kind, checkpoint):
    """'n'/'s'/'m'/'l'/'x' of a checkpoint (loads it; needs torch / ultralytics), or None."""
    try:
        if kind == "scratch":
            import torch
            cfg = torch.load(checkpoint, map_location="cpu", weights_only=False).get("cfg", "")
        else:
            from ultralytics import YOLO
            y = YOLO(checkpoint).model.yaml
            if y.get("scale") in tuple("nslmx"):
                return y["scale"]
            cfg = str(y.get("yaml_file", ""))
        m = re.search(r"yolo\d+([nslmx])", str(cfg))
        return m.group(1) if m else None
    except Exception:  # noqa: BLE001
        return None


def legacy_dir(kind, runs_dir=RUNS_DIR):
    return os.path.join(runs_dir, PREFIX[kind])


def legacy_size(kind, runs_dir=RUNS_DIR, detect=True):
    """Size of the old single-folder run, remembered in its model_info.json. None if there is none / unknown."""
    d = legacy_dir(kind, runs_dir)
    if not os.path.exists(best_path(d)):
        return None
    sz = read_info(d).get("size")
    if sz in SIZE_NAMES:
        return sz
    if not detect:
        return None
    sz = detect_size(kind, best_path(d))
    if sz:
        try:
            write_info(d, kind=kind, size=sz)
        except OSError:
            pass
    return sz


def slot_dir(kind, size, runs_dir=RUNS_DIR):
    """The folder training `size` of `kind` should use: its own slot, or the old single folder when that one
    is already a model of exactly this size."""
    own = os.path.join(runs_dir, f"{PREFIX[kind]}_{size}")
    if os.path.exists(own):
        return own
    if legacy_size(kind, runs_dir) == size:
        return legacy_dir(kind, runs_dir)
    return own


def resolve_run_dir(kind, size, runs_dir=RUNS_DIR):
    """Folder + size for a training run. size None = the old single folder if it exists (whatever its size),
    otherwise the default size's slot."""
    if size:
        return slot_dir(kind, size, runs_dir), size
    if os.path.exists(best_path(legacy_dir(kind, runs_dir))):
        return legacy_dir(kind, runs_dir), legacy_size(kind, runs_dir)
    sz = DEFAULT_SIZE[kind]
    return slot_dir(kind, sz, runs_dir), sz


def discover(runs_dir=RUNS_DIR, detect_legacy=False):
    """Every trained model slot: [{'kind','size','label','dir','path'}], helpers first, small to large.
    detect_legacy=True also identifies old single-folder runs by loading them (slow once, then remembered)."""
    found = []
    for kind in ("helper", "scratch"):
        for size in SIZES + ("x",):
            own = os.path.join(runs_dir, f"{PREFIX[kind]}_{size}")
            if os.path.exists(best_path(own)):
                found.append(dict(kind=kind, size=size, label=label(kind, size), dir=own, path=best_path(own)))
        ld = legacy_dir(kind, runs_dir)
        if os.path.exists(best_path(ld)):
            sz = legacy_size(kind, runs_dir, detect=detect_legacy)
            if sz and not any(f["kind"] == kind and f["size"] == sz for f in found):
                found.append(dict(kind=kind, size=sz, label=label(kind, sz), dir=ld, path=best_path(ld)))
    order = {s: i for i, s in enumerate(SIZES + ("x",))}
    found.sort(key=lambda f: (f["kind"] != "helper", order.get(f["size"], 9)))
    return found
