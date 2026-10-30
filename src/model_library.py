"""
model_library.py - the shared Models folder: every trained model, as dated versions.

    <Models>/Assist/    helper models   (pretrained start, boxes + classes)
    <Models>/Scratch/   from-scratch models (+ distance)

YOU make these three folders and paste the first models in (any file name). This program never creates
them: when one is missing it says which one and does nothing.

A version is a .pt file with a small .json "info" file of the same name:

    Medium_2026-10-08_1530_Ian.pt
    Medium_2026-10-08_1530_Ian.json     {kind, size, created, trainer, classes, scores, based_on}

  * Training a size starts from the NEWEST version of that size (it is copied into the training slot first)
    and, when it finishes cleanly, saves a NEW version here. Older versions are never changed or deleted.
  * A model you pasted by hand gets its info file the first time it is looked at (the .pt is never touched):
    the size is read from the model itself, the date is the file's date.
  * Files are written under a temporary name and renamed when complete, and the info file goes last, so a
    half-synced Drive folder never shows a half-written model.
  * Newest = latest date. A name with the same stem gets _2, _3 ... so nothing is overwritten.

Light on purpose: no torch / tkinter here (only reading the size of a pasted model loads torch, via
model_registry.detect_size).
"""

import json
import os
import re
import shutil
import time

import model_registry as mr

FORMAT = "gimble-model-v1"
FOLDERS = {"assist": "Assist", "scratch": "Scratch"}          # key -> folder name inside Models
REG_KIND = {"assist": "helper", "scratch": "scratch"}          # key -> model_registry kind
KEY_OF_REG = {v: k for k, v in REG_KIND.items()}
SIZE_ORDER = ("n", "s", "m", "l", "x")

_no_size = set()          # pasted files whose size could not be read (do not retry on every menu open)


def _safe(text, limit=24):
    t = re.sub(r"[^A-Za-z0-9_-]+", "_", (text or "").strip()).strip("_")
    return (t or "unknown")[:limit]


def folder_path(models_dir, key):
    return os.path.join(models_dir, FOLDERS[key])


def missing_folders(models_dir, key=None):
    """Folders you still have to make: [] when everything needed exists. key None = Models + both subfolders."""
    if not models_dir or not os.path.isdir(models_dir):
        return [models_dir or "(Models folder not set)"]
    keys = [key] if key else list(FOLDERS)
    return [folder_path(models_dir, k) for k in keys if not os.path.isdir(folder_path(models_dir, k))]


def missing_message(models_dir, key=None):
    miss = missing_folders(models_dir, key)
    if not miss:
        return None
    if not models_dir:
        return "Pick the Models folder first."
    return "Please make this folder (the program does not create it): " + "; ".join(miss)


# ---------------------------------------------------------------- info files

def info_path(pt):
    return os.path.splitext(pt)[0] + ".json"


def read_info(pt):
    try:
        with open(info_path(pt), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) and d.get("format") == FORMAT else None
    except (OSError, ValueError):
        return None


def _write_json_atomic(path, data):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def ensure_info(key, pt):
    """The info of a model file; for a pasted one without info, reads its size, saves the info file next to it
    (the .pt is untouched) and returns it. None when the size cannot be read (loading torch failed, etc.)."""
    info = read_info(pt)
    if info:
        return info
    if pt in _no_size:
        return None
    size = mr.detect_size(REG_KIND[key], pt)
    if size not in mr.SIZE_NAMES:
        _no_size.add(pt)
        return None
    info = {"format": FORMAT, "kind": key, "size": size, "created": os.path.getmtime(pt),
            "trainer": None, "classes": None, "scores": None, "based_on": None, "pasted": True}
    try:
        _write_json_atomic(info_path(pt), info)
    except OSError:
        pass                                            # read-only folder: still usable this run
    return info


# ---------------------------------------------------------------- listing

def _fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def list_versions(models_dir, key, detect=False):
    """Every model file in Models/<Assist|Scratch>, newest first:
    [{'path','file','key','size','ts','trainer','classes','scores','based_on'}].
    detect=True reads the size of pasted models that have no info yet (slow once per file; loads torch);
    otherwise those are returned with size None."""
    d = folder_path(models_dir, key) if models_dir else ""
    if not d or not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        if not f.lower().endswith(".pt"):
            continue
        pt = os.path.join(d, f)
        info = ensure_info(key, pt) if detect else read_info(pt)
        try:
            mtime = os.path.getmtime(pt)
        except OSError:
            continue
        info = info or {}
        out.append({"path": pt, "file": f, "key": key, "size": info.get("size"), "ts": info.get("created") or mtime,
                    "trainer": info.get("trainer"), "classes": info.get("classes"), "scores": info.get("scores"),
                    "based_on": info.get("based_on")})
    out.sort(key=lambda v: (v["ts"], v["file"]), reverse=True)
    return out


def newest_per_size(versions):
    """The newest version of each size, small to large (versions without a known size are left out)."""
    best = {}
    for v in versions:                                  # versions are newest first
        if v["size"] and v["size"] not in best:
            best[v["size"]] = v
    return [best[s] for s in SIZE_ORDER if s in best]


def older_per_size(versions):
    """{size: [older versions, newest first]} - everything except each size's newest."""
    seen, out = set(), {}
    for v in versions:
        if not v["size"]:
            continue
        if v["size"] not in seen:
            seen.add(v["size"])
            continue
        out.setdefault(v["size"], []).append(v)
    return out


def describe(v):
    """One line for a menu: 'Medium - 2026-10-08 15:30 - Ian - mAP50-95 0.41'."""
    parts = [mr.SIZE_NAMES.get(v["size"], "Unknown size"), _fmt_ts(v["ts"]), v["trainer"] or "pasted by hand"]
    sc = v.get("scores") or {}
    if sc.get("map50_95") is not None:
        parts.append(f"mAP50-95 {sc['map50_95']}")
    return " - ".join(parts)


def short_label(v):
    """Name for the toolbar AI box: 'Medium Assist 2026-10-08 15:30'."""
    return f"{mr.SIZE_NAMES.get(v['size'], '?')} {FOLDERS[v['key']]} {_fmt_ts(v['ts'])}"


# ---------------------------------------------------------------- writing versions

def version_stem(size, ts, trainer):
    return f"{mr.SIZE_NAMES.get(size, size)}_{time.strftime('%Y-%m-%d_%H%M', time.localtime(ts))}_{_safe(trainer)}"


def save_version(models_dir, key, src_pt, size, trainer, classes=None, scores=None, based_on=None, ts=None):
    """Adds src_pt as a new version. Never creates the Models/Assist/Scratch folders (returns None if one is
    missing) and never overwrites: a taken name gets _2, _3 ... Returns the new .pt path."""
    if missing_folders(models_dir, key):
        return None
    ts = ts or os.path.getmtime(src_pt)
    d = folder_path(models_dir, key)
    stem = version_stem(size, ts, trainer)
    cand, n = stem, 1
    while os.path.exists(os.path.join(d, cand + ".pt")) or os.path.exists(os.path.join(d, cand + ".json")):
        n += 1
        cand = f"{stem}_{n}"
    dest = os.path.join(d, cand + ".pt")
    tmp = dest + ".part"
    shutil.copyfile(src_pt, tmp)
    os.replace(tmp, dest)
    _write_json_atomic(info_path(dest), {"format": FORMAT, "kind": key, "size": size, "created": ts, "trainer": trainer,
                                          "classes": list(classes) if classes else None, "scores": scores,
                                          "based_on": based_on, "pasted": False})            # info goes last
    return dest


def seed_slot(models_dir, key, size, run_dir):
    """Before training: if the Models folder has a newer version of this size than the training slot's own
    best.pt, copy it there so training continues from the newest model. The slot's old best.pt is kept as
    weights/before_seed_<time>_best.pt first. Returns the version used, or None."""
    vs = [v for v in list_versions(models_dir, key, detect=True) if v["size"] == size]
    if not vs:
        return None
    v = vs[0]
    wdir = os.path.join(run_dir, "weights")
    best = os.path.join(wdir, "best.pt")
    if os.path.exists(best) and os.path.getmtime(best) >= v["ts"]:
        return None                                      # the slot already has this or something newer
    os.makedirs(wdir, exist_ok=True)
    if os.path.exists(best):
        shutil.copy2(best, os.path.join(wdir, f"before_seed_{time.strftime('%Y%m%d_%H%M%S')}_best.pt"))
    tmp = best + ".seed"
    shutil.copyfile(v["path"], tmp)
    os.replace(tmp, best)
    os.utime(best, (v["ts"], v["ts"]))                   # the slot is now exactly as old as that version
    return v


def publish_run(models_dir, key, run_dir, size, trainer, classes=None, scores=None, based_on=None):
    """After a clean training run: saves run_dir/weights/best.pt as a new version. Returns the path or None."""
    best = os.path.join(run_dir, "weights", "best.pt")
    if not models_dir or not os.path.exists(best):
        return None
    return save_version(models_dir, key, best, size, trainer, classes, scores, based_on)


# ---------------------------------------------------------------- the annotator side

def local_copy(v, dest_root):
    """Copies a version (model + info) to <dest_root>/<Assist|Scratch>/ on this computer so the annotator does
    not depend on Drive being synced while you work. Reuses an identical earlier copy. Returns the local .pt."""
    d = os.path.join(dest_root, FOLDERS[v["key"]])
    os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, v["file"])
    src_stat = os.stat(v["path"])
    if not (os.path.exists(dest) and os.path.getsize(dest) == src_stat.st_size):
        tmp = dest + ".part"
        shutil.copyfile(v["path"], tmp)
        os.replace(tmp, dest)
        if os.path.exists(info_path(v["path"])):
            shutil.copyfile(info_path(v["path"]), info_path(dest))
    return dest
