"""
model_download.py - get the ready-made AI models from the project's GitHub page.

The models live as downloads on one GitHub release tagged "models" (a pre-release, so it never counts as the
program's newest version). Each model is two files with the same name:

    Nano_2026-10-08_0609_Ian_Selga.pt      the model
    Nano_2026-10-08_0609_Ian_Selga.json    its info (kind, size, created, trainer, classes, scores)

They are saved into the shared Models folder layout (Models/Assist, Models/Scratch) exactly like model_library.py
expects, so File > Insert Assist Model finds them straight away.

Safe on purpose: only this project's GitHub addresses are used, a model is written under a temporary name and only
renamed when its size matches what GitHub said, and an existing model is never overwritten. No tkinter / torch here.
"""

import json
import os
import urllib.request

import model_library as ml

GITHUB_REPO = "Bubbletoppoet47/General-Classification-Annotator"
MODELS_TAG = "models"
MODELS_PAGE = f"https://github.com/{GITHUB_REPO}/releases/tag/{MODELS_TAG}"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{MODELS_TAG}"
_OK_PREFIXES = (f"https://github.com/{GITHUB_REPO}/", f"https://api.github.com/repos/{GITHUB_REPO}/")
_HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "GeneralClassificationAnnotator"}


def check_url(url):
    """Only this project's GitHub addresses are allowed (the download redirects are followed by urllib itself)."""
    if not isinstance(url, str) or not url.startswith(_OK_PREFIXES):
        raise ValueError(f"Refusing to use an address outside this project's GitHub page: {url}")
    return url


def _open(url, timeout):
    req = urllib.request.Request(check_url(url), headers=_HEADERS)
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_json(url, timeout=15):
    with _open(url, timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_file(url, dest, progress=None, timeout=60):
    """Streams `url` into the file `dest`; progress(bytes_so_far) is called as it goes."""
    done = 0
    with _open(url, timeout) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done)
    return done


# ---------------------------------------------------------------- what is on GitHub

def list_remote(fetch=fetch_json):
    """Every model on the models release, newest first:
    [{'file','stem','key','size','ts','trainer','classes','scores','bytes','url','info'}].
    A .pt without its .json next to it is left out. Raises OSError / ValueError when GitHub can't be read."""
    rel = fetch(API_URL)
    assets = {a.get("name", ""): a for a in (rel.get("assets") or [])}
    out = []
    for name, a in assets.items():
        if not name.lower().endswith(".pt"):
            continue
        stem = name[:-3]
        ja = assets.get(stem + ".json")
        if not ja:
            continue
        try:
            info = fetch(ja["browser_download_url"]) or {}
        except (OSError, ValueError):
            continue
        key = {"assist": "assist", "scratch": "scratch"}.get(str(info.get("kind", "assist")).lower(), "assist")
        out.append({"file": name, "stem": stem, "key": key, "size": info.get("size"),
                    "ts": info.get("created") or 0, "trainer": info.get("trainer"),
                    "classes": info.get("classes"), "scores": info.get("scores"),
                    "bytes": int(a.get("size") or 0), "url": a["browser_download_url"], "info": info})
    out.sort(key=lambda v: (v["ts"], v["file"]), reverse=True)
    return out


def newest(entries):
    """The newest model of each size, per kind (Assist / Scratch)."""
    out = []
    for key in ml.FOLDERS:
        out += ml.newest_per_size([e for e in entries if e["key"] == key])
    return out


def pick(entries, mode):
    """mode 'newest' or 'all'."""
    return list(entries) if mode == "all" else newest(entries)


def describe(e):
    """One line: 'Nano - 2026-10-08 06:09 - Ian_Selga - 5.1 MB'."""
    mb = f"{e['bytes'] / 1048576:.1f} MB" if e.get("bytes") else "size unknown"
    return ml.describe(e) + " - " + mb


def already_have(models_dir, e):
    p = os.path.join(ml.folder_path(models_dir, e["key"]), e["file"])
    return os.path.isfile(p) and (not e.get("bytes") or os.path.getsize(p) == e["bytes"])


# ---------------------------------------------------------------- saving them

def install(models_dir, entries, progress=None, download=fetch_file):
    """Downloads `entries` into models_dir (making Models/Assist and Models/Scratch when needed).
    progress(index, total, file, bytes_done) is called while it works.
    Returns {'installed': [files], 'skipped': [files already there], 'errors': [(file, message)]}."""
    res = {"installed": [], "skipped": [], "errors": []}
    for i, e in enumerate(entries):
        folder = ml.folder_path(models_dir, e["key"])
        pt = os.path.join(folder, e["file"])
        try:
            os.makedirs(folder, exist_ok=True)
            if already_have(models_dir, e):
                res["skipped"].append(e["file"])
                continue
            if os.path.exists(pt):                                   # a different file with this name: keep both
                base, n = e["stem"], 2
                while os.path.exists(os.path.join(folder, f"{base}_{n}.pt")):
                    n += 1
                pt = os.path.join(folder, f"{base}_{n}.pt")
            part = pt + ".part"
            download(e["url"], part, (lambda b, i=i, e=e: progress(i, len(entries), e["file"], b)) if progress else None)
            have = os.path.getsize(part)
            if e.get("bytes") and have != e["bytes"]:
                os.remove(part)
                raise OSError(f"the download was incomplete ({have} of {e['bytes']} bytes)")
            os.replace(part, pt)
            info = dict(e["info"])
            info.setdefault("format", ml.FORMAT)
            info["pasted"] = False
            ml._write_json_atomic(ml.info_path(pt), info)                    # the info file goes last
            res["installed"].append(os.path.basename(pt))
        except (OSError, ValueError) as ex:
            for leftover in (pt + ".part",):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
            res["errors"].append((e["file"], str(ex)))
    return res
