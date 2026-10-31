"""
app_update.py - File > Check for updates: compare this program's files with the Annotator_Share folder and copy
the newer ones over.

Only program files are touched: the *.py files, requirements.txt and UPDATES.txt that are in the share folder.
Never your project (GimbleDataset), sessions, settings or models. Before anything is replaced the old copies are
kept in <program folder>/_previous/<time>/, and if a copy fails part way everything is put back.

Two ways to update, depending on how the program is run:
  - Run from the .py files: compare with an Annotator_Share folder and swap the files (check / apply, below).
  - Packaged as an .exe: a running exe can't replace itself, so it asks GitHub for the latest release and the
    program opens the download page (latest_release / github_check, below).

No tkinter here, so it can be tested alone.
"""

import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

APP_VERSION = "1.2.0"          # keep in step with the release tag: the tag is "v" + this number
GITHUB_REPO = "Bubbletoppoet47/General-Classification-Annotator"
RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"

KEEP_NAMES = ("requirements.txt", "UPDATES.txt")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\b.*$", re.M)


def program_files(share_dir):
    """Names (in the share folder) that make up the program: *.py (not test files), requirements.txt, UPDATES.txt."""
    out = []
    for f in sorted(os.listdir(share_dir)):
        if not os.path.isfile(os.path.join(share_dir, f)):
            continue
        if (f.lower().endswith(".py") and not f.lower().startswith("test_")) or f in KEEP_NAMES:
            out.append(f)
    return out


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def newest_entry(text):
    """(date, text) of the newest entry of an UPDATES.txt (entries start with a YYYY-MM-DD line), or (None, '')."""
    ms = list(DATE_RE.finditer(text or ""))
    if not ms:
        return None, ""
    end = ms[1].start() if len(ms) > 1 else len(text)
    return ms[0].group(1), text[ms[0].start():end].strip()


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def check(share_dir, app_dir):
    """What would an update change?
    {'ok': bool, 'error': str|None, 'changed': [names that differ], 'new': [names missing here],
     'requirements': bool, 'date': newest share entry date, 'note': its text, 'older': True when the share is older
     than this copy (so an update would go backwards)}"""
    res = {"ok": False, "error": None, "changed": [], "new": [], "requirements": False, "date": None,
           "note": "", "older": False}
    if not share_dir or not os.path.isdir(share_dir):
        res["error"] = "The update folder is not set, or can't be found."
        return res
    if os.path.normcase(os.path.abspath(share_dir)) == os.path.normcase(os.path.abspath(app_dir)):
        res["error"] = "The update folder is the folder you are running from. Pick the Annotator_Share folder."
        return res
    if not os.path.exists(os.path.join(share_dir, "annotator_gui.py")):
        res["error"] = "That folder doesn't look like Annotator_Share (no annotator_gui.py in it)."
        return res
    for name in program_files(share_dir):
        mine = os.path.join(app_dir, name)
        if not os.path.exists(mine):
            res["new"].append(name)
        elif _sha1(mine) != _sha1(os.path.join(share_dir, name)):
            res["changed"].append(name)
    res["requirements"] = "requirements.txt" in res["changed"] + res["new"]
    res["date"], res["note"] = newest_entry(_read(os.path.join(share_dir, "UPDATES.txt")))
    mine_date, _ = newest_entry(_read(os.path.join(app_dir, "UPDATES.txt")))
    res["older"] = bool(res["date"] and mine_date and res["date"] < mine_date)
    res["ok"] = True
    return res


def apply(share_dir, app_dir, names):
    """Copies `names` from the share folder into the program folder. Returns the backup folder (or None when
    nothing needed backing up). Raises OSError after putting everything back if a copy fails."""
    backup = os.path.join(app_dir, "_previous", time.strftime("%Y%m%d_%H%M%S"))
    staged = []
    try:
        for n in names:                                  # 1. stage every file first; nothing is replaced yet
            tmp = os.path.join(app_dir, n + ".update")
            shutil.copyfile(os.path.join(share_dir, n), tmp)
            if _sha1(tmp) != _sha1(os.path.join(share_dir, n)):
                raise OSError(f"{n} did not copy completely (try again when the folder has finished syncing)")
            staged.append(n)
    except Exception:
        for n in staged:
            _rm(os.path.join(app_dir, n + ".update"))
        _rm(os.path.join(app_dir, names[len(staged)] + ".update") if len(staged) < len(names) else "")
        raise
    done, saved = [], []
    try:
        for n in staged:                                 # 2. keep the old copy, then swap
            mine = os.path.join(app_dir, n)
            if os.path.exists(mine):
                os.makedirs(backup, exist_ok=True)
                shutil.copy2(mine, os.path.join(backup, n))
                saved.append(n)
            os.replace(mine + ".update", mine)
            done.append(n)
    except Exception:
        for n in done:                                   # put the old ones back
            if n in saved:
                shutil.copy2(os.path.join(backup, n), os.path.join(app_dir, n))
            else:
                _rm(os.path.join(app_dir, n))
        for n in staged:
            _rm(os.path.join(app_dir, n + ".update"))
        raise
    return backup if saved else None


def _rm(path):
    try:
        if path:
            os.remove(path)
    except OSError:
        pass


# ----------------------------------------------------------------------------------------------------------
# Packaged (.exe) helpers and the GitHub release check
# ----------------------------------------------------------------------------------------------------------

def is_frozen():
    """True when running as a packaged program (PyInstaller) rather than from the .py files."""
    return bool(getattr(sys, "frozen", False))


def app_dir(source_file=None):
    """The folder the program lives in: the folder of the .exe when packaged, else the folder of the .py file.
    (A packaged program's __file__ points into a temporary unpack folder, so it can't be used for this.)"""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(source_file or __file__))


def parse_version(text):
    """'v1.2.3' / '1.2' -> (1, 2, 3) / (1, 2); None when it has no version number in it."""
    m = re.search(r"(\d+(?:\.\d+)*)", text or "")
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def _pad(v, n):
    return tuple(v) + (0,) * (n - len(v))


def is_newer(latest, current):
    """True when version `latest` is higher than `current` (1.0 == 1.0.0)."""
    a, b = parse_version(latest), parse_version(current)
    if a is None or b is None:
        return False
    n = max(len(a), len(b))
    return _pad(a, n) > _pad(b, n)


def latest_release(repo=GITHUB_REPO, timeout=8):
    """Asks GitHub for the newest release: {'tag', 'name', 'notes', 'url'}. Raises OSError / ValueError on trouble."""
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "GeneralClassificationAnnotator"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    return {"tag": d.get("tag_name") or "", "name": d.get("name") or d.get("tag_name") or "",
            "notes": d.get("body") or "", "url": d.get("html_url") or RELEASES_URL,
            "assets": [{"name": a.get("name", ""), "url": a.get("browser_download_url", ""), "size": a.get("size", 0)}
                       for a in (d.get("assets") or [])]}


def github_check(current=APP_VERSION, fetch=latest_release):
    """Is there a newer release on GitHub?
    {'ok': bool, 'error': str|None, 'newer': bool, 'current': str, 'latest': tag, 'notes': str, 'url': str}
    `fetch` can be swapped for a fake in tests."""
    res = {"ok": False, "error": None, "newer": False, "current": current, "latest": "", "notes": "",
           "url": RELEASES_URL}
    try:
        rel = fetch()
    except urllib.error.HTTPError as e:
        res["error"] = ("There is no published release yet." if e.code == 404
                        else f"GitHub answered with an error ({e.code}). Try again later.")
        return res
    except (OSError, ValueError) as e:
        res["error"] = f"Couldn't reach GitHub ({e}). Check your internet connection and try again."
        return res
    if parse_version(rel.get("tag")) is None:
        res["error"] = "The newest release on GitHub has no version number in its name."
        return res
    res.update(ok=True, latest=rel["tag"], notes=rel.get("notes", ""), url=rel.get("url") or RELEASES_URL,
               newer=is_newer(rel["tag"], current), assets=rel.get("assets") or [])
    return res


# ----------------------------------------------------------------------------------------------------------
# Small updates for the packaged program
#
# The packaged program is a launcher .exe plus a big runtime (Python, PyTorch ...) that rarely changes, and an `app`
# folder of plain .py files (our own code) that changes often. A release publishes the full zip, a small code zip and
# update-manifest.json. When the runtime id matches, an update only swaps the `app` folder (about 1 MB). The old
# folder is kept in _previous; a version that fails to start is rolled back by the launcher.
# ----------------------------------------------------------------------------------------------------------

import zipfile

MANIFEST_NAME = "update-manifest.json"
PENDING = ".pending"                 # inside app/: "0" = just updated, never started; "1" = a start is in progress
KEEP_PREVIOUS = 3
_OK_PREFIXES = (f"https://github.com/{GITHUB_REPO}/", f"https://api.github.com/repos/{GITHUB_REPO}/")
_HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "GeneralClassificationAnnotator"}


def check_url(url):
    """Only this project's GitHub addresses (redirects to GitHub's file servers are followed by urllib itself)."""
    if not isinstance(url, str) or not url.startswith(_OK_PREFIXES):
        raise ValueError(f"Refusing an address outside this project's GitHub page: {url}")
    return url


def fetch_json(url, timeout=15):
    with urllib.request.urlopen(urllib.request.Request(check_url(url), headers=_HEADERS), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_file(url, dest, progress=None, timeout=60):
    done = 0
    with urllib.request.urlopen(urllib.request.Request(check_url(url), headers=_HEADERS), timeout=timeout) as r, \
            open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 18)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done)
    return done


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_id(exe_dir=None):
    """The runtime this packaged program was built with (runtime.txt next to the .exe), or None."""
    return (_read(os.path.join(exe_dir or app_dir(), "runtime.txt")).strip() or None)


def plan_update(res, local_runtime, fetch=fetch_json):
    """What to do about the newer release `res` (from github_check):
      {'mode': 'code', 'manifest', 'url', 'bytes'}   only the small code pack is needed
      {'mode': 'full', 'manifest'}                   the runtime changed: the full download is needed
      {'mode': 'page'}                               not set up for small updates: open the download page"""
    assets = {a["name"]: a for a in res.get("assets") or []}
    man_asset = assets.get(MANIFEST_NAME)
    if not man_asset or not local_runtime:
        return {"mode": "page"}
    try:
        man = fetch(man_asset["url"])
    except (OSError, ValueError):
        return {"mode": "page"}
    if man.get("runtime") != local_runtime:
        return {"mode": "full", "manifest": man}
    za = assets.get(man.get("code_zip", ""))
    if not za:
        return {"mode": "page"}
    return {"mode": "code", "manifest": man, "url": za["url"], "bytes": int(man.get("code_bytes") or za.get("size") or 0)}


def _safe_extract(zf, dest):
    root = os.path.realpath(dest)
    for m in zf.infolist():
        target = os.path.realpath(os.path.join(dest, m.filename))
        if target != root and not target.startswith(root + os.sep):
            raise ValueError(f"unsafe path in the update: {m.filename}")
    zf.extractall(dest)


def apply_code_update(exe_dir, plan, progress=None, download=fetch_file, current=APP_VERSION):
    """Downloads the code pack, checks it, and swaps the `app` folder. The old folder goes to _previous/.
    Returns the new version. Raises OSError / ValueError with a plain message and leaves everything as it was."""
    import shutil
    man = plan["manifest"]
    app = os.path.join(exe_dir, "app")
    if not os.path.isdir(app):
        raise OSError("This copy has no app folder, so it can't update itself. Download the full zip.")
    tmp = os.path.join(exe_dir, "_update_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        os.makedirs(tmp)
    except OSError as e:
        raise OSError(f"The program folder isn't writable ({e}). Move the program to a folder you own, "
                      "like C:\\GeneralClassificationAnnotator.")
    try:
        zpath = os.path.join(tmp, "code.zip")
        download(plan["url"], zpath, progress)
        if man.get("code_sha256") and sha256_file(zpath) != man["code_sha256"]:
            raise ValueError("The download was damaged. Try again.")
        new = os.path.join(tmp, "app")
        os.makedirs(new)
        with zipfile.ZipFile(zpath) as zf:
            _safe_extract(zf, new)
        for rel, digest in (man.get("files") or {}).items():
            fp = os.path.join(new, *rel.split("/"))
            if not os.path.isfile(fp) or sha256_file(fp) != digest:
                raise ValueError(f"The update is missing or has a damaged file: {rel}")
        if not os.path.isfile(os.path.join(new, "annotator_gui.py")):
            raise ValueError("The update doesn't contain the program.")
        with open(os.path.join(new, PENDING), "w") as f:
            f.write("0")
        prev_root = os.path.join(exe_dir, "_previous")
        os.makedirs(prev_root, exist_ok=True)
        keep = os.path.join(prev_root, f"app_{current}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.replace(app, keep)
        try:
            os.replace(new, app)
        except OSError:
            os.replace(keep, app)
            raise
        olds = sorted(d for d in os.listdir(prev_root) if d.startswith("app_"))
        for d in olds[:-KEEP_PREVIOUS]:
            shutil.rmtree(os.path.join(prev_root, d), ignore_errors=True)
        return man.get("version") or ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mark_started(exe_dir=None):
    """The program came up fine: forget the 'just updated' marker."""
    try:
        os.remove(os.path.join(exe_dir or app_dir(), "app", PENDING))
    except OSError:
        pass
