"""
make_release.py - after PyInstaller: put the program's code beside the .exe and make the small-update files.

    python packaging/make_release.py dist/GeneralClassificationAnnotator release-files

Writes into the program folder:  app/ (our .py files + assets), runtime.txt
Writes into the output folder:   GeneralClassificationAnnotator-code.zip, update-manifest.json
(the workflow zips the whole program folder as the full download).

The runtime id changes by itself whenever the libraries (requirements), the spec or the launcher change - that is
when a small update is no longer enough.
"""

import hashlib
import json
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "src")
CODE_ZIP = "GeneralClassificationAnnotator-code.zip"
KEEP = ("requirements.txt", "UPDATES.txt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version():
    return re.search(r'APP_VERSION\s*=\s*"([^"]+)"', open(os.path.join(SRC, "app_update.py")).read()).group(1)


def runtime_id():
    h = hashlib.sha256()
    for rel in ("src/requirements.txt", "packaging/annotator.spec", "packaging/launcher.py"):
        h.update(open(os.path.join(ROOT, rel), "rb").read().replace(b"\r\n", b"\n"))
    h.update(f"py{sys.version_info[0]}.{sys.version_info[1]}".encode())
    return h.hexdigest()[:12]


def build_app_folder(app):
    shutil.rmtree(app, ignore_errors=True)
    os.makedirs(os.path.join(app, "assets"))
    for f in sorted(os.listdir(SRC)):
        if (f.endswith(".py") and not f.startswith("test_")) or f in KEEP:
            shutil.copy2(os.path.join(SRC, f), os.path.join(app, f))
    shutil.copy2(os.path.join(ROOT, "assets", "icon_256.png"), os.path.join(app, "assets", "icon_256.png"))


def main(prog_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    app = os.path.join(prog_dir, "app")
    build_app_folder(app)
    rid = runtime_id()
    with open(os.path.join(prog_dir, "runtime.txt"), "w") as f:
        f.write(rid)
    files = {}
    for dp, _, fs in os.walk(app):
        for f in fs:
            full = os.path.join(dp, f)
            files[os.path.relpath(full, app).replace(os.sep, "/")] = sha256(full)
    zpath = os.path.join(out_dir, CODE_ZIP)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(files):
            z.write(os.path.join(app, *rel.split("/")), rel)
    man = {"format": "gca-update-v1", "version": version(), "runtime": rid, "files": files, "code_zip": CODE_ZIP,
           "code_sha256": sha256(zpath), "code_bytes": os.path.getsize(zpath)}
    with open(os.path.join(out_dir, "update-manifest.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print(f"version {man['version']}  runtime {rid}  {len(files)} files  code zip {man['code_bytes'] // 1024} KB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
