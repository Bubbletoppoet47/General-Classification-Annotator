"""Packaged-program behaviour: GitHub update check, relaunch, menu, --selftest (the .exe itself is built on GitHub)."""
import os, subprocess, sys, tempfile, shutil, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import tkinter as tk
from tkinter import messagebox
import app_update as au, annotator_gui as ag, share_zip as sz

tmp = tempfile.mkdtemp(prefix="gca_exe_")
sz.CONFIG_PATH = os.path.join(tmp, "cfg.json")

# ---- version logic
assert au.is_newer("v1.0.1", "1.0.0") and au.is_newer("v1.10.0", "1.9.9") and au.is_newer("v2", "1.9")
assert not au.is_newer("v1.0", "1.0.0") and not au.is_newer("v0.9", "1.0.0") and not au.is_newer("nonsense", "1.0.0")
assert au.RELEASES_URL.endswith("/releases/latest") and au.GITHUB_REPO in au.RELEASES_URL

# ---- github_check with a fake GitHub
r = au.github_check("1.0.0", fetch=lambda: {"tag": "v1.1.0", "notes": "n", "url": "http://x"})
assert r["ok"] and r["newer"] and r["latest"] == "v1.1.0" and r["url"] == "http://x"
assert not au.github_check("1.1.0", fetch=lambda: {"tag": "v1.1.0"})["newer"]
def offline(): raise OSError("offline")
r = au.github_check("1.0.0", fetch=offline); assert not r["ok"] and "internet" in r["error"]
import urllib.error
def nf(): raise urllib.error.HTTPError("u", 404, "nf", {}, None)
r = au.github_check("1.0.0", fetch=nf); assert not r["ok"] and "no published release" in r["error"]
assert not au.github_check("1.0.0", fetch=lambda: {"tag": "latest"})["ok"]

# ---- app_dir: next to the exe when packaged, next to the .py otherwise
assert au.app_dir(os.path.join(tmp, "x", "a.py")) == os.path.join(tmp, "x")
au.is_frozen = lambda: True
assert au.app_dir("ignored.py") == os.path.dirname(os.path.abspath(sys.executable))

# ---- the packaged window: no "Update folder location", Check for updates goes to GitHub
root = tk.Tk(); a = ag.AnnotatorApp(root, os.path.join(tmp, "proj"))
labels = [a.file_menu.entryconfigure(i, "label")[-1] for i in range(a.file_menu.index("end") + 1) if a.file_menu.type(i) == "command"]
assert "Check for updates..." in labels and "Update folder location..." not in labels
msgs, asked, opened = [], [], []
messagebox.showinfo = lambda t, m, **k: msgs.append(m)
messagebox.askyesno = lambda t, m, **k: (asked.append(m), True)[1]
import webbrowser; webbrowser.open = lambda u, *x, **k: opened.append(u)

au.github_check = lambda: {"ok": True, "error": None, "newer": True, "current": "1.0.0", "latest": "v1.2.0",
                           "notes": "fixed zoom", "url": "http://dl"}
a.check_for_updates()                                      # runs in the background; pump the window until it answers
t0 = time.time()
while not asked and time.time() - t0 < 10:
    root.update(); time.sleep(0.02)
assert asked and "1.2.0" in asked[0] and "fixed zoom" in asked[0] and opened == ["http://dl"], (asked, opened)
a._show_github_result({"ok": True, "newer": False, "current": "1.0.0", "latest": "v1.0.0", "notes": "", "url": "u", "error": None})
assert "up to date" in msgs[-1]
a._show_github_result({"ok": False, "error": "Couldn't reach GitHub (x).", "newer": False})
assert "reach GitHub" in msgs[-1]

# ---- relaunch: the packaged program starts itself (no script path), with a clean PyInstaller environment
cmds = []
_real_popen = subprocess.Popen
subprocess.Popen = lambda cmd, **k: cmds.append((cmd, k))
a.root.destroy = lambda: None
a._relaunch(os.path.join(tmp, "p2"))
assert cmds[0][0] == [sys.executable, os.path.join(tmp, "p2")] and cmds[0][1]["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
au.is_frozen = lambda: False
a._relaunch(os.path.join(tmp, "p3"))
assert cmds[1][0][1].endswith("annotator_gui.py") and cmds[1][0][-1] == os.path.join(tmp, "p3")

# ---- --selftest (run for real in a child process, like the built program is)
subprocess.Popen = _real_popen
out = os.path.join(tmp, "self.txt")
rc = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "annotator_gui.py"),
                     "--selftest", out], capture_output=True, text=True, timeout=120).returncode
text = open(out, encoding="utf-8").read()
assert rc == 0 and "SELFTEST OK" in text and "window built" in text and f"version {au.APP_VERSION}" in text, text
try:
    import torch, ultralytics  # noqa: F401
    have_ai = True
except ImportError:
    have_ai = False
rc2 = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "annotator_gui.py"),
                      "--selftest", out, "--expect-ai"], capture_output=True, text=True, timeout=120).returncode
assert (rc2 == 0) == have_ai                                # --expect-ai fails exactly when the AI packages are missing

print("EXE OK")
shutil.rmtree(tmp, ignore_errors=True)
