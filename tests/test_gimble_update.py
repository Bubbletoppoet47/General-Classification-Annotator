import os, sys, tempfile, shutil, time
import tkinter as tk
from tkinter import messagebox, filedialog
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import app_update as au, annotator_core as ac, annotator_gui as ag, share_zip as sz

tmp = tempfile.mkdtemp()
share, app = os.path.join(tmp, "share"), os.path.join(tmp, "app")
os.makedirs(share); os.makedirs(app)
def w(d, n, t): open(os.path.join(d, n), "w").write(t)
for n in ("annotator_gui.py", "hotkeys.py"): w(share, n, "new " + n); w(app, n, "old " + n)
w(share, "extra.py", "x"); w(share, "test_foo.py", "t"); w(share, "requirements.txt", "pillow\n")
w(share, "UPDATES.txt", "head\n\n2026-10-20 - second\nB\n\n2026-10-08 - first\nA\n")
w(app, "UPDATES.txt", "2026-10-08 - first\nA\n"); w(app, "same.py", "s"); w(share, "same.py", "s")
os.makedirs(os.path.join(app, "GimbleDataset")); w(os.path.join(app, "GimbleDataset"), "manifest.json", "{}")

assert au.program_files(share) == ["UPDATES.txt", "annotator_gui.py", "extra.py", "hotkeys.py", "requirements.txt", "same.py"]
r = au.check(share, app)
assert r["ok"] and sorted(r["changed"]) == ["UPDATES.txt", "annotator_gui.py", "hotkeys.py"] and sorted(r["new"]) == ["extra.py", "requirements.txt"]
assert r["requirements"] and r["date"] == "2026-10-20" and r["note"].startswith("2026-10-20") and "first" not in r["note"] and not r["older"]
assert not au.check(share, share)["ok"] and not au.check(None, app)["ok"] and not au.check(tmp, app)["ok"]
bk = au.apply(share, app, r["changed"] + r["new"])
assert open(os.path.join(app, "annotator_gui.py")).read() == "new annotator_gui.py"
assert open(os.path.join(bk, "annotator_gui.py")).read() == "old annotator_gui.py" and not os.path.exists(os.path.join(bk, "extra.py"))
assert not [f for f in os.listdir(app) if f.endswith(".update")] and os.path.exists(os.path.join(app, "GimbleDataset", "manifest.json"))
r2 = au.check(share, app); assert r2["ok"] and not r2["changed"] and not r2["new"]
# share older than this copy
w(app, "UPDATES.txt", "2026-12-01 - mine\n"); w(share, "hotkeys.py", "different")
assert au.check(share, app)["older"]
# failed swap rolls back
w(app, "hotkeys.py", "keep me"); real = os.replace; calls = {"n": 0}
def bad(a, b):
    calls["n"] += 1
    if calls["n"] == 2: raise OSError("disk full")
    return real(a, b)
w(share, "annotator_gui.py", "newer"); au.os.replace = bad
try: au.apply(share, app, ["annotator_gui.py", "hotkeys.py"]); raise SystemExit("should fail")
except OSError: pass
au.os.replace = real
assert open(os.path.join(app, "annotator_gui.py")).read() == "new annotator_gui.py" and open(os.path.join(app, "hotkeys.py")).read() == "keep me"
assert not [f for f in os.listdir(app) if f.endswith(".update")]

# ---- GUI
sz.CONFIG_PATH = os.path.join(tmp, "cfg.json")
pdir = os.path.join(tmp, "proj")
root = tk.Tk(); a = ag.AnnotatorApp(root, pdir)
a.APP_DIR = app
labels = [a.file_menu.entryconfigure(i, "label")[-1] for i in range(a.file_menu.index("end") + 1) if a.file_menu.type(i) == "command"]
assert "Check for updates..." in labels and "Update folder location..." in labels
msgs = []
messagebox.showinfo = lambda t, m, **k: msgs.append(m); messagebox.showerror = lambda t, m, **k: msgs.append(m)
filedialog.askdirectory = lambda **k: share
asked = []; messagebox.askyesno = lambda t, m, **k: (asked.append(m), False)[1]
w(app, "UPDATES.txt", "2026-10-08 - first\nA\n"); w(share, "hotkeys.py", "different2")
a.check_for_updates()
assert a.cfg["update_dir"] == share and asked and "annotator_gui.py" in asked[0] and "2026-10-20" in asked[0]
assert open(os.path.join(app, "hotkeys.py")).read() == "keep me"                      # declined: nothing changed
finished = []; a._finish_update = lambda req: finished.append(req)
messagebox.askyesno = lambda t, m, **k: True
a.check_for_updates(); assert finished == [False] and open(os.path.join(app, "hotkeys.py")).read() == "different2"
a.check_for_updates(); assert msgs[-1] == "You are up to date."
# File > Dataset location (links a folder anywhere; nothing is copied)
flabels = [a.file_menu.entryconfigure(i, "label")[-1] for i in range(a.file_menu.index("end") + 1) if a.file_menu.type(i) == "command"]
assert "Dataset location..." in flabels and flabels.index("Dataset location...") < flabels.index("Models folder location...")
real_ds = os.path.join(tmp, "GimbleDataset"); os.makedirs(real_ds); w(real_ds, "manifest.json", "{}")
launched2 = []; import subprocess as sp; sp.Popen = lambda cmd, **k: launched2.append(cmd)
filedialog.askdirectory = lambda **k: real_ds; messagebox.askyesno = lambda t, m, **k: True
a.cfg.setdefault("projects", {})[os.path.normcase(os.path.abspath(app))] = "old-per-copy-link"
a.root.destroy = lambda: None; a.choose_dataset()
assert launched2 and launched2[0][-1] == os.path.normpath(real_ds)
assert a.cfg["dataset_dir"] == os.path.normpath(real_ds) and os.path.normcase(os.path.abspath(app)) not in a.cfg["projects"]
assert ag.pick_project(["x"], a.cfg, app)[0] == os.path.normpath(real_ds)                 # even when it is named GimbleDataset
assert ag.pick_project(["x"], a.cfg, os.path.join(tmp, "some_other_folder"))[0] == os.path.normpath(real_ds)   # any copy of the program
# relaunch uses the same project
import subprocess; launched = []
subprocess.Popen = lambda cmd, **k: launched.append(cmd)
ag.AnnotatorApp._finish_update.__get__(a)(False)
assert launched and launched[0][-1] == a.project.root and launched[0][-2].endswith("annotator_gui.py")
# main(): remembers the project; asks before an empty one
# ---- which project opens (pick_project / main)
here1, here2 = os.path.join(tmp, "copy1"), os.path.join(tmp, "copy2")
for h in (here1, here2): os.makedirs(h)
cfg = {}
# nothing linked and nothing beside the program: it asks (and creates nothing)
p, note = ag.pick_project(["x"], cfg, here1); assert p is None and note is None and os.listdir(here1) == []
# a Dataset already beside the program keeps working
os.makedirs(os.path.join(here1, "Dataset")); w(os.path.join(here1, "Dataset"), "manifest.json", "{}")
assert ag.pick_project(["x"], cfg, here1)[0] == os.path.join(here1, "Dataset"); shutil.rmtree(os.path.join(here1, "Dataset"))
# an older GimbleDataset is renamed, contents kept
os.makedirs(os.path.join(here1, "GimbleDataset", "images")); w(os.path.join(here1, "GimbleDataset"), "manifest.json", "{}")
p, note = ag.pick_project(["x"], cfg, here1)
assert p == os.path.join(here1, "Dataset") and note and os.path.exists(os.path.join(p, "manifest.json")) and os.path.isdir(os.path.join(p, "images"))
assert not os.path.exists(os.path.join(here1, "GimbleDataset"))
shutil.rmtree(os.path.join(here1, "Dataset"))
# the linked dataset (File > Dataset location) wins everywhere; an older per-copy memory still works once
other = os.path.join(tmp, "elsewhere"); os.makedirs(other)
cfg = {"dataset_dir": other}
assert ag.pick_project(["x"], cfg, here1)[0] == other and ag.pick_project(["x"], cfg, here2)[0] == other
assert ag.pick_project(["x", "given"], cfg, here1)[0] == "given"
cfg = {"projects": {os.path.normcase(os.path.abspath(here1)): other}}
assert ag.pick_project(["x"], cfg, here1)[0] == other and ag.pick_project(["x"], cfg, here2)[0] is None
cfg = {"dataset_dir": os.path.join(tmp, "gone")}                                                      # linked folder deleted: ask again
assert ag.pick_project(["x"], cfg, here2)[0] is None

opened = []
class FakeApp:
    def __init__(self, root, proj):
        opened.append(proj); self.project = ac.Project(proj, default_classes=["a"])
    def first_run_models_prompt(self): pass
ag.AnnotatorApp = FakeApp
tk.Tk.mainloop = lambda self: None
ag.__file__ = os.path.join(here2, "annotator_gui.py"); sys.argv = ["x"]
# first start: nothing linked -> the link window; "Quit" opens nothing, a new folder is linked and remembered
sz.save_config({"project_dir": other})                                                                 # an old one-folder-for-all memory is ignored
answers = [None]; ag.ask_dataset_location = lambda root, cfg=None: answers[0]
n = len(opened); ag.main(); assert len(opened) == n and not os.path.exists(os.path.join(here2, "Dataset"))
newds = os.path.join(tmp, "my_data"); answers[0] = newds
ag.main()
assert opened[-1] == os.path.normpath(newds) and os.path.isdir(newds) and "project_dir" not in sz.load_config()
assert sz.load_config()["dataset_dir"] == os.path.normpath(newds) and not os.path.exists(os.path.join(here2, "Dataset"))
ag.main()                                                                                              # next start: linked, no question
assert opened[-1] == os.path.normpath(newds)
print("UPDATE OK")
shutil.rmtree(tmp, ignore_errors=True)
