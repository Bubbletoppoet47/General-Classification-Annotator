import os, sys, tempfile, shutil, time
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox, filedialog
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac, annotator_gui as ag, share_zip as sz, model_library as ml, model_registry as mr

tmp = tempfile.mkdtemp()
sz.CONFIG_PATH = os.path.join(tmp, "cfg.json")
ag.INSERTED_DIR = os.path.join(tmp, "inserted")
models = os.path.join(tmp, "Models")
msgs = []
messagebox.showerror = lambda *a, **k: msgs.append(("ERR", a)); messagebox.showinfo = lambda *a, **k: msgs.append(("info", a))
ag.messagebox = messagebox
P = ac.Project(os.path.join(tmp, "p"), default_classes=["person"]); del P
root = tk.Tk(); root.geometry("1200x800")
app = ag.AnnotatorApp(root, os.path.join(tmp, "p")); root.update()
loaded = []
app.load_ai_model = lambda kind=None: loaded.append(kind)               # loading real models needs torch; tested elsewhere


def wait(cond, secs=20):
    t = time.time()
    def poll():
        if cond() or time.time() - t > secs: root.quit()
        else: root.after(30, poll)
    root.after(30, poll); root.mainloop(); assert cond()


def entries(menu):
    out = []
    for i in range(menu.index("end") + 1):
        t = menu.type(i)
        out.append("-" if t == "separator" else menu.entrycget(i, "label"))
    return out

# ---- nothing set / folders missing: greyed-out lines that say what to do; nothing is created
app._rebuild_model_menus()
assert "Set the Models folder first" in entries(app.assist_menu)[0] and app.assist_menu.entrycget(0, "state") == "disabled"
filedialog.askdirectory = lambda *a, **k: models
ag.filedialog = filedialog
os.makedirs(models)
assert app.choose_models_dir() == os.path.normpath(models) and sz.load_config()["models_dir"] == os.path.normpath(models)
app._rebuild_model_menus()
assert entries(app.assist_menu)[0].startswith("Missing folder:") and entries(app.scratch_menu)[0].startswith("Missing folder:")
assert os.listdir(models) == []                                                         # the program did not make Assist / Scratch
os.makedirs(os.path.join(models, "Assist")); os.makedirs(os.path.join(models, "Scratch"))
app._rebuild_model_menus()
assert "yet - paste one in" in entries(app.assist_menu)[0]

# ---- versions: newest of each size first (recommended), older ones in nested lists
src = os.path.join(tmp, "x.pt"); open(src, "wb").write(b"A")
t0 = time.mktime((2026, 10, 8, 9, 0, 0, 0, 0, -1))
va1 = ml.save_version(models, "assist", src, "m", "Ian", scores={"map50_95": .31}, ts=t0)
open(src, "wb").write(b"B"); va2 = ml.save_version(models, "assist", src, "m", "Sam", scores={"map50_95": .35}, ts=t0 + 3600)
open(src, "wb").write(b"C"); vn = ml.save_version(models, "assist", src, "n", "Ian", ts=t0 + 60)
open(src, "wb").write(b"S"); vs_ = ml.save_version(models, "scratch", src, "s", "Ian", ts=t0)
app._rebuild_model_menus()
e = entries(app.assist_menu)
assert e[0] == "Newest:  Nano - 2026-10-08 09:01 - Ian" and e[1] == "Newest:  Medium - 2026-10-08 10:00 - Sam - mAP50-95 0.35", e
assert e[2] == "-" and e[3] == "Older Medium versions" and len(e) == 4, e
sub = app._model_submenus[0]; assert entries(sub) == ["Medium - 2026-10-08 09:00 - Ian - mAP50-95 0.31"]
assert entries(app.scratch_menu) == ["Newest:  Small - 2026-10-08 09:00 - Ian"]
assert [m.type(i) for m in [app.file_menu] for i in range(m.index("end") + 1)].count("cascade") == 2

# ---- clicking the newest Medium Assist: copied to this computer, shown in the toolbar box, loaded
app.assist_menu.invoke(1)
wait(lambda: loaded)
label = "Medium Assist 2026-10-08 10:00"
assert loaded == [label] and label in app.ai_choice_box.cget("values"), (loaded, app.ai_choice_box.cget("values"))
ent = app._ai_entries[label]; assert ent["kind"] == "helper" and open(ent["path"], "rb").read() == b"B" and ent["path"].startswith(ag.INSERTED_DIR)
assert os.path.exists(ml.info_path(ent["path"]))
# an older version through the nested list; scratch maps to the scratch loader
sub.invoke(0); wait(lambda: len(loaded) == 2)
assert loaded[1] == "Medium Assist 2026-10-08 09:00" and open(app._ai_entries[loaded[1]]["path"], "rb").read() == b"A"
app.scratch_menu.invoke(0); wait(lambda: len(loaded) == 3)
assert app._ai_entries[loaded[2]]["kind"] == "scratch" and loaded[2] == "Small Scratch 2026-10-08 09:00"
assert not [m for m in msgs if m[0] == "ERR"], msgs
# the Models folder files were not touched; inserted models are remembered after a restart
assert open(va2, "rb").read() == b"B" and open(va1, "rb").read() == b"A"
root.destroy()
root = tk.Tk(); app2 = ag.AnnotatorApp(root, os.path.join(tmp, "p")); root.update()
vals = app2.ai_choice_box.cget("values")
assert label in vals and "Small Scratch 2026-10-08 09:00" in vals and "Medium Assist 2026-10-08 09:00" in vals, vals

# ---- a model pasted by hand: unknown size at first; read in the background; then listed
app2.cfg["models_dir"] = models
pasted = os.path.join(models, "Scratch", "old_model.pt"); open(pasted, "wb").write(b"OLD")
mr.detect_size = lambda kind, ck: "m"
app2._rebuild_model_menus()
assert any("Reading the size of 1 pasted model" in x for x in entries(app2.scratch_menu))
wait(lambda: os.path.exists(ml.info_path(pasted)) and not app2._detecting_models)
app2._rebuild_model_menus()
assert any(x.startswith("Newest:  Medium") and "pasted by hand" in x for x in entries(app2.scratch_menu)), entries(app2.scratch_menu)
assert open(pasted, "rb").read() == b"OLD"
print("MODELS GUI OK"); root.destroy(); shutil.rmtree(tmp, ignore_errors=True)
