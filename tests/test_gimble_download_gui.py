import os, sys, tempfile, time
import tkinter as tk
from tkinter import messagebox
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac, annotator_gui as ag, share_zip as sz, model_download as md

tmp = tempfile.mkdtemp()
sz.CONFIG_PATH = os.path.join(tmp, "cfg.json")
ag.INSERTED_DIR = os.path.join(tmp, "inserted")
msgs = []
messagebox.showerror = lambda *a, **k: msgs.append(("ERR", a)); messagebox.showinfo = lambda *a, **k: msgs.append(("info", a))
ag.messagebox = messagebox
ac.Project(os.path.join(tmp, "p"), default_classes=["person"])
root = tk.Tk(); root.geometry("1200x800")
app = ag.AnnotatorApp(root, os.path.join(tmp, "p")); root.update()
labels = [app.file_menu.entrycget(i, "label") for i in range(app.file_menu.index("end") + 1) if app.file_menu.type(i) != "separator"]
assert "Download models from GitHub..." in labels

def E(stem, size, ts): return {"file": stem + ".pt", "stem": stem, "key": "assist", "size": size, "ts": ts, "trainer": "T", "classes": None,
                               "scores": None, "bytes": 3, "url": "u" + stem, "info": {"kind": "assist", "size": size, "created": ts}}
entries = [E("New", "n", 2), E("Old", "n", 1), E("Mid", "m", 1)]
def wait(cond, secs=15):
    t = time.time()
    while not cond() and time.time() - t < secs: root.update(); time.sleep(0.02)
    assert cond()

# offline: friendly message, no window
def off(): raise OSError("offline")
assert app.download_models(fetch=off) is None and "Couldn't reach" in msgs[-1][1][1]

# first run, "newest" chosen
got = {}
def fake_install(models_dir, chosen, progress):
    got["dir"], got["chosen"] = models_dir, [e["stem"] for e in chosen]; progress(0, len(chosen), "x", 1048576)
    return {"installed": [e["file"] for e in chosen], "skipped": [], "errors": []}
app.cfg.pop("models_prompt_done", None)
app.download_models(first_run=True, fetch=lambda: entries, install=fake_install); root.update()
assert app._models_dialog_mode.get() == "newest" and app._models_dialog.winfo_exists()
app._models_dialog_go()
wait(lambda: got and not app._models_dialog.winfo_exists())
assert got["chosen"] == ["New", "Mid"] and got["dir"].endswith("Models"), got
assert app.cfg["models_dir"] == os.path.normpath(got["dir"]) and "Downloaded 2" in msgs[-1][1][1]

# "all"
got.clear()
app.download_models(fetch=lambda: entries, install=fake_install); root.update()
app._models_dialog_mode.set("all"); app._models_dialog_go()
wait(lambda: got and not app._models_dialog.winfo_exists())
assert got["chosen"] == ["New", "Old", "Mid"] or sorted(got["chosen"]) == ["Mid", "New", "Old"], got

# "no thanks" on first run
got.clear(); n = len(msgs)
app.download_models(first_run=True, fetch=lambda: entries, install=fake_install); root.update()
app._models_dialog_mode.set("none"); app._models_dialog_go(); root.update()
assert not got and not app._models_dialog.winfo_exists() and len(msgs) == n

# the first-run prompt only happens once
calls = []
app.download_models = lambda **k: calls.append(k)
app.cfg.pop("models_prompt_done", None)
app.first_run_models_prompt(); app.first_run_models_prompt()
assert len(calls) == 1 and calls[0] == {"first_run": True}
root.destroy()
print("DOWNLOAD GUI OK")
