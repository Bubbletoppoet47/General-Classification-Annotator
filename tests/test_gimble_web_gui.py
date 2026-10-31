import os, sys, tempfile, urllib.request
import tkinter as tk
from tkinter import messagebox, ttk
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac, annotator_gui as ag, share_zip as sz

tmp = tempfile.mkdtemp()
sz.CONFIG_PATH = os.path.join(tmp, "cfg.json"); ag.INSERTED_DIR = os.path.join(tmp, "inserted")
messagebox.showerror = lambda *a, **k: print("ERR", a); messagebox.showinfo = lambda *a, **k: None
ag.messagebox = messagebox
ac.Project(os.path.join(tmp, "p"), default_classes=["person"])
root = tk.Tk(); root.geometry("1200x800")
app = ag.AnnotatorApp(root, os.path.join(tmp, "p")); root.update()
labels = [app.file_menu.entrycget(i, "label") for i in range(app.file_menu.index("end") + 1) if app.file_menu.type(i) != "separator"]
assert "Open in browser (tablet)..." in labels

win = app.open_in_browser(); root.update()
w = app._web
assert w.running and len(w.code) == 6 and win.winfo_exists()
def texts(widget):
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ttk.Label) and c.cget("text"): out.append(c.cget("text"))
        out += texts(c)
    return out
t = " | ".join(texts(win)); assert w.code in t and w.url in t, t
assert urllib.request.urlopen(f"http://127.0.0.1:{w.port}/").status == 200
assert app.open_in_browser() is win                                  # a second click shows the same window
# stop sharing closes the window and the server
stop = [c for c in win.winfo_children()[-1].winfo_children() if c.cget("text") == "Stop sharing"][0]
stop.invoke(); root.update()
assert not w.running and not win.winfo_exists()
try: urllib.request.urlopen(f"http://127.0.0.1:{w.port}/", timeout=2); assert False
except OSError: pass
# opening again gives a NEW code
app.open_in_browser(); root.update(); w2 = app._web
assert w2 is not w and w2.running and w2.code != w.code
w2.stop(); root.destroy()
print("WEB GUI OK")
