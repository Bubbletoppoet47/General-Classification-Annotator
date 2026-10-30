import os, sys, tempfile, shutil, time
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox, simpledialog
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac, annotator_gui as ag

tmp = tempfile.mkdtemp(); pdir = os.path.join(tmp, "proj")
P = ac.Project(pdir, default_classes=["person", "car", "bike"])
ids = [P.add_uploaded_image(Image.fromarray(np.full((300, 400, 3), 100 + 20 * i, np.uint8)), f"i{i}") for i in range(3)]
P.set_boxes(ids[0], [(0, .5, .5, .2, .3), (1, .2, .2, .1, .1)])
P.set_boxes(ids[1], [(2, .5, .5, .2, .3)])
del P

msgs = []
messagebox.showinfo = lambda *a, **k: msgs.append(("info", a)); messagebox.showerror = lambda *a, **k: msgs.append(("ERR", a))
messagebox.showwarning = lambda *a, **k: msgs.append(("WARN", a)); messagebox.askyesno = lambda *a, **k: True
ag.messagebox, ag.simpledialog = messagebox, simpledialog
root = tk.Tk(); root.geometry("1280x820")
app = ag.AnnotatorApp(root, pdir)
def wait(cond, secs=15):
    t = time.time()
    def poll():
        if cond() or time.time() - t > secs: root.quit()
        else: root.after(30, poll)
    root.after(30, poll); root.mainloop(); assert cond()
root.update()
assert app.class_listbox.size() == 3 and len(app.ai_class_vars) == 3
app.manage_classes(); root.update()
win = app._classes_win; act = app._classes_actions
def find(w, cls):
    out = []
    for c in w.winfo_children():
        if isinstance(c, cls): out.append(c)
        out += find(c, cls)
    return out
lb = find(win, tk.Listbox)[0]
assert lb.size() == 3 and "box(es)" in lb.get(0)
# check AI box for 'bike' so we can see it survives by name
app.ai_class_vars[2].set(True)
# add
simpledialog.askstring = lambda *a, **k: "trash"
act["add"](); wait(lambda: not app._bg_job); root.update()
assert app.project.classes == ["person", "car", "bike", "trash"] and app.class_listbox.size() == 4 and len(app.ai_class_vars) == 4
assert app.ai_class_vars[2].get() and not app.ai_class_vars[3].get()          # checked state kept by name
# bad names refused with a message
msgs.clear(); simpledialog.askstring = lambda *a, **k: "bad name"; act["add"](); assert msgs and msgs[-1][0] == "ERR" and len(app.project.classes) == 4
simpledialog.askstring = lambda *a, **k: "CAR"; msgs.clear(); act["add"](); assert msgs[-1][0] == "ERR"
# rename car -> vehicle
lb.selection_clear(0, tk.END); lb.selection_set(1)
simpledialog.askstring = lambda *a, **k: "vehicle"; act["rename"](); wait(lambda: not app._bg_job); root.update()
assert app.project.classes[1] == "vehicle" and "vehicle" in app.class_listbox.get(1) and "vehicle" in lb.get(1)
assert ac.load_classes(os.path.join(pdir, "classes.txt")) == ["person", "vehicle", "bike", "trash"]
# delete bike (2): boxes moved to person; open image 1 (had a bike box) is reloaded
app.load_image(ids[1]); root.update(); assert app.boxes[0][0] == 2
lb.selection_clear(0, tk.END); lb.selection_set(2)
app._ask_delete_class = lambda parent, idx, n: (0,)
act["delete"](); wait(lambda: not app._bg_job); root.update()
assert app.project.classes == ["person", "vehicle", "trash"] and app.boxes[0][0] == 0, (app.project.classes, app.boxes)
assert app.ai_class_vars[0].get() is False and len(app.ai_class_vars) == 3          # 'bike' was the checked one - gone with it
assert [b[0] for b in app.project.get_boxes(ids[0])] == [0, 1]
# delete 'vehicle' removing boxes
lb.selection_clear(0, tk.END); lb.selection_set(1)
app._ask_delete_class = lambda parent, idx, n: (None,)
act["delete"](); wait(lambda: not app._bg_job); root.update()
assert app.project.classes == ["person", "trash"] and [b[0] for b in app.project.get_boxes(ids[0])] == [0]
# reopen the project: classes persisted
assert ac.Project(pdir, default_classes=[]).classes == ["person", "trash"]
# helper-model class numbers are mapped by name
class FakeAssist: class_names = ["person", "car", "bike", "trash"]
app.model_assist = FakeAssist()
assert app._map_model_classes([(0, .5, .5, .1, .1), (1, .5, .5, .1, .1), (3, .5, .5, .1, .1)]) == [(0, .5, .5, .1, .1), (1, .5, .5, .1, .1)]
print("CLASSES GUI OK"); root.destroy(); shutil.rmtree(tmp, ignore_errors=True)
