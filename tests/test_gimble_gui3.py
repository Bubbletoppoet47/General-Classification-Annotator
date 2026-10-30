import os, sys, tempfile, csv, shutil
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_gui as ag
tmp = tempfile.mkdtemp(); ds = os.path.join(tmp, "dataset"); U = "/mnt/user-data/uploads"
if not os.path.exists(U + "/images/000010.jpg"): print("SKIPPED (needs the sample photos from the original test setup)"); sys.exit(0)
for name, ids in (("setA", (1, 2)), ("setB", (1, 2, 3))):
    d = os.path.join(ds, name)
    for sub in ("images", "depth"): os.makedirs(os.path.join(d, sub))
    with open(f"{d}/index.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","time_unix","image","depth","lidar","lidar_points_total","depth_pixels","sharpness"])
        for i in ids:
            shutil.copy(f"{U}/images/000010.jpg", f"{d}/images/{i:06d}.jpg"); shutil.copy(f"{U}/depth/000010.png", f"{d}/depth/{i:06d}.png")
            w.writerow([i, 1790000000 + i, f"images/{i:06d}.jpg", f"depth/{i:06d}.png", "", 0, 7315, 1.0])
msgs = []
messagebox.showinfo = lambda *a, **k: msgs.append(("info", a)); messagebox.showerror = lambda *a, **k: msgs.append(("ERR", a))
messagebox.askyesno = lambda *a, **k: True
filedialog.askdirectory = lambda **k: ds; simpledialog.askinteger = lambda *a, **k: 1
ag.filedialog, ag.messagebox, ag.simpledialog = filedialog, messagebox, simpledialog
root = tk.Tk(); root.geometry("1280x820")
app = ag.AnnotatorApp(root, os.path.join(tmp, "proj"))
def wait(cond, secs=15):
    import time; t = time.time()
    def poll():
        if cond() or time.time() - t > secs: root.quit()
        else: root.after(30, poll)
    root.after(30, poll); root.mainloop(); assert cond()
app.import_gimble_session(); wait(lambda: msgs and not app._bg_job); msgs.clear(); root.update()
vals = list(app.session_combo.cget("values")); print(vals)
assert vals[0].startswith("All sets  (0/5)") and vals[1] == "setA  (0/2)" and vals[2] == "setB  (0/3)"
assert len(app._visible_image_ids()) == 5
# pick set B through the combobox
app.session_combo.current(2); app._on_session_pick(); root.update()
assert app.active_session == "setB" and all("setB" in i for i in app._visible_image_ids()) and len(app._visible_image_ids()) == 3
assert app.current_image_id == "gim_setB_000001" and "set: setB" in app.status_var.get()
assert app.image_listbox.size() == 3
# annotate all of set A -> [A]
app.session_combo.current(1); app._on_session_pick(); root.update()
for _ in range(2):
    app.boxes.append((0, .5, .5, .2, .2)); app.box_dist.append(None); app._save_current_boxes()
    import time; 
    app.next_image(); root.update()
vals = list(app.session_combo.cget("values")); print(vals)
assert vals[1] == "[A] setA  (2/2)" and vals[2] == "setB  (0/3)" and vals[0].startswith("All sets  (2/5)"), vals
# switching to the finished set opens it in review mode; the other set shows only its own frames
app.session_combo.current(2); app._on_session_pick(); root.update()
app.session_combo.current(1); app._on_session_pick(); root.update()
assert app.show_annotated_var.get() and app.image_listbox.size() == 2 and app.current_image_id.startswith("gim_setA")
# frame edit options still work inside a set
app.rotate_current_frame(); root.update()
app.delete_current_frame(); root.update()
assert len(app.project.list_images()) == 4 and app.project.session_progress()["setA"] == (1, 1)
# manage: rename active set keeps it active; Work on This Set button
simpledialog.askstring = lambda *a, **k: "setA2"
app.manage_sessions(); root.update(); win = app._sessions_win
def find(w, cls, text=None):
    for c in w.winfo_children():
        if isinstance(c, cls) and (text is None or c.cget("text") == text): return c
        r = find(c, cls, text)
        if r: return r
tree = find(win, ag.Treeview); tree.selection_set("setA")
find(win, ag.ttk.Button, "Rename...").invoke(); wait(lambda: not app._bg_job)
assert app.active_session == "setA2", app.active_session
tree.selection_set("setB"); find(win, ag.ttk.Button, "Current Set").invoke(); root.update()
assert app.active_session == "setB" and app.current_image_id.startswith("gim_setB")
# no validation role in the annotator
app.manage_sessions(); root.update(); win = app._sessions_win; tree = find(win, ag.Treeview)
assert find(win, ag.ttk.Button, "Train / Validation") is None and "VALIDATION" not in str(tree.item("setB", "values"))
assert not any("(V)" in v for v in app.session_combo.cget("values"))
win.destroy(); root.update()
# delete the active set -> back to all sets
app.manage_sessions(); root.update(); win = app._sessions_win; tree = find(win, ag.Treeview); tree.selection_set("setB")
find(win, ag.ttk.Button, "Delete...").invoke(); wait(lambda: not app._bg_job); root.update()
assert app.active_session is None and all("setA2" in i for i in app.project.list_images())
print("GUI3 TEST OK")
