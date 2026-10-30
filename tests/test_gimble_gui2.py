import os, sys, tempfile, csv, shutil
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
from tkinter.ttk import Treeview
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac
import annotator_gui as ag

tmp = tempfile.mkdtemp()
ds = os.path.join(tmp, "dataset"); sess = os.path.join(ds, "20260930_095437")
for sub in ("images", "depth"): os.makedirs(os.path.join(sess, sub))
U = "/mnt/user-data/uploads"
if not os.path.exists(U + "/images/000010.jpg"): print("SKIPPED (needs the sample photos from the original test setup)"); sys.exit(0)
with open(f"{sess}/index.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["idx","time_unix","image","depth","lidar","lidar_points_total","depth_pixels","sharpness"])
    for i in (10, 11, 12):
        shutil.copy(f"{U}/images/000010.jpg", f"{sess}/images/{i:06d}.jpg"); shutil.copy(f"{U}/depth/000010.png", f"{sess}/depth/{i:06d}.png")
        w.writerow([i, 1790000000 + i, f"images/{i:06d}.jpg", f"depth/{i:06d}.png", "", 0, 7315, 486.0])

msgs = []
messagebox.showinfo = lambda *a, **k: msgs.append(("info", a))
messagebox.showerror = lambda *a, **k: msgs.append(("ERR", a))
messagebox.askyesno = lambda *a, **k: True
filedialog.askdirectory = lambda **k: ds
simpledialog.askinteger = lambda *a, **k: 1
ag.filedialog, ag.messagebox, ag.simpledialog = filedialog, messagebox, simpledialog

root = tk.Tk(); root.geometry("1280x820")
app = ag.AnnotatorApp(root, os.path.join(tmp, "proj"), ) if False else ag.AnnotatorApp(root, os.path.join(tmp, "proj"))
app.project.classes = ["person", "bottle"]; app.project.save_classes()

def wait(cond, secs=20):
    import time; t = time.time()
    def poll():
        if cond() or time.time() - t > secs: root.quit()
        else: root.after(30, poll)
    root.after(30, poll); root.mainloop()
    assert cond(), "timeout"

app.import_gimble_session(); wait(lambda: msgs); assert "Added 3 new" in msgs[0][1][1]; msgs.clear()
root.update()
cur = app.current_image_id; assert cur.endswith("000010")
class Ev:  # fake canvas event
    def __init__(s, x, y): s.x, s.y = x, y
# draw a box that covers the middle of the picture, via real drag handlers
cw, ch = app.canvas.winfo_width(), app.canvas.winfo_height()
app._redraw()
x0, y0 = app._image_px_to_canvas(120, 100); x1, y1 = app._image_px_to_canvas(520, 400)
app._on_canvas_press(Ev(x0, y0)); app._on_canvas_release(Ev(x1, y1))
assert len(app.boxes) == 1 and app.box_dist == [None]
# pick mode via the P key
root.event_generate("<p>"); root.update(); assert app.pick_var.get() and app.show_depth_var.get()
# find a real LiDAR point inside the box and hover just beside it
u, v, z = app.project.depth_points(cur)
ins = np.nonzero((u > 200) & (u < 450) & (v > 150) & (v < 350))[0]; k = int(ins[len(ins)//2])
px, py = app._image_px_to_canvas(u[k], v[k])
app._on_canvas_motion(Ev(px, py)); assert app.canvas.find_withtag("hover"), "no hover overlay"
cand = app._pick_candidate(px, py); assert cand and cand[0] == 0 and abs(cand[3] - z[k]) < 1e-6, cand
# hovering outside the box gives no candidate; outside radius none either
assert app._pick_candidate(*app._image_px_to_canvas(10, 10)) is None
app._on_canvas_motion(Ev(*app._image_px_to_canvas(10, 10)))
# click (tiny move) assigns
app._on_canvas_press(Ev(px, py)); app._on_canvas_release(Ev(px, py))
pk = app.box_dist[0]; assert pk and pk["distance_m"] == round(float(z[k]), 3), pk
assert app.project.get_distances(cur) == [pk] and app.project.is_annotated(cur)
# double-click does nothing in pick mode
app._on_canvas_double_click(Ev(px, py)); assert len(app.boxes) == 1
assert "with distance" in app.status_var.get() and "(1 with" in app.status_var.get(), app.status_var.get()
# rotation keeps the pick on the same LiDAR return
app.rotate_current_frame(); root.update()
pk2 = app.box_dist[0]; d = app.project.load_depth_mm(cur)
assert d[pk2["v"], pk2["u"]] == int(round(pk["distance_m"] * 1000)), (pk, pk2)
# right click in pick mode clears the distance, keeps the box
bx = app.boxes[0]; img_w, img_h = app.current_pil_image.size
cxp, cyp = app._image_px_to_canvas(bx[1] * img_w, bx[2] * img_h)
app._on_canvas_right_click(Ev(cxp, cyp)); assert app.box_dist == [None] and len(app.boxes) == 1
# assign again then Clear button
app._on_canvas_motion(Ev(cxp, cyp))
u, v, z = app.project.depth_points(cur)
cand = None
for dx in range(0, 200, 4):
    cand = app._pick_candidate(cxp + dx, cyp)
    if cand: app._assign_pick(cxp + dx, cyp); break
assert app.box_dist[0] is not None
app.clear_selected_distance(); assert app.box_dist == [None]
app._assign_pick(cxp + dx, cyp); assert app.box_dist[0]
# turn pick mode off: double-click delete works again is NOT tested (would delete). toggle:
root.event_generate("<p>"); root.update(); assert not app.pick_var.get() and not app.show_depth_var.get()
# D is the same single switch: dots and picking turn on and off together
root.event_generate("<d>"); root.update(); assert app.pick_var.get() and app.show_depth_var.get()
root.event_generate("<d>"); root.update(); assert not app.pick_var.get() and not app.show_depth_var.get()
app.pick_var.set(True); app._on_pick_toggle(); assert app.show_depth_var.get()      # checkbox
app.pick_var.set(False); app._on_pick_toggle(); assert not app.show_depth_var.get()
# a second image with a box, so exports are non-trivial
app.next_image(); root.update()
app.boxes.append((1, .5, .5, .3, .3)); app.box_dist.append(None); app._save_current_boxes()
# the annotator no longer exports or trains
assert not hasattr(app, "export_dataset") and not hasattr(app, "train_models") and not hasattr(app, "_auto_distort")
msgs.clear()

# ---- Manage Sessions ----
app.manage_sessions(); root.update()
win = app._sessions_win
def find(w, cls, text=None):
    for c in w.winfo_children():
        if isinstance(c, cls) and (text is None or c.cget("text") == text): return c
        r = find(c, cls, text)
        if r: return r
tree = find(win, Treeview)
rows = [tree.item(i, "values") for i in tree.get_children()]
print("rows", rows); assert (rows[0][0], *rows[0][1:4]) == ("20260930_095437", "3", "2", "3"), rows
# rename: bad name rejected then good name
answers = iter(["bad name", "kitchen_A"])
simpledialog.askstring = lambda *a, **k: next(answers)
tree.selection_set("20260930_095437")
find(win, tk.Widget.__mro__[0] if False else ag.ttk.Button, "Rename...").invoke()
assert msgs and msgs[0][0] == "ERR", msgs
assert "Renamed" in app.progress_var.get() and not [m for m in msgs if m[0] != "ERR"]
msgs.clear()
ids = app.project.list_images(); print(ids); assert all(i.startswith("gim_kitchen_A_") for i in ids)
assert app.current_image_id in ids
assert [tree.item(i, "values")[0] for i in tree.get_children()] == ["kitchen_A"]
app.load_image(ids[0]); root.update()   # navigation still works after rename
# delete
tree.selection_set("kitchen_A")
find(win, ag.ttk.Button, "Delete...").invoke()
root.update()
assert app.project.list_images() == [] and app.current_image_id is None and app.boxes == []
assert not tree.get_children()
win.destroy(); root.update(); assert app._sessions_refresh is None
print("GUI2 TEST OK", [m[1][0] for m in msgs])
