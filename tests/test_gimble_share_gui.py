import os, sys, tempfile, shutil, time, zipfile, json
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac, annotator_gui as ag, share_zip as sz

tmp = tempfile.mkdtemp()
sz.CONFIG_PATH = os.path.join(tmp, "cfg_ian.json")           # never touch the real ~/.trash_annotator.json
shared = os.path.join(tmp, "drive"); os.makedirs(shared)


def add_frame(P, session, idx, seed):
    arr = np.random.default_rng(seed).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    iid = P.add_uploaded_image(Image.fromarray(arr), f"f{seed}")
    P.manifest["images"][iid].update(source="gimble", session_id=session, sample_index=idx, source_name=session)
    d = np.zeros((60, 80), np.uint16); d[10:20, 10:20] = 2500
    Image.fromarray(d).save(P.depth_path(iid)); P.manifest["images"][iid].update(has_depth=True, depth_pixels=100)
    P.save_manifest()
    return iid


pdir = os.path.join(tmp, "ian")
P = ac.Project(pdir, default_classes=["person", "car"])
i1, i2, i3, i4 = (add_frame(P, "kitchen", n, 20 + n) for n in range(4))
P.set_boxes(i1, [(0, .5, .5, .2, .3)], distances=[{"distance_m": 2.0, "u": 12, "v": 12}])
P.set_boxes(i2, [(1, .4, .4, .2, .2)]); P.set_boxes(i3, [])
del P

msgs = []
messagebox.showinfo = lambda *a, **k: msgs.append(("info", a))
messagebox.showerror = lambda *a, **k: msgs.append(("ERR", a))
messagebox.showwarning = lambda *a, **k: msgs.append(("WARN", a))
asked = []
messagebox.askyesno = lambda *a, **k: (asked.append(a), True)[1]
simpledialog.askstring = lambda *a, **k: "Ian"
filedialog.askdirectory = lambda *a, **k: shared
ag.messagebox, ag.simpledialog, ag.filedialog = messagebox, simpledialog, filedialog

root = tk.Tk(); root.geometry("1280x820")
app = ag.AnnotatorApp(root, pdir)


def wait(cond, secs=30):
    t = time.time()
    def poll():
        if cond() or time.time() - t > secs: root.quit()
        else: root.after(30, poll)
    root.after(30, poll); root.mainloop(); assert cond()


def find(w, cls):
    out = []
    for c in w.winfo_children():
        if isinstance(c, cls): out.append(c)
        out += find(c, cls)
    return out


def info_texts():
    return [m[1][1] for m in msgs if m[0] == "info"]


root.update()
# ---- a project that used the OLD single-zip sharing thinks everything is already shared: first publish still writes all
app.me = sz.set_identity(app.cfg, "Ian"); app.project.editor_id = app.me["id"]
for iid in (i1, i2, i3):
    inf = app.project.manifest["images"][iid]; inf["uid"] = "olduid" + iid[-3:]; inf["owner_id"] = app.me["id"]
    inf["share_base"] = sz.local_state(app.project, iid)
app.project.save_manifest()
assert sz.changed_ids(app.project) == [i for i in sz.changed_ids(app.project)] and len(sz.changed_ids(app.project)) == 0
# ---- the menu bar: File / Edit / Manage, and nothing about training in the annotator
bar = app.menubar
assert [bar.entrycget(i, "label") for i in range(bar.index("end") + 1)] == ["File", "Edit", "Manage"]
def labels(m): return [m.entrycget(i, "label") for i in range(m.index("end") + 1) if m.type(i) != "separator"]
assert labels(app.file_menu) == ["Import Gimble Session...", "Import Video...", "Insert Assist Model", "Insert Scratch Model",
                                 "Publish to shared folder", "Import data to edit...", "Shared folder location...",
                                 "Models folder location...", "Check for updates...", "Update folder location...", "Exit"], labels(app.file_menu)
assert labels(app.edit_menu) == ["Classes...", "Hotkeys...", "My name...", "AI confidence...", "Toggle theme"], labels(app.edit_menu)
assert labels(app.manage_menu) == ["Sessions...", "Sync Deleted Files", "Dataset location..."]
btn_texts = {b.cget("text") for b in find(root, ttk.Button)}
assert not btn_texts & {"Export Dataset", "Train...", "Manage Classes...", "Hotkeys...", "Import Gimble Session...", "Sync Deleted Files", "Confidence..."}, btn_texts
assert "Confirm Image" in btn_texts and "Confirm As-Is" not in btn_texts
assert not hasattr(app, "export_dataset") and not hasattr(app, "train_models")

# ---- first publish: asks the name once and the shared folder once, writes everything annotated into my own folder
app.publish_to_shared()
wait(lambda: not app._bg_job and os.path.exists(os.path.join(shared, "Ian", "index.json")))
root.update()
assert app.me["name"] == "Ian" and app.project.editor_id == app.me["id"]
cfg = sz.load_config(); assert cfg["name"] == "Ian" and cfg["shared_dir"] == shared
ian_dir = os.path.join(shared, "Ian")
assert sorted(os.listdir(ian_dir)) == ["depth", "images", "index.json"] and len(os.listdir(os.path.join(ian_dir, "images"))) == 3
def index(): return json.load(open(os.path.join(ian_dir, "index.json")))
assert len(index()["images"]) == 3
assert any("Published 3 new" in t for t in info_texts()), msgs     # even though the old bookkeeping said "shared"
assert not [m for m in msgs if m[0] == "ERR"]
# nothing new -> nothing written
msgs.clear()
app.publish_to_shared(); wait(lambda: not app._bg_job); root.update()
assert len(index()["images"]) == 3 and any("already has all 3" in t for t in info_texts()), info_texts()
# a new annotation is appended to the same folder
time.sleep(1.1)
app.project.set_boxes(i4, [(1, .3, .3, .1, .1)]); msgs.clear()
app.publish_to_shared(); wait(lambda: not app._bg_job and len(index()["images"]) == 4); root.update()
assert len(os.listdir(os.path.join(ian_dir, "images"))) == 4 and any("Published 1 new" in t for t in info_texts())
# the shared folder gets emptied/moved while the project still thinks it is published: everything is sent again
shutil.rmtree(ian_dir); msgs.clear()
app.publish_to_shared(); wait(lambda: not app._bg_job and os.path.exists(os.path.join(ian_dir, "index.json"))); root.update()
assert len(index()["images"]) == 4 and any("Published 4 new" in t for t in info_texts()), info_texts()

# ---- Sam publishes his own folder: a fix of Ian's i1, a fix of i2, and new work with a new class
sam = sz.set_identity({}, "Sam")
S = ac.Project(os.path.join(tmp, "sam"), default_classes=["car", "person", "bottle"])
good, bad = sz.list_folders(shared); merged = sz.merge_folders(good)
pl = sz.analyze_entries(S, merged["entries"], merged["people"], sam); sz.apply_import(S, pl, None, sam); S.editor_id = sam["id"]
byuid = {inf["uid"]: i for i, inf in S.manifest["images"].items()}
s1 = byuid[app.project.manifest["images"][i1]["uid"]]; s2 = byuid[app.project.manifest["images"][i2]["uid"]]
time.sleep(1.1)
S.set_boxes(s1, [(1, .3, .3, .2, .2)], distances=[None])
S.set_boxes(s2, [(0, .6, .6, .2, .2), (0, .1, .1, .1, .1)])
f9 = add_frame(S, "garage", 0, 77); S.set_boxes(f9, [(2, .5, .5, .1, .1)])
r = sz.publish(S, shared, sam); assert r["count"] == 3, r
assert os.path.isdir(os.path.join(shared, "Sam"))

# ---- Ian edits i1 LATER than Sam did (a "conflict"): nothing is asked, the newest edit wins
time.sleep(1.1)
app.project.set_boxes(i1, [(1, .7, .7, .2, .2)], distances=[None])
app.load_image(i1); root.update()
msgs.clear(); asked.clear()
app.import_to_edit(); root.update()
win = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
assert win.title() == "Import data to edit"
tree = find(win, ag.Treeview)[0]
rows = [tree.item(i, "values") for i in tree.get_children()]
assert [r[0] for r in rows].count("Sam") >= 1 and any(r[:2] == ("Sam", "garage") for r in rows), rows
garage = next(i for i in tree.get_children() if tree.item(i, "values")[:2] == ("Sam", "garage"))
tree.selection_set(garage)
btns = {b.cget("text"): b for b in find(win, ttk.Button)}
btns["Import selected"].invoke()
wait(lambda: not app._bg_job and any("Import complete" == m[1][0] for m in msgs if m[0] == "info"))
root.update()
assert not asked and not [m for m in msgs if m[0] == "ERR"], msgs
P2 = app.project
fs = [i for i, inf in P2.manifest["images"].items() if inf.get("owner_id") == sam["id"]]
assert len(fs) == 1 and P2.is_annotated(fs[0]) and P2.manifest["images"][fs[0]]["session_id"] == "Sam.garage"
assert "bottle" in P2.classes and app.class_listbox.size() == 3
app.show_annotated_var.set(True)
assert sam["id"] in app.access_people and fs[0] in app._visible_image_ids()
# only the picked set came in: Ian's i1 was not touched by Sam's older fix
assert abs(P2.get_boxes(i1)[0][1] - .7) < 1e-4

# ---- import EVERYTHING: Sam's fix of i2 (only he changed it) arrives; Ian's later edit of i1 stays
msgs.clear()
app.import_to_edit(); root.update()
win = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
{b.cget("text"): b for b in find(win, ttk.Button)}["Import everything"].invoke()
wait(lambda: not app._bg_job and any("Import complete" == m[1][0] for m in msgs if m[0] == "info"))
root.update()
assert not asked and len(P2.get_boxes(i2)) == 2 and abs(P2.get_boxes(i1)[0][1] - .7) < 1e-4
assert P2.manifest["images"][i2]["owner_id"] == app.me["id"]          # the picture is Ian's; Sam's fix is the edit
assert sz.people_name(P2, P2.manifest["images"][i2]["edited_by"]) == "Sam"

# ---- fixing somebody else's image: the original owner stays, and only edited images go into my folder
app.set_active_session("Sam.garage"); root.update(); app.load_image(fs[0]); root.update()
app.project.set_boxes(fs[0], [(2, .4, .4, .1, .1)])
inf = app.project.manifest["images"][fs[0]]
assert inf["owner_id"] == sam["id"] and inf["edited_by"] == app.me["id"]
time.sleep(1.1); msgs.clear()
app.publish_to_shared(); wait(lambda: not app._bg_job and any(e["owner_id"] == sam["id"] for e in index()["images"])); root.update()
hdr = index()
assert any(e["owner_id"] == sam["id"] and e["edited_by"] == app.me["id"] for e in hdr["images"])
# imported-but-untouched images are NOT copied into my folder: only my own and the ones I edited
assert len(hdr["images"]) == 4 + 1, len(hdr["images"])
# a collector sees one winner per image (the newest edit), with a log of what was replaced
good, bad = sz.list_folders(shared); m = sz.merge_folders(good)
assert not bad and len({e["uid"] for e in m["entries"]}) == len(m["entries"])

# ---- Sessions window: other people's data is behind Access, no Role column
app.show_annotated_var.set(True)
app.access_people.discard(sam["id"]); app._refresh_session_combo(); app._refresh_image_list()
assert fs[0] not in app._visible_image_ids() and all("Sam." not in v for v in app.session_combo.cget("values"))
app.manage_sessions(); root.update()
w = app._sessions_win; tree = find(w, ag.Treeview)[0]
assert "role" not in tree["columns"]
person = f"person:{sam['id']}"
assert person in tree.get_children() and "[Sam] data" in tree.item(person, "values")[0]
tree.selection_set(person); app._sessions_access(); root.update()
assert "Sam.garage" in tree.get_children() and fs[0] in app._visible_image_ids()
tree.selection_set(person); app._sessions_access(); root.update()
assert "Sam.garage" not in tree.get_children()
w.destroy(); root.update()

# ---- name change keeps the id
simpledialog.askstring = lambda *a, **k: "Ian_G"
old_id = app.me["id"]; app.change_my_name()
assert app.me["name"] == "Ian_G" and app.me["id"] == old_id and sz.load_config()["name"] == "Ian_G"
print("SHARE GUI OK"); root.destroy(); shutil.rmtree(tmp, ignore_errors=True)
