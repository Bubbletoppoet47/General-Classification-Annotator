import os, sys, tempfile, shutil, time
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac, annotator_gui as ag

tmp = tempfile.mkdtemp(); pdir = os.path.join(tmp, "proj")
P = ac.Project(pdir, default_classes=["person", "car", "bike"])
ids = [P.add_uploaded_image(Image.fromarray(np.full((300, 400, 3), 100 + 20 * i, np.uint8)), f"i{i}") for i in range(3)]
P.set_boxes(ids[0], [(0, .5, .5, .2, .3), (1, .2, .2, .1, .1)])
del P

messagebox.showinfo = lambda *a, **k: None; messagebox.showerror = lambda *a, **k: (_ for _ in ()).throw(AssertionError(a))
messagebox.askyesno = lambda *a, **k: True
ag.messagebox = messagebox
root = tk.Tk(); root.geometry("1280x820")
app = ag.AnnotatorApp(root, pdir)
root.update()

def press(seq):
    root.focus_force(); root.update(); root.event_generate(seq); root.update()

# defaults still work: Right = next image, 2 = class 1, d toggles the LiDAR switch
app.show_annotated_var.set(True); root.update()
order = app._visible_image_ids(); first = order[0]; assert len(order) == 3, (order, ids)
app.load_image(first); root.update()
press("<Right>"); assert app.current_image_id == order[1]
press("<Left>"); assert app.current_image_id == first
press("<Key-2>"); assert app.current_class_id == 1
press("<d>"); assert app.pick_var.get(); press("<p>"); assert not app.pick_var.get()
assert "[1] person" in app.class_listbox.get(0) and "(D/P)" in app.pick_check.cget("text")

# open the window; change 'next image' to n
app.edit_hotkeys(); root.update()
win = app._hotkeys_win
def find(w, cls):
    out = []
    for c in w.winfo_children():
        if isinstance(c, cls): out.append(c)
        out += find(c, cls)
    return out
tree = find(win, ag.Treeview)[0]
assert tree.item("next_image", "values")[0] == "→" and tree.item("class:2", "values")[0] == "3"
assert len(tree.get_children()) == 6 + 3
tree.selection_set("next_image")
btn = {b.cget("text"): b for b in find(win, ag.ttk.Button)}
btn["Change..."].invoke(); root.update()
cap = app._hotkey_capture_win
cap.focus_force(); root.update()
cap.event_generate("<KeyPress>", keysym="Shift_L"); root.update()           # a bare modifier waits for the real key
assert cap.winfo_exists()
cap.event_generate("<KeyPress>", keysym="n"); root.update()
assert not cap.winfo_exists()
assert app.hotkeys.get("next_image") == "n" and tree.item("next_image", "values")[0] == "N"
assert ac.Project(pdir, default_classes=[]).load_hotkeys() == {"next_image": "n"}
app.load_image(first); root.update()
press("<Right>"); assert app.current_image_id == first                       # old key no longer does it
press("<n>"); assert app.current_image_id == order[1]                         # new one does

# give class 'bike' the key 'b' and take 'd' for class 'car': d stops toggling LiDAR
app._apply_hotkey("class:2", "b"); app._apply_hotkey("class:1", "d"); root.update()
assert app.hotkeys.get("toggle_pick") is None and "(P)" in app.pick_check.cget("text")
assert "[B] bike" in app.class_listbox.get(2) and "[D] car" in app.class_listbox.get(1)
press("<b>"); assert app.current_class_id == 2
press("<d>"); assert app.current_class_id == 1 and not app.pick_var.get()
press("<Key-3>"); assert app.current_class_id == 1                            # 3 belonged to bike, which now uses b
press("<Key-1>"); assert app.current_class_id == 0                            # other number keys unchanged
# the key on a selected box relabels it, through the new key as before
app.load_image(ids[0]); root.update(); app.selected_box_index = 0
press("<b>"); assert app.boxes[0][0] == 2

# clear / default / all defaults
tree.selection_set("class:1"); btn["Clear"].invoke(); root.update()
assert app.hotkeys.get("class:1") is None and tree.item("class:1", "values")[0] == "(none)"
btn["Default"].invoke(); root.update()
assert app.hotkeys.get("class:1") == "2"
btn["All defaults"].invoke(); root.update()
assert app.hotkeys.to_dict() == {} and app.hotkeys.get("next_image") == "Right"
assert ac.Project(pdir, default_classes=[]).load_hotkeys() == {}
press("<Right>"); press("<d>"); assert app.pick_var.get()
win.destroy(); root.update(); assert app._hotkeys_win is None

# a second app on the same project loads the saved keys
app._apply_hotkey("delete_box", "x"); root.update()
root.destroy()
root = tk.Tk(); root.geometry("1280x820")
app = ag.AnnotatorApp(root, pdir); root.update()
assert app.hotkeys.get("delete_box") == "x" and app.hotkeys.get("next_image") == "Right"
app.load_image(ids[0]); root.update(); app.selected_box_index = 0
n = len(app.boxes); root.focus_force(); root.update(); root.event_generate("<x>"); root.update()
assert len(app.boxes) == n - 1
print("HOTKEYS GUI OK"); root.destroy(); shutil.rmtree(tmp, ignore_errors=True)
