import os, sys, tempfile, time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_gui as ag

tmp = tempfile.mkdtemp()
ag.filedialog, ag.messagebox = filedialog, messagebox
root = tk.Tk(); root.withdraw()

def run(buttons, pick=None, yes=True):
    """Opens the real 'Link your dataset' window and presses the buttons (in order) after it shows."""
    filedialog.askdirectory = lambda **k: pick
    messagebox.askyesno = lambda *a, **k: yes
    def press(i=0):
        wins = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
        if not wins or i >= len(buttons):
            return
        btn = [c for c in wins[0].winfo_children() if isinstance(c, ttk.Button) and c.cget("text").startswith(buttons[i])][0]
        btn.invoke()
        root.after(100, lambda: press(i + 1))
    root.after(100, press)
    return ag.ask_dataset_location(root)

ds = os.path.join(tmp, "data"); os.makedirs(ds); open(os.path.join(ds, "manifest.json"), "w").write("{}")
assert run(["Link an existing"], ds) == os.path.normpath(ds)                        # a real dataset: linked as is
empty = os.path.join(tmp, "empty"); os.makedirs(empty)
assert run(["Link an existing"], empty, yes=True) == os.path.normpath(empty)        # no manifest: asks, yes = a new one there
assert run(["Link an existing", "Quit"], empty, yes=False) is None                  # no manifest, declined: still asking; Quit
newd = os.path.join(tmp, "fresh")
assert run(["Make a new"], newd) == os.path.normpath(newd)                          # a new dataset in the folder picked
assert not os.path.exists(os.path.join(newd, "manifest.json")) and not os.path.exists(newd)   # (the program makes it when it opens)
assert run(["Quit"]) is None                                                        # Quit: nothing linked
root.destroy()
print("LINK GUI OK")
