import os, sys, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import hotkeys as hk
import annotator_core as ac

# ---- key text <-> Tk sequence <-> label
assert hk.from_event("d", 0) == "d"
assert hk.from_event("D", 0) == "d"                 # caps lock alone changes nothing
assert hk.from_event("D", 0x1) == "Shift+d"
assert hk.from_event("z", 0x4) == "Ctrl+z"
assert hk.from_event("Left", 0) == "Left"
assert hk.from_event("F5", 0x1) == "Shift+F5"
assert hk.from_event("Control_L", 0x4) is None and hk.from_event("Shift_R", 1) is None
assert hk.to_tk("d") == "<Key-d>" and hk.to_tk("Delete") == "<Key-Delete>" and hk.to_tk("5") == "<Key-5>"
assert hk.to_tk("Ctrl+z") == "<Control-Key-z>" and hk.to_tk("Shift+d") == "<Key-D>"
assert hk.to_tk("Ctrl+Shift+d") == "<Control-Key-D>" and hk.to_tk("Shift+F5") == "<Shift-Key-F5>"
assert hk.label("d") == "D" and hk.label("Left") == "←" and hk.label("Ctrl+z") == "Ctrl+Z" and hk.label(None) == ""

# ---- defaults are exactly what the annotator always had
h = hk.Hotkeys()
assert h.get("delete_box") == "Delete" and h.get("toggle_pick") == "d" and h.get("toggle_depth") == "p"
assert h.get("prev_image") == "Left" and h.get("next_image") == "Right"
assert [h.get(f"class:{i}") for i in range(11)] == list("1234567890") + [None]
b = h.bindings(4)
assert b["<Key-1>"] == "class:0" and b["<Key-4>"] == "class:3" and "<Key-5>" not in b and b["<Key-Delete>"] == "delete_box"
assert h.to_dict() == {}

# ---- changing a key; the old one stops working, nothing else moves
assert h.set("next_image", "n", 4) == []
assert h.get("next_image") == "n" and "<Key-Right>" not in h.bindings(4) and h.bindings(4)["<Key-n>"] == "next_image"
assert h.to_dict() == {"next_image": "n"}
# giving a used key to another action takes it away from the first one
assert h.set("prev_image", "n", 4) == ["next_image"]
assert h.get("prev_image") == "n" and h.get("next_image") is None
assert h.to_dict() == {"prev_image": "n", "next_image": ""}
assert len(set(h.bindings(4))) == len(h.bindings(4))                   # a key never maps to two things
# a class can take an unused key, and a default key of another action counts as used
assert h.set("class:1", "d", 4) == ["toggle_pick"] and h.get("toggle_pick") is None and h.get("class:1") == "d"
# clearing, resetting
assert h.set("delete_box", None, 4) == [] and h.get("delete_box") is None and "<Key-Delete>" not in h.bindings(4)
h.reset("delete_box", 4); assert h.get("delete_box") == "Delete" and "delete_box" not in h.to_dict()
h.reset("next_image", 4)                                              # default Right is free again
assert h.get("next_image") == "Right" and "next_image" not in h.to_dict()
h.reset_all(); assert h.to_dict() == {} and h.get("toggle_pick") == "d"
# resetting an action whose default key is now used elsewhere takes the key back from that one
h.set("class:0", "Left", 4); assert h.get("prev_image") is None
assert h.reset("prev_image", 4) == ["class:0"] and h.get("prev_image") == "Left" and h.get("class:0") is None
# classes beyond the tenth can get keys; junk in a saved file is ignored
h2 = hk.Hotkeys({"class:11": "q", "next_image": 5, 3: "x", "delete_box": "x"})
assert h2.get("class:11") == "q" and h2.get("next_image") == "Right" and h2.get("delete_box") == "x"
assert h2.bindings(12)["<Key-q>"] == "class:11" and "<Key-q>" not in h2.bindings(5)

# ---- saved with the project
tmp = tempfile.mkdtemp()
try:
    P = ac.Project(os.path.join(tmp, "p"), default_classes=["a", "b"])
    assert P.load_hotkeys() == {}
    P.save_hotkeys({"next_image": "n", "class:0": ""})
    assert ac.Project(os.path.join(tmp, "p"), default_classes=[]).load_hotkeys() == {"next_image": "n", "class:0": ""}
    with open(os.path.join(tmp, "p", "hotkeys.json"), "w") as f:
        f.write("not json")
    assert P.load_hotkeys() == {}
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("HOTKEYS OK")
