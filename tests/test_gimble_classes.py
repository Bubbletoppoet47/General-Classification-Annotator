"""Class management: add / rename / delete (remove or move boxes), distances, exports, persistence."""
import os, sys, tempfile, shutil
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac

tmp = tempfile.mkdtemp()
P = ac.Project(os.path.join(tmp, "proj"), default_classes=["person", "car", "bike", "trash"])
def add_img(name):
    iid = P.add_uploaded_image(Image.fromarray(np.full((60, 80, 3), 120, np.uint8)), name); return iid
a, b, c, d = (add_img(n) for n in "abcd")
pick = lambda m: {"distance_m": m, "u": 1, "v": 2}
P.set_boxes(a, [(0, .5, .5, .2, .2), (1, .3, .3, .1, .1), (3, .7, .7, .1, .1)], distances=[pick(2.0), pick(3.0), pick(4.0)])
P.set_boxes(b, [(1, .5, .5, .2, .2)], distances=[pick(5.0)])
P.set_boxes(c, [(2, .4, .4, .2, .2), (3, .6, .6, .1, .1)])
P.generate_distortions(a, 2, seed=1)
P.set_ai_predicted_boxes(d, [(1, .5, .5, .2, .2), (2, .2, .2, .1, .1)])     # a draft, not annotated
assert P.class_box_counts() == [1, 3, 2, 2], P.class_box_counts()

# validation
assert P.validate_class_name("") and P.validate_class_name("two words") and P.validate_class_name("CAR", P.classes)
assert P.validate_class_name("a" * 41) and P.validate_class_name("ok_name-2", P.classes) is None
try: P.add_class("bad:name"); raise SystemExit("should fail")
except ValueError: pass
P.add_class("bench"); P.add_class("bench"); assert P.classes == ["person", "car", "bike", "trash", "bench"]
assert ac.Project(os.path.join(tmp, "proj"), default_classes=[]).classes == P.classes        # persisted

# exports: a class-filtered export containing 'car'
P.export_dataset(classes=["person", "car"]); assert os.path.isdir(os.path.join(P.root, "exports", "person_car"))
# rename
assert P.rename_class(1, "vehicle") == "car" and P.classes[1] == "vehicle"
assert os.path.isdir(os.path.join(P.root, "exports", "person_vehicle")) and not os.path.exists(os.path.join(P.root, "exports", "person_car"))
P.refresh_exports(); y = open(os.path.join(P.root, "exports", "person_vehicle", "data.yaml")).read(); assert "vehicle" in y and "car" not in y
for bad in (("person",), ("VEHICLE",), ("x y",)):
    try: P.rename_class(2, *bad); raise SystemExit("should fail")
    except ValueError as e: print("refused:", e)
assert [b[0] for b in P.get_boxes(b)] == [1]                     # labels untouched by a rename

# delete 'bike' (idx 2), moving its boxes to 'person' (idx 0): trash 3->2, bench 4->3
r = P.delete_class(2, move_to=0); print(r)
assert P.classes == ["person", "vehicle", "trash", "bench"] and r["moved"] == 2 and r["removed"] == 0
assert [x[0] for x in P.get_boxes(c)] == [0, 2] and [x[0] for x in P.get_boxes(d)] == [1, 0]
assert [x[0] for x in P.get_boxes(a)] == [0, 1, 2]
assert [p["distance_m"] for p in P.get_distances(a)] == [2.0, 3.0, 4.0]
assert os.path.isdir(r["backup"]) and os.path.exists(os.path.join(r["backup"], "classes.txt"))
assert ac.Project(os.path.join(tmp, "proj"), default_classes=[]).classes == P.classes
# delete 'vehicle' (idx 1), removing its boxes: image b loses its only box -> back to the queue; a keeps picks aligned
r = P.delete_class(1); print(r)
assert P.classes == ["person", "trash", "bench"] and r["removed"] == 3 and r["unannotated"] == 1
assert not P.is_annotated(b) and P.get_boxes(b) == []
assert [x[0] for x in P.get_boxes(a)] == [0, 1] and [p["distance_m"] for p in P.get_distances(a)] == [2.0, 4.0]
for v in P.manifest["distortions"].get(a, []):
    assert [x[0] for x in ac.load_label_file(os.path.join(P.dist_labels_dir, f"{v}.txt"))] == [0, 1]
assert not os.path.exists(os.path.join(P.root, "exports", "person_vehicle"))        # export of a deleted class is dropped
assert [x[0] for x in P.get_boxes(d)] == [0] and P.is_ai_pending(d)
# export still works and the numbers are all valid
res = P.export_dataset(); assert res["classes"] == ["person", "trash", "bench"]
for i in P.list_images():
    assert all(0 <= x[0] < 3 for x in P.get_boxes(i))
try: P.delete_class(0, move_to=0); raise SystemExit("should fail")
except ValueError: pass
try:
    Q = ac.Project(os.path.join(tmp, "q"), default_classes=["only"]); Q.delete_class(0); raise SystemExit("should fail")
except ValueError as e: print("refused:", e)
print("CLASSES CORE OK"); shutil.rmtree(tmp, ignore_errors=True)
