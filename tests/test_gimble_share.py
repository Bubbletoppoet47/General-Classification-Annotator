import os, sys, tempfile, shutil, json, zipfile, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac
import share_zip as sz

tmp = tempfile.mkdtemp()
shared = os.path.join(tmp, "drive"); os.makedirs(shared)


def make_project(name, classes):
    return ac.Project(os.path.join(tmp, name), default_classes=classes)


def add_frame(P, session, idx, seed, depth=True):
    arr = np.random.default_rng(seed).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    iid = P.add_uploaded_image(Image.fromarray(arr), f"f{seed}")
    info = P.manifest["images"][iid]
    info.update(source="gimble", session_id=session, sample_index=idx, source_name=session)
    if depth:
        d = np.zeros((60, 80), np.uint16); d[10:20, 10:20] = 2500
        Image.fromarray(d).save(P.depth_path(iid))
        info.update(has_depth=True, depth_pixels=100)
    P.save_manifest()
    return iid


def pick(d):
    return {"distance_m": d, "u": 12, "v": 12}


def import_for_edit(P, me, picks=None):
    packs, bad = sz.list_folders(shared)
    merged = sz.merge_folders(packs)
    plan = sz.analyze_entries(P, sz.select_entries(merged, picks), merged["people"], me)
    return plan, sz.apply_import(P, plan, None, me)


# ---- names / config
assert sz.person_name_error("Ian") is None
assert sz.person_name_error("Sam Lee") and sz.person_name_error("a.b") and sz.person_name_error("")
cfgA, cfgB = {}, {}
ian = sz.set_identity(cfgA, "Ian"); sam = sz.set_identity(cfgB, "Sam")
assert ian["id"] != sam["id"] and sz.set_identity(cfgA, "Ian2")["id"] == ian["id"] and cfgA["name"] == "Ian2"
cfgA["name"] = "Ian"; ian = sz.identity(cfgA)
cp = os.path.join(tmp, "cfg.json"); sz.save_config(cfgA, cp); assert sz.load_config(cp)["id"] == ian["id"] and sz.load_config(cp + "x") == {}

# ---- Ian: 3 annotated frames, 1 not
A = make_project("A", ["person", "car"])
a1, a2, a3, a4 = (add_frame(A, "kitchen", i, 10 + i) for i in range(4))
A.set_boxes(a1, [(0, .5, .5, .2, .3), (1, .2, .2, .1, .1)], distances=[pick(2.5), None])
A.set_boxes(a2, [(1, .4, .4, .2, .2)])
A.set_boxes(a3, [])
A.editor_id = ian["id"]
r1 = sz.publish(A, shared, ian)
assert r1["count"] == 3 and r1["new"] == 3 and r1["edits"] == 0
ian_dir = os.path.join(shared, "Ian")
assert r1["path"] == ian_dir and sorted(os.listdir(ian_dir)) == ["depth", "images", "index.json"]      # plain folders, no zip
assert not [f for _, _, fs in os.walk(ian_dir) for f in fs if f.endswith(".part")]
hdr = json.load(open(os.path.join(ian_dir, "index.json")))
assert hdr["owner"]["name"] == "Ian" and len(hdr["images"]) == 3 and hdr["classes"] == ["person", "car"]
assert len(os.listdir(os.path.join(ian_dir, "images"))) == 3 and len(os.listdir(os.path.join(ian_dir, "depth"))) == 3
assert not A.manifest["images"][a4].get("uid")                       # unannotated: not shared
e1 = next(e for e in hdr["images"] if e["uid"] == A.manifest["images"][a1]["uid"])
assert e1["boxes"][0][0] == "person" and e1["distances"][0]["distance_m"] == 2.5 and e1["set"] == "kitchen" and e1["size"] > 0
# nothing changed -> nothing to publish
assert sz.publish(A, shared, ian)["count"] == 0
# one fix -> only that image is rewritten
time.sleep(0.02)
mt = {f: os.path.getmtime(os.path.join(ian_dir, "images", f)) for f in os.listdir(os.path.join(ian_dir, "images"))}
A.set_boxes(a2, [(1, .45, .45, .2, .2)])
r2 = sz.publish(A, shared, ian)
assert r2["count"] == 1 and r2["new"] == 0 and r2["edits"] == 1 and len(os.listdir(os.path.join(ian_dir, "images"))) == 3
folders, bad = sz.list_folders(shared)
assert len(folders) == 1 and not bad
# the folder is the truth, not a note in the project: stale "already shared" notes change nothing ...
for iid in (a1, a2, a3):
    A.manifest["images"][iid]["share_base"] = sz.local_state(A, iid)
assert sz.publish(A, shared, ian)["count"] == 0
# ... a deleted picture file is put back, and a deleted folder is rebuilt completely, appending only what's needed
uid1 = A.manifest["images"][a1]["uid"]
os.remove(os.path.join(ian_dir, "images", uid1 + ".jpg"))
r3 = sz.publish(A, shared, ian); assert r3["count"] == 1 and os.path.exists(os.path.join(ian_dir, "images", uid1 + ".jpg"))
shutil.copytree(ian_dir, os.path.join(tmp, "ian_backup"))
shutil.rmtree(ian_dir)
r4 = sz.publish(A, shared, ian); assert r4["count"] == 3 and r4["new"] == 3 and len(json.load(open(os.path.join(ian_dir, "index.json")))["images"]) == 3
# a project that has never published anything (but is marked from the old sharing) still writes everything to an empty folder
shutil.rmtree(ian_dir); os.makedirs(ian_dir)
assert sz.publish(A, shared, ian)["count"] == 3

# ---- the training program collects: newest edit of each image, nothing asked
W = make_project("W", [])
rep = sz.collect_into(W, shared)
assert rep["new"] == 3 and rep["folders"] == 1 and rep["total"] == 3 and rep["pending"] == 0
assert W.classes == ["person", "car"] and len(W.list_images()) == 3
sets = {r["session_id"]: r for r in W.session_summary()}
assert list(sets) == ["Ian.kitchen"] and sets["Ian.kitchen"]["samples"] == 3 and sets["Ian.kitchen"]["with_depth"] == 3
w2 = next(i for i, inf in W.manifest["images"].items() if inf["uid"] == A.manifest["images"][a2]["uid"])
assert abs(W.get_boxes(w2)[0][1] - .45) < 1e-5                         # the later pack's fix
assert W.manifest["images"][w2]["owner_id"] == ian["id"]
# validation is a training-side decision
W.set_session_role("Ian.kitchen", "val")
res = W.export_dataset(include_distortions=False)
assert res["n_val_sources"] == 3 and res["train_count"] == 0 and res["val_count"] == 3
W.set_session_role("Ian.kitchen", "train")
# collecting again changes nothing
rep = sz.collect_into(W, shared); assert rep["new"] == 0 and rep["updated"] == 0 and len(W.list_images()) == 3

# ---- Sam imports Ian's kitchen to edit it (class lists differ), fixes one image, adds his own work
B = make_project("B", ["car", "person", "bottle"])
B.editor_id = sam["id"]
merged = sz.merge_folders(sz.list_folders(shared)[0])
av = sz.available_sets(merged)
assert [(g["name"], g["set"], g["images"]) for g in av] == [("Ian", "kitchen", 3)]
plan, rep = import_for_edit(B, sam, picks=[(ian["id"], "kitchen")])
assert rep["new"] == 3 and len(B.manifest["images"]) == 3
bi1 = next(i for i, inf in B.manifest["images"].items() if inf["uid"] == e1["uid"])
bx = B.get_boxes(bi1)
assert [B.classes[b[0]] for b in bx] == ["person", "car"]                    # mapped BY NAME
assert B.get_distances(bi1)[0]["distance_m"] == 2.5 and B.has_depth(bi1) and B.manifest["images"][bi1]["session_id"] == "Ian.kitchen"
# importing again, or publishing right after importing, does nothing (untouched imports are not re-published)
plan, rep = import_for_edit(B, sam); assert rep["new"] == 0 and rep["updated"] == 0 and plan["same"] == 3
assert sz.publish(B, shared, sam)["count"] == 0
time.sleep(0.02)
B.set_boxes(bi1, [(1, .5, .5, .25, .3), (0, .2, .2, .1, .1)], distances=[None, None])        # Sam's fix
f9 = add_frame(B, "garage", 0, 77); B.set_boxes(f9, [(2, .5, .5, .1, .1)])                   # his own work, new class
assert B.manifest["images"][bi1]["owner_id"] == ian["id"] and B.manifest["images"][bi1]["edited_by"] == sam["id"]
rs = sz.publish(B, shared, sam)
assert rs["count"] == 2 and rs["new"] == 1 and rs["edits"] == 1 and rs["path"] == os.path.join(shared, "Sam")

# ---- the trainer: Sam's fix wins (edited last), Sam's frame arrives under his name, the log says what was dropped
rep = sz.collect_into(W, shared)
assert rep["new"] == 1 and rep["updated"] == 1 and rep["folders"] == 2
assert {r["session_id"] for r in W.session_summary()} == {"Ian.kitchen", "Sam.garage"} and "bottle" in W.classes
w1 = next(i for i, inf in W.manifest["images"].items() if inf["uid"] == e1["uid"])
assert abs(W.get_boxes(w1)[0][3] - .25) < 1e-5 and W.manifest["images"][w1]["owner_id"] == ian["id"]
assert any(o["dropped_from"] == "Ian" and o["kept_from"] == "Sam" and o["uid"] == e1["uid"] for o in rep["overridden"])

# ---- Ian: edits collide (both changed): the newer edit wins automatically
time.sleep(0.02)
A.set_boxes(a2, [(0, .3, .3, .1, .1)])                      # Ian edits a2 (newer than anything published)
plan, rep = None, None
packs, _ = sz.list_folders(shared); merged = sz.merge_folders(packs)
plan = sz.analyze_entries(A, merged["entries"], merged["people"], ian)
uidc = {c["uid"] for c in plan["conflicts"]}
assert uidc == set() and plan["kept_local"] == 1                  # a2: only Ian changed it since the pack -> his stays
# a1: Sam fixed it, Ian did not touch it -> simply updated from the pack; Sam's new frame is new
assert len(plan["new"]) == 1 and any(e["uid"] == e1["uid"] for e, _ in plan["update"])
rep = sz.apply_import(A, plan, None, ian)
assert abs(A.get_boxes(a1)[0][3] - .25) < 1e-5 and A.manifest["images"][a1]["owner_id"] == ian["id"]
assert sz.people_name(A, A.manifest["images"][a1]["edited_by"]) == "Sam"
assert len(A.get_boxes(a2)) == 1 and abs(A.get_boxes(a2)[0][1] - .3) < 1e-5            # his own unpublished newer edit stays
# real conflict: Ian's local edit is OLDER than Sam's pack -> Sam's wins; the other way round -> Ian's wins and is published
a5 = add_frame(A, "kitchen", 5, 55); time.sleep(0.02)
A.set_boxes(a5, [(0, .1, .1, .1, .1)]); sz.publish(A, shared, ian)
bi5 = None
plan, rep = import_for_edit(B, sam, picks=[(ian["id"], "kitchen")])
bi5 = next(i for i, inf in B.manifest["images"].items() if inf.get("uid") == A.manifest["images"][a5]["uid"])
time.sleep(0.02); A.set_boxes(a5, [(0, .9, .9, .1, .1)])                   # Ian edits locally first ...
time.sleep(0.02); B.set_boxes(bi5, [(1, .5, .5, .1, .1)]); sz.publish(B, shared, sam)       # ... Sam edits and publishes later
packs, _ = sz.list_folders(shared); merged = sz.merge_folders(packs)
plan = sz.analyze_entries(A, merged["entries"], merged["people"], ian)
assert [c["uid"] for c in plan["conflicts"]] == [A.manifest["images"][a5]["uid"]]
rep = sz.apply_import(A, plan, None, ian)
assert abs(A.get_boxes(a5)[0][1] - .5) < 1e-5 and rep["took_theirs"] == 1
time.sleep(0.02); A.set_boxes(a5, [(0, .2, .2, .1, .1)])                    # now Ian is the newest
packs, _ = sz.list_folders(shared); merged = sz.merge_folders(packs)
plan = sz.analyze_entries(A, merged["entries"], merged["people"], ian)
rep = sz.apply_import(A, plan, None, ian)
assert abs(A.get_boxes(a5)[0][1] - .2) < 1e-5 and sz.publish(A, shared, ian)["count"] >= 1

# ---- same picture, not annotated yet, is linked instead of duplicated; a picture shared under another id is skipped
C = make_project("C", ["person", "car"])
shutil.copyfile(A.image_path(a3), os.path.join(C.images_dir, "unannotated", "img_1.jpg"))
C.manifest["images"]["img_1"] = {"source": "upload", "annotated": False, "ai_pending": False, "has_depth": False, "width": 80, "height": 60}
C.save_manifest()
packs, _ = sz.list_folders(shared); merged = sz.merge_folders(packs)
cplan = sz.analyze_entries(C, sz.select_entries(merged, [(ian["id"], "kitchen")]), merged["people"], sam)
assert len(cplan["link"]) == 1
n_ian = len(sz.select_entries(merged, [(ian["id"], "kitchen")]))
rep = sz.apply_import(C, cplan, None, sam)
assert len(C.manifest["images"]) == n_ian and C.is_annotated("img_1") and C.manifest["images"]["img_1"]["owner_id"] == ian["id"]
D = make_project("D", ["person", "car"])
shutil.copyfile(A.image_path(a3), os.path.join(D.images_dir, "unannotated", "img_1.jpg"))
D.manifest["images"]["img_1"] = {"source": "upload", "annotated": False, "ai_pending": False, "has_depth": False, "width": 80, "height": 60}
D.save_manifest(); D.set_boxes("img_1", [(0, .5, .5, .1, .1)])
dee = sz.set_identity({}, "Dee"); sz.publish(D, shared, dee)
packs, _ = sz.list_folders(shared); merged = sz.merge_folders(packs)
pl = sz.analyze_entries(C, sz.select_entries(merged, [(dee["id"], "uploads")]), merged["people"], sam)
assert pl["duplicates"] == 1 and not pl["new"]

# ---- two different people called Ian get separate folders; unreadable packs are reported, not fatal
ian2 = sz.set_identity({}, "Ian")
assert sz.person_dir(shared, ian2) != sz.person_dir(shared, ian) and sz.person_dir(shared, ian) == os.path.join(shared, "Ian")
os.makedirs(os.path.join(shared, "Half", "images"))                     # pictures arrived, index.json not yet
with open(os.path.join(shared, "Half", "images", "x.jpg"), "wb") as f: f.write(b"x")
os.makedirs(os.path.join(shared, "Docs"))                                # an unrelated folder is simply ignored
folders, bad = sz.list_folders(shared)
assert len(bad) == 1 and "Half" in bad[0][0] and len(folders) == 3 and not any("Docs" in b[0] for b in bad)
shutil.rmtree(os.path.join(shared, "Half")); shutil.rmtree(os.path.join(shared, "Docs"))
# an index entry whose picture has not synced yet is left out (and counted), not fatal
shutil.copytree(os.path.join(shared, "Sam"), os.path.join(tmp, "partial", "Sam"))
sd = os.path.join(tmp, "partial", "Sam"); os.remove(os.path.join(sd, "images", sorted(os.listdir(os.path.join(sd, "images")))[0]))
pm = sz.merge_folders(sz.list_folders(os.path.join(tmp, "partial"))[0]); assert pm["pending"] == 1

# ---- a picture that was damaged in the folder is refused
tdir = os.path.join(tmp, "tampered", "Bob"); shutil.copytree(os.path.join(shared, "Ian"), tdir)
fn = os.path.join(tdir, "images", sorted(os.listdir(os.path.join(tdir, "images")))[0])
data = open(fn, "rb").read(); open(fn, "wb").write(data[:-5] + b"xxxxx")
F = make_project("F", ["person", "car"])
tp, _ = sz.list_folders(os.path.join(tmp, "tampered")); tm = sz.merge_folders(tp)
try:
    sz.apply_import(F, sz.analyze_entries(F, tm["entries"], tm["people"], sam), None, sam); assert False
except ValueError as ex:
    assert "damaged" in str(ex)

# ---- projects still open and export as before
P2 = ac.Project(A.root, default_classes=[])
assert P2.manifest["images"][a1]["uid"]
P2.export_dataset(include_distortions=False)
print("SHARE CORE OK"); shutil.rmtree(tmp, ignore_errors=True)
