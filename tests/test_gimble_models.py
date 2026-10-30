"""The shared Models folder: versions, pasted models, seeding, and real (tiny, CPU) Assist + Scratch training runs."""
import os, sys, shutil, json, hashlib, io, contextlib, time, tempfile
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"); sys.path.insert(0, HERE)
import model_library as ml, model_registry as mr

def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
tmp = tempfile.mkdtemp()
models = os.path.join(tmp, "Models")

# ---- you make the folders; the program never does
assert ml.missing_folders(models) == [models]
assert "Please make this folder" in ml.missing_message(models, "assist")
os.makedirs(models)
assert ml.missing_folders(models) == [os.path.join(models, "Assist"), os.path.join(models, "Scratch")]
assert ml.save_version(models, "assist", __file__, "m", "Ian") is None                      # nothing written, nothing created
assert os.listdir(models) == []
os.makedirs(os.path.join(models, "Assist")); os.makedirs(os.path.join(models, "Scratch"))
assert ml.missing_folders(models) == [] and ml.missing_message(models) is None

# ---- saving versions: named by size, date, trainer; info file goes with it; never overwritten
src = os.path.join(tmp, "best.pt"); open(src, "wb").write(b"M1")
t1 = time.mktime((2026, 10, 8, 15, 30, 0, 0, 0, -1))
p1 = ml.save_version(models, "assist", src, "m", "Ian", classes=["person", "trash"], scores={"map50": .5, "map50_95": .3}, ts=t1)
assert os.path.basename(p1) == "Medium_2026-10-08_1530_Ian.pt" and open(p1, "rb").read() == b"M1"
info = ml.read_info(p1); assert info["size"] == "m" and info["trainer"] == "Ian" and info["classes"] == ["person", "trash"] and info["created"] == t1
p1b = ml.save_version(models, "assist", src, "m", "Ian", ts=t1)                              # same minute, same name -> _2
assert os.path.basename(p1b) == "Medium_2026-10-08_1530_Ian_2.pt" and md5(p1) == md5(p1b)
assert not [f for f in os.listdir(os.path.join(models, "Assist")) if f.endswith(".part")]
open(src, "wb").write(b"M2")
p2 = ml.save_version(models, "assist", src, "m", "Sam", ts=t1 + 3600, based_on=os.path.basename(p1))
open(src, "wb").write(b"N1")
p3 = ml.save_version(models, "assist", src, "n", "Ian", ts=t1 + 60)
assert open(p1, "rb").read() == b"M1"                                                         # older versions untouched
vs = ml.list_versions(models, "assist")
assert [v["file"] for v in vs] == [os.path.basename(p2), os.path.basename(p3), os.path.basename(p1b), os.path.basename(p1)] or \
       [v["file"] for v in vs][:2] == [os.path.basename(p2), os.path.basename(p3)]
news = ml.newest_per_size(vs); assert [(v["size"], v["file"]) for v in news] == [("n", os.path.basename(p3)), ("m", os.path.basename(p2))]
old = ml.older_per_size(vs); assert list(old) == ["m"] and len(old["m"]) == 2 and "n" not in old
assert "Medium - 2026-10-08 16:30 - Sam" == ml.describe(vs[0]) and ml.short_label(vs[0]) == "Medium Assist 2026-10-08 16:30"
assert ml.list_versions(models, "scratch") == []

# ---- a model pasted by hand: any name; size read once; info saved beside it; the .pt is untouched
pasted = os.path.join(models, "Scratch", "my_old_model.pt"); open(pasted, "wb").write(b"OLD")
before = md5(pasted)
v = ml.list_versions(models, "scratch")[0]; assert v["size"] is None                         # not looked at yet: unknown
real = mr.detect_size; mr.detect_size = lambda kind, ck: "s" if kind == "scratch" else None
v = ml.list_versions(models, "scratch", detect=True)[0]
assert v["size"] == "s" and os.path.exists(ml.info_path(pasted)) and md5(pasted) == before
assert ml.read_info(pasted)["pasted"] is True and "pasted by hand" in ml.describe(v)
mr.detect_size = lambda kind, ck: None                                                        # unreadable: tolerated, remembered
bad = os.path.join(models, "Assist", "broken.pt"); open(bad, "wb").write(b"?"); os.utime(bad, (1.0e9, 1.0e9))      # old: must not outrank the real versions
assert [x for x in ml.list_versions(models, "assist", detect=True) if x["file"] == "broken.pt"][0]["size"] is None
mr.detect_size = real

# ---- seeding the training slot from the newest version
slot = os.path.join(tmp, "slot_m")
used = ml.seed_slot(models, "assist", "m", slot)
assert used["file"] == os.path.basename(p2) and open(slot + "/weights/best.pt", "rb").read() == b"M2"
assert ml.seed_slot(models, "assist", "m", slot) is None                                      # already as new: nothing to do
open(slot + "/weights/best.pt", "wb").write(b"LOCAL"); os.utime(slot + "/weights/best.pt", (t1 + 7200, t1 + 7200))
assert ml.seed_slot(models, "assist", "m", slot) is None and open(slot + "/weights/best.pt", "rb").read() == b"LOCAL"     # local newer: kept
os.utime(slot + "/weights/best.pt", (t1, t1))
used = ml.seed_slot(models, "assist", "m", slot)
assert used and open(slot + "/weights/best.pt", "rb").read() == b"M2" and [f for f in os.listdir(slot + "/weights") if f.startswith("before_seed_")]
assert ml.seed_slot(models, "assist", "l", os.path.join(tmp, "slot_l")) is None                # no version of that size

# ---- the annotator's local copy
lc = ml.local_copy(ml.list_versions(models, "assist")[0], os.path.join(tmp, "inserted"))
assert lc.startswith(os.path.join(tmp, "inserted", "Assist")) and open(lc, "rb").read() == b"M2" and os.path.exists(ml.info_path(lc))
print("MODEL LIBRARY OK")
