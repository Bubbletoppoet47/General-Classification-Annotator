"""Headless checks for the Gimble session import / depth sidecars in annotator_core.py."""
import csv, json, os, shutil, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from PIL import Image
import annotator_core as ac

def make_session(root, name="20260930_100000", n=6, w=640, h=480, with_lidar=True):
    d = os.path.join(root, name)
    for sub in ("images", "depth", "lidar"):
        os.makedirs(os.path.join(d, sub))
    rng = np.random.default_rng(0)
    with open(os.path.join(d, "index.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["idx","time_unix","image","depth","lidar","lidar_points_total","depth_pixels","sharpness"])
        for i in range(1, n + 1):
            img = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)); img.save(f"{d}/images/{i:06d}.jpg")
            depth = np.zeros((h, w), np.uint16)
            depth[10 + i, 20] = 1500 + i          # a marker point per sample: row 10+i, col 20
            depth[400, 600] = 4000
            Image.fromarray(depth).save(f"{d}/depth/{i:06d}.png")
            np.savez_compressed(f"{d}/lidar/{i:06d}.npz", points=rng.random((50, 4)).astype(np.float32))
            cw.writerow([i, 1790000000 + i * 0.3, f"images/{i:06d}.jpg", f"depth/{i:06d}.png", f"lidar/{i:06d}.npz", 50, 2, 100.0 + i])
    json.dump({"camera_matrix": [[240,0,320],[0,240,240],[0,0,1]]}, open(d + "/meta.json", "w"))
    return d

tmp = tempfile.mkdtemp()
sess = make_session(tmp)
p = ac.Project(os.path.join(tmp, "proj"))
assert ac.is_gimble_session(sess) and not ac.is_gimble_session(tmp)

# import with stride 2 -> samples 1,3,5
r = p.import_gimble_session(sess, stride=2)
assert len(r["new_ids"]) == 3 and r["skipped_existing"] == 0 and not r["missing"] and not r["no_depth"], r
iid = "gim_20260930_100000_000003"
assert iid in r["new_ids"]
assert os.path.exists(p.image_path(iid)) and p.has_depth(iid) and os.path.exists(p.lidar_path(iid))
assert os.path.exists(os.path.join(p.sessions_dir, "20260930_100000", "meta.json"))
info = p.manifest["images"][iid]
assert info["source"] == "gimble" and info["width"] == 640 and info["has_depth"] and info["sample_index"] == 3
# byte-identical jpg copy
assert open(p.image_path(iid), "rb").read() == open(f"{sess}/images/000003.jpg", "rb").read()
u, v, z = p.depth_points(iid)
assert len(u) == 2 and np.allclose(sorted(z.tolist()), [1.503, 4.0]), z
# re-import: nothing new for same samples; stride 1 adds the rest only
r2 = p.import_gimble_session(sess, stride=2); assert not r2["new_ids"] and r2["skipped_existing"] == 3
r3 = p.import_gimble_session(sess, stride=1); assert len(r3["new_ids"]) == 3 and r3["skipped_existing"] == 3
assert len(p.list_images()) == 6 and p.list_sessions() == [("20260930_100000", 6)]
assert p.manifest["sessions_imported"]["20260930_100000"]["samples_imported"] == 6

# annotate one: image + label move to annotated, depth stays put and readable
p.set_boxes(iid, [(0, 0.5, 0.5, 0.2, 0.2)])
assert p.is_annotated(iid) and p.has_depth(iid) and "annotated" in p.image_path(iid)

# rotate clockwise: image 480x640, depth follows, marker moves correctly
before = p.load_depth_mm(iid)
p.rotate_image(iid, clockwise=True)
after = p.load_depth_mm(iid)
assert after.shape == (640, 480) and after.dtype == np.uint16
assert Image.open(p.image_path(iid)).size == (480, 640)
# marker at (row 13, col 20) in 480x640 -> clockwise: new_row = col, new_col = H-1-row = 480-1-13
assert after[20, 480 - 1 - 13] == before[13, 20] == 1503, (after[20, 480 - 1 - 13],)
assert p.manifest["images"][iid]["quarter_turns_cw"] == 1
for _ in range(3): p.rotate_image(iid, clockwise=True)
assert (p.load_depth_mm(iid) == before).all() and p.manifest["images"][iid]["quarter_turns_cw"] == 0
p.rotate_image(iid, clockwise=False); p.rotate_image(iid, clockwise=True)

# distortions still work, depth untouched
p.generate_distortions(iid, 2)
assert len(p.manifest["distortions"][iid]) == 2

# delete removes image, label, depth, lidar, distortions
other = "gim_20260930_100000_000001"
p.delete_image(iid)
assert iid not in p.manifest["images"]
assert not any(os.path.exists(x) for x in p._sidecar_paths(iid)) and not os.path.exists(p.image_path(iid))
# reconcile: delete an image file by hand -> sidecars cleaned too
os.remove(p.image_path(other))
gone = p.reconcile_deleted_images()
assert gone == [other] and not any(os.path.exists(x) for x in p._sidecar_paths(other))

# reopen project: manifest round-trips
p2 = ac.Project(os.path.join(tmp, "proj")); assert len(p2.list_images()) == 4

# session without depth / lidar columns, missing image
s2 = make_session(tmp, "20260930_110000", n=3, with_lidar=False)
os.remove(f"{s2}/images/000002.jpg"); os.remove(f"{s2}/depth/000003.png")
r = p2.import_gimble_session(s2, copy_lidar=False)
assert r["missing"] == [2] and r["no_depth"] == [3] and len(r["new_ids"]) == 2, r
assert not p2.has_depth("gim_20260930_110000_000003") and p2.depth_points("gim_20260930_110000_000003") is None
assert not os.path.exists(p2.lidar_path("gim_20260930_110000_000001"))

# depth with wrong size is rejected (image imported, depth ignored)
s3 = make_session(tmp, "20260930_120000", n=1)
Image.fromarray(np.zeros((100, 100), np.uint16)).save(f"{s3}/depth/000001.png")
r = p2.import_gimble_session(s3); assert r["no_depth"] == [1]

# bad folder
try: p2.import_gimble_session(tmp); raise SystemExit("should have raised")
except ValueError: pass

# export still works
res = p2.export_dataset(ensure_distortions=False)

# ================= distances + session management =================
tmp2 = tempfile.mkdtemp()
A = make_session(tmp2, "20260930_100000", n=4); B = make_session(tmp2, "20260930_110000", n=3)
q = ac.Project(os.path.join(tmp2, "proj"), default_classes=["person", "bottle"])
q.import_gimble_session(A); q.import_gimble_session(B)
a1, a2, b1 = "gim_20260930_100000_000001", "gim_20260930_100000_000002", "gim_20260930_110000_000001"
pick = {"distance_m": 2.5, "u": 100, "v": 50}
q.set_boxes(a1, [(0, .5, .5, .2, .2), (1, .3, .3, .1, .1)], distances=[pick, None])
assert q.get_distances(a1) == [pick, None] and q.get_distances(a2) == []
q.set_boxes(a2, [(0, .5, .5, .2, .2)])
assert q.get_distances(a2) == [None]
q.set_boxes(b1, [(1, .5, .5, .2, .2)], distances=[{"distance_m": 1.0, "u": 5, "v": 6}])
# rotate carries picks: cw (u,v)->(H-1-v, u) with H=480
q.rotate_image(a1, clockwise=True)
assert q.get_distances(a1)[0]["u"] == 480 - 1 - 50 and q.get_distances(a1)[0]["v"] == 100, q.get_distances(a1)
q.rotate_image(a1, clockwise=False)
assert (q.get_distances(a1)[0]["u"], q.get_distances(a1)[0]["v"]) == (100, 50)
# summary
sm = {r["session_id"]: r for r in q.session_summary()}
assert sm["20260930_100000"]["samples"] == 4 and sm["20260930_100000"]["annotated"] == 2 and sm["20260930_100000"]["with_depth"] == 4
# exports (root + class-filtered)
r0 = q.export_dataset(distortion_count=1); assert r0["no_validation"] and r0["n_val_sources"] == 0
q.manifest["distortions"].get(b1) and None
assert q.manifest["distortions"].get(b1) is not None          # B has distorted copies while it trains
removed = q.set_session_role("20260930_110000", "val"); assert removed >= 1 and q.session_role("20260930_110000") == "val"
assert b1 not in q.manifest["distortions"] and not [f for f in os.listdir(q.dist_images_dir) if f.startswith(b1)]
try: q.set_session_role("nope", "val"); raise SystemExit("should fail")
except ValueError: pass
q.export_dataset(distortion_count=1)
r1 = q.export_dataset(distortion_count=1, classes=["bottle"])
assert r1["val_sessions"] == ["20260930_110000"] and r1["train_sessions"] == ["20260930_100000"] and not r1["no_validation"]
vt = open(os.path.join(q.root, "val.txt")).read().split(); tt = open(os.path.join(q.root, "train.txt")).read().split()
assert len(vt) == 1 and "20260930_110000" in vt[0] and all("20260930_100000" in x for x in tt) and len(tt) >= 4 and not any("_dist" in x for x in vt)
q.set_session_role("20260930_110000", "val")
assert q.session_summary()[1]["role"] == "val"
rows = list(csv.DictReader(open(os.path.join(q.root, "distances.csv"))))
assert any(r["image"] == a1 and r["distance_m"] == "2.500" for r in rows) and any(r["image"] == a1 + "_dist0" for r in rows)
assert len(q.existing_exports()) == 2
split_before = {l.strip().split(os.sep)[-1].replace(".jpg","").split("gim_")[-1].split("_",2)[-1] for l in open(os.path.join(q.root, "val.txt"))}
# rename
for bad in ("", "a b", "a/b", "20260930_100000", "20260930_110000"):
    try: q.rename_session("20260930_100000", bad); raise SystemExit("should reject " + repr(bad))
    except ValueError: pass
res = q.rename_session("20260930_100000", "kitchen_run1")
na1 = "gim_kitchen_run1_000001"
assert res["renamed"] == 4 and na1 in q.manifest["images"] and a1 not in q.manifest["images"]
assert q.manifest["images"][na1]["session_id"] == "kitchen_run1" and q.is_annotated(na1)
assert os.path.exists(q.image_path(na1)) and os.path.exists(q.label_path(na1)) and q.has_depth(na1) and os.path.exists(q.lidar_path(na1))
assert q.get_distances(na1)[0]["distance_m"] == 2.5 and len(q.get_boxes(na1)) == 2
assert q.manifest["distortions"][na1] == [na1 + "_dist0"] and a1 not in q.manifest["distortions"]
assert os.path.exists(os.path.join(q.dist_images_dir, na1 + "_dist0.jpg")) and os.path.exists(os.path.join(q.sessions_dir, "kitchen_run1", "meta.json"))
assert "kitchen_run1" in q.manifest["sessions_imported"] and "20260930_100000" not in q.manifest["sessions_imported"]
assert not any(a1 in f for dp, _, fs in os.walk(q.root) if os.sep + "exports" not in dp for f in fs), "old id left on disk"
done = q.refresh_exports()
assert done == ["(all classes)", "bottle"], done
tv = open(os.path.join(q.root, "train.txt")).read() + open(os.path.join(q.root, "val.txt")).read()
assert "kitchen_run1" in tv and "20260930_100000" not in tv
assert not any("20260930_100000" in f for f in os.listdir(os.path.join(q.root, "exports", "bottle", "images")))
assert any("kitchen_run1" in f for f in os.listdir(os.path.join(q.root, "exports", "bottle", "images")))
# the split is rename-stable
split_after = {l.strip().split(os.sep)[-1].replace(".jpg","").split("gim_")[-1].split("_",2)[-1] for l in open(os.path.join(q.root, "val.txt"))}
assert split_before == split_after, (split_before, split_after)
assert q.session_role("20260930_110000") == "val"
# delete
n = q.delete_session("kitchen_run1")
assert n == 4 and [r["session_id"] for r in q.session_summary()] == ["20260930_110000"]
assert not os.path.exists(os.path.join(q.sessions_dir, "kitchen_run1")) and "kitchen_run1" not in q.manifest["sessions_imported"]
q.refresh_exports()
tv = open(os.path.join(q.root, "train.txt")).read() + open(os.path.join(q.root, "val.txt")).read()
assert "kitchen" not in tv and "20260930_110000" in tv
assert not [f for f in os.listdir(os.path.join(q.root, "exports", "bottle", "images")) if "kitchen" in f]
rows = list(csv.DictReader(open(os.path.join(q.root, "distances.csv"))))
assert rows and all("kitchen" not in r["image"] for r in rows)
q.rename_session("20260930_110000", "lab_B"); assert q.session_role("lab_B") == "val" and "20260930_110000" not in q.manifest["session_roles"]
q.refresh_exports(); assert "lab_B" in open(os.path.join(q.root, "val.txt")).read()
q.delete_session("lab_B"); assert q.manifest["session_roles"] == {}
print("colors", ac.depth_colors([0.5, 3.0, 6.0]).tolist())
print("CORE TESTS PASSED")
