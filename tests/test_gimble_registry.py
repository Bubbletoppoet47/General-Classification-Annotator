"""model_registry: per-size slots, old single folders keep working, discovery."""
import os, sys, shutil, tempfile, time
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"); sys.path.insert(0, HERE)
try:
    import torch
except ImportError:
    print("SKIPPED (needs torch - AI version only)"); sys.exit(0)
import gimble_model as gm, model_registry as mr
runs = tempfile.mkdtemp(prefix="runs_")
def mk_scratch(d, size, **kw):
    m = gm.GimbleNet(["person"], cfg=f"yolo26{size}.yaml"); m.set_epochs(2)
    os.makedirs(d + "/weights", exist_ok=True); gm.save_checkpoint(d + "/weights/best.pt", m, **kw)
def mk_fake(d):                                               # a "helper" slot - content irrelevant for discovery
    os.makedirs(d + "/weights", exist_ok=True); open(d + "/weights/best.pt", "wb").write(b"x")

# nothing yet: defaults
assert mr.discover(runs) == []
assert mr.resolve_run_dir("scratch", None, runs) == (runs + "/gimble_model_n", "n")
assert mr.resolve_run_dir("helper", None, runs) == (runs + "/helper_m", "m")
assert mr.resolve_run_dir("helper", "s", runs) == (runs + "/helper_s", "s")
# sized slots are found by folder name, helpers first, small -> large
mk_fake(runs + "/helper_m"); mk_fake(runs + "/helper_n"); mk_scratch(runs + "/gimble_model_s", "s"); mk_scratch(runs + "/gimble_model_n", "n")
labels = [e["label"] for e in mr.discover(runs)]
assert labels == ["Nano helper", "Medium helper", "Nano scratch", "Small scratch"], labels
assert mr.resolve_run_dir("helper", "m", runs)[0] == runs + "/helper_m" and mr.resolve_run_dir("scratch", "s", runs)[0] == runs + "/gimble_model_s"
# old single scratch folder (a MEDIUM model): not listed until its size is known; then it IS the medium slot
mk_scratch(runs + "/gimble_model", "m", round=3)
assert "Medium scratch" not in [e["label"] for e in mr.discover(runs)]
assert mr.resolve_run_dir("scratch", None, runs)[0] == runs + "/gimble_model"                    # no --size -> the old folder
assert mr.resolve_run_dir("scratch", "m", runs) == (runs + "/gimble_model", "m")                  # detected as medium -> its slot
assert mr.read_info(runs + "/gimble_model")["size"] == "m"                                        # ... and remembered
assert [e["label"] for e in mr.discover(runs)] == ["Nano helper", "Medium helper", "Nano scratch", "Small scratch", "Medium scratch"]
e = [e for e in mr.discover(runs) if e["label"] == "Medium scratch"][0]; assert e["dir"] == runs + "/gimble_model" and e["path"].endswith("best.pt")
# a size that only exists as the old folder is not used for a DIFFERENT size
assert mr.resolve_run_dir("scratch", "l", runs)[0] == runs + "/gimble_model_l"
# an own slot of the same size wins over the old folder and the old one is not listed twice
mk_scratch(runs + "/gimble_model_m", "m")
assert mr.resolve_run_dir("scratch", "m", runs)[0] == runs + "/gimble_model_m"
assert sum(1 for e in mr.discover(runs) if e["label"] == "Medium scratch") == 1
# write_info merges
mr.write_info(runs + "/helper_n", kind="helper", size="n"); mr.write_info(runs + "/helper_n", classes=["a"]); assert mr.read_info(runs + "/helper_n") == {"kind": "helper", "size": "n", "classes": ["a"]}
# an unreadable old helper folder (size unknown) is skipped, not fatal
mk_fake(runs + "/helper"); assert "Medium helper" in [e["label"] for e in mr.discover(runs, detect_legacy=True)]
shutil.rmtree(runs, ignore_errors=True)
print("REGISTRY OK")
