import os, sys, tempfile, json, zipfile, shutil, importlib.util, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import app_update as au
spec = importlib.util.spec_from_file_location("mkrel", os.path.join(HERE, "..", "packaging", "make_release.py")); mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
spec = importlib.util.spec_from_file_location("launcher", os.path.join(HERE, "..", "packaging", "launcher.py")); lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)

tmp = tempfile.mkdtemp()
# ---- a fake installed program (old code) and a fake release (new code), made with the real release script
def make_install(base, text, ver):
    app = os.path.join(base, "app"); os.makedirs(app, exist_ok=True)
    open(os.path.join(app, "annotator_gui.py"), "w").write(text); open(os.path.join(base, "runtime.txt"), "w").write("rt1")
    return app
inst = os.path.join(tmp, "prog"); app = make_install(inst, "VERSION='old'\n", "1.0")
# the real make_release over the real source works and its manifest matches its files
prog = os.path.join(tmp, "built"); os.makedirs(prog)
mk.main(prog, os.path.join(tmp, "rel"))
man = json.load(open(os.path.join(tmp, "rel", "update-manifest.json")))
assert man["version"] == au.APP_VERSION and len(man["runtime"]) == 12 and open(os.path.join(prog, "runtime.txt")).read() == man["runtime"]
assert "annotator_gui.py" in man["files"] and "assets/icon_256.png" in man["files"] and man["code_bytes"] < 3_000_000
codezip = os.path.join(tmp, "rel", man["code_zip"])
assert hashlib.sha256(open(codezip, "rb").read()).hexdigest() == man["code_sha256"]

BASE = f"https://github.com/{au.GITHUB_REPO}/releases/download/v9/"
assets = [{"name": au.MANIFEST_NAME, "url": BASE + au.MANIFEST_NAME, "size": 5}, {"name": man["code_zip"], "url": BASE + man["code_zip"], "size": man["code_bytes"]}]
res = {"ok": True, "newer": True, "assets": assets}
# ---- plan: small update when the runtime matches, full download when it does not, page when unsupported
fetch = lambda url: dict(man, runtime="rt1")
p = au.plan_update(res, "rt1", fetch); assert p["mode"] == "code" and p["url"].endswith(man["code_zip"]) and p["bytes"] == man["code_bytes"]
assert au.plan_update(res, "OTHER", fetch)["mode"] == "full"
assert au.plan_update(res, None, fetch)["mode"] == "page"                       # an old-layout copy has no runtime id
assert au.plan_update({"assets": []}, "rt1", fetch)["mode"] == "page"
def boom(url): raise OSError("offline")
assert au.plan_update(res, "rt1", boom)["mode"] == "page"
for bad in ("http://evil/x.zip", "https://github.com/other/repo/a.zip"):
    try: au.check_url(bad); assert False
    except ValueError: pass

plan = dict(p, manifest=dict(man, runtime="rt1"))
def good_dl(url, dest, progress=None):
    shutil.copy(codezip, dest)
    if progress: progress(10)
# ---- apply: swaps app, keeps the old one in _previous, marks it pending, leaves no temp behind
calls = []
v = au.apply_code_update(inst, plan, calls.append, good_dl, current="1.0.0")
assert v == au.APP_VERSION and calls == [10]
assert os.path.isfile(os.path.join(inst, "app", "annotator_gui.py")) and "VERSION='old'" not in open(os.path.join(inst, "app", "annotator_gui.py")).read()
assert open(os.path.join(inst, "app", ".pending")).read() == "0"
prevs = os.listdir(os.path.join(inst, "_previous")); assert len(prevs) == 1 and prevs[0].startswith("app_1.0.0_")
assert open(os.path.join(inst, "_previous", prevs[0], "annotator_gui.py")).read() == "VERSION='old'\n"
assert not os.path.exists(os.path.join(inst, "_update_tmp"))
au.mark_started(inst); assert not os.path.exists(os.path.join(inst, "app", ".pending"))

# ---- failures change nothing
def damaged(url, dest, progress=None): open(dest, "wb").write(b"not a zip")
inst2 = os.path.join(tmp, "prog2"); app2 = make_install(inst2, "VERSION='old'\n", "1.0")
for dl, msg in ((damaged, "damaged"), (lambda u, d, p=None: (_ for _ in ()).throw(OSError("offline")), "offline")):
    try: au.apply_code_update(inst2, plan, None, dl); assert False
    except (ValueError, OSError) as e: assert msg in str(e) or isinstance(e, zipfile.BadZipFile), e
    assert open(os.path.join(app2, "annotator_gui.py")).read() == "VERSION='old'\n" and not os.path.exists(os.path.join(inst2, "_update_tmp"))
# a zip that has a file that doesn't match the manifest
bad = os.path.join(tmp, "bad.zip")
with zipfile.ZipFile(codezip) as zi, zipfile.ZipFile(bad, "w") as zo:
    for n in zi.namelist(): zo.writestr(n, zi.read(n) + (b"x" if n == "hotkeys.py" else b""))
m2 = dict(plan["manifest"], code_sha256=hashlib.sha256(open(bad, "rb").read()).hexdigest())
try: au.apply_code_update(inst2, dict(plan, manifest=m2), None, lambda u, d, p=None: shutil.copy(bad, d)); assert False
except ValueError as e: assert "hotkeys.py" in str(e)
# path escape in a zip
evil = os.path.join(tmp, "evil.zip")
with zipfile.ZipFile(evil, "w") as z: z.writestr("../../escaped.txt", "x")
m3 = dict(plan["manifest"], code_sha256=hashlib.sha256(open(evil, "rb").read()).hexdigest(), files={})
try: au.apply_code_update(inst2, dict(plan, manifest=m3), None, lambda u, d, p=None: shutil.copy(evil, d)); assert False
except ValueError as e: assert "unsafe" in str(e)
assert not os.path.exists(os.path.join(tmp, "escaped.txt"))
# no app folder: it can't update itself
try: au.apply_code_update(os.path.join(tmp, "nothing"), plan, None, good_dl); assert False
except OSError as e: assert "full zip" in str(e)
# only the last 3 old copies are kept
inst3 = os.path.join(tmp, "prog3"); make_install(inst3, "x", "1")
import time
for i in range(5):
    au.apply_code_update(inst3, plan, None, good_dl, current=f"1.0.{i}"); time.sleep(1.1)
assert len(os.listdir(os.path.join(inst3, "_previous"))) == 3

# ---- the launcher: a just-updated copy gets one try; one that crashed is rolled back
inst4 = os.path.join(tmp, "prog4"); make_install(inst4, "VERSION='old'\n", "1.0")
au.apply_code_update(inst4, plan, None, good_dl, current="1.0.0")
assert lc.check_pending(inst4) is False and open(os.path.join(inst4, "app", ".pending")).read() == "1"      # first start: allowed
assert lc.check_pending(inst4) is True                                                                     # it never cleared the marker: roll back
assert open(os.path.join(inst4, "app", "annotator_gui.py")).read() == "VERSION='old'\n"
assert os.path.isdir(os.path.join(inst4, "_previous", "app_failed")) and not os.path.exists(os.path.join(inst4, "app", ".pending"))
assert lc.check_pending(inst4) is False                                                                    # nothing pending now
open(os.path.join(inst4, "app", ".pending"), "w").write("1")
assert lc.check_pending(inst4) is False and open(os.path.join(inst4, "app", "annotator_gui.py")).read() == "VERSION='old'\n"   # nothing older to go back to: leave it alone
assert lc.restore_previous(os.path.join(tmp, "nothing")) is False
print("SELF UPDATE OK")
