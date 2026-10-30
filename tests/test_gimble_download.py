import os, sys, tempfile, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import model_download as md, model_library as ml

REPO = md.GITHUB_REPO
BASE = f"https://github.com/{REPO}/releases/download/models/"
def info(size, created, trainer, kind="assist"):
    return {"format": "gimble-model-v1", "kind": kind, "size": size, "created": created, "trainer": trainer,
            "classes": ["person"], "scores": None, "based_on": None}
files = {  # name -> (bytes, info)
    "Nano_old": (b"a" * 100, info("n", 1000, "A")), "Nano_new": (b"b" * 120, info("n", 2000, "B")),
    "Medium_x": (b"c" * 300, info("m", 1500, "C")),
}
release = {"assets": []}
store = {}
for stem, (data, inf) in files.items():
    release["assets"].append({"name": stem + ".pt", "size": len(data), "browser_download_url": BASE + stem + ".pt"})
    release["assets"].append({"name": stem + ".json", "size": 10, "browser_download_url": BASE + stem + ".json"})
    store[BASE + stem + ".pt"] = data; store[BASE + stem + ".json"] = inf
release["assets"].append({"name": "orphan.pt", "size": 5, "browser_download_url": BASE + "orphan.pt"})   # no .json: ignored
def fake_fetch(url):
    if url == md.API_URL: return release
    return store[url]
def fake_download(url, dest, progress=None):
    data = store[url]; open(dest, "wb").write(data)
    if progress: progress(len(data))
    return len(data)

# address check
for bad in ("http://evil.example/x.pt", "https://github.com/someone/else/releases/download/a/b.pt", None):
    try: md.check_url(bad); assert False
    except ValueError: pass
md.check_url(BASE + "x.pt")

es = md.list_remote(fake_fetch)
assert [e["stem"] for e in es] == ["Nano_new", "Medium_x", "Nano_old"], [e["stem"] for e in es]
assert [e["stem"] for e in md.pick(es, "newest")] == ["Nano_new", "Medium_x"]
assert len(md.pick(es, "all")) == 3
assert "Nano" in md.describe(es[0]) and "MB" in md.describe(es[0])

tmp = tempfile.mkdtemp(); models = os.path.join(tmp, "Models")
prog = []
r = md.install(models, md.pick(es, "newest"), lambda i, n, f, b: prog.append((i, n, f)), fake_download)
assert sorted(r["installed"]) == ["Medium_x.pt", "Nano_new.pt"] and not r["errors"] and not r["skipped"], r
assert prog and prog[0][1] == 2
assert os.path.isfile(os.path.join(models, "Assist", "Nano_new.pt")) and os.path.isfile(os.path.join(models, "Assist", "Nano_new.json"))
vs = ml.list_versions(models, "assist")                                   # the normal library sees them, newest first
assert [v["file"] for v in vs] == ["Nano_new.pt", "Medium_x.pt"] and vs[0]["size"] == "n" and vs[0]["trainer"] == "B", vs
assert not [f for f in os.listdir(os.path.join(models, "Assist")) if f.endswith(".part")]
assert md.already_have(models, es[0])

# again: nothing re-downloaded; then "all" adds just the missing one
r = md.install(models, md.pick(es, "all"), None, fake_download)
assert r["installed"] == ["Nano_old.pt"] and len(r["skipped"]) == 2, r

# an incomplete download is refused and leaves nothing behind
def short_download(url, dest, progress=None): open(dest, "wb").write(b"x")
tmp2 = tempfile.mkdtemp(); m2 = os.path.join(tmp2, "Models")
r = md.install(m2, [es[0]], None, short_download)
assert r["errors"] and "incomplete" in r["errors"][0][1] and not r["installed"], r
assert os.listdir(os.path.join(m2, "Assist")) == []

# a different file already using the name is kept; the new one gets _2
open(os.path.join(m2, "Assist", "Nano_new.pt"), "wb").write(b"mine")
r = md.install(m2, [es[0]], None, fake_download)
assert r["installed"] == ["Nano_new_2.pt"] and open(os.path.join(m2, "Assist", "Nano_new.pt"), "rb").read() == b"mine", r

# GitHub unreachable
def down(url): raise OSError("offline")
try: md.list_remote(down); assert False
except OSError: pass
print("DOWNLOAD CORE OK")
