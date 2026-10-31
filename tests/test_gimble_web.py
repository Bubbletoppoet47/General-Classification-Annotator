import os, sys, tempfile, json, urllib.request, urllib.error
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac, web_mode as wm

tmp = tempfile.mkdtemp()
P = ac.Project(os.path.join(tmp, "p"), default_classes=["person", "bag"])
ids = [P.add_uploaded_image(Image.new("RGB", (400, 300), (i * 40, 90, 160))) for i in range(3)]
calls = {"pub": 0, "chg": 0}
def predict(i): return [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.2, 0.2, 0.1, 0.1)]
w = wm.WebShare(P, predict=predict, publish=lambda: calls.__setitem__("pub", calls["pub"] + 1),
                on_change=lambda: calls.__setitem__("chg", calls["chg"] + 1), port=8791, code="123456")
w.start(); base = f"http://127.0.0.1:{w.port}"

def req(path, body=None, cookie=None):
    r = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", **({"Cookie": cookie} if cookie else {})})
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers

assert len(wm.WebShare(P).code) == 6 and wm.WebShare(P).code.isdigit()
assert req("/")[0] == 200 and b"Annotator" in req("/")[1]                       # the page itself needs no login
for p in ("/api/state", "/api/list", f"/api/img/{ids[0]}", f"/api/boxes/{ids[0]}"):
    assert req(p)[0] == 401, p                                                   # everything else does
assert req("/api/boxes/" + ids[0], {"boxes": []})[0] == 401 and req("/api/publish", {})[0] == 401

# wrong codes, then the lock-out
for i in range(5):
    c, b, _ = req("/api/login", {"name": "Sam", "code": "000000"}); assert c == 403
c, b, _ = req("/api/login", {"name": "Sam", "code": "123456"}); assert c == 403 and b"Too many" in b      # even the right one is refused now
w.fails.clear()
assert req("/api/login", {"name": "", "code": "123456"})[0] == 403               # a name is required
c, b, h = req("/api/login", {"name": "Sam", "code": "123456"}); assert c == 200
ck = h["Set-Cookie"].split(";")[0]; assert "HttpOnly" in h["Set-Cookie"]

st = json.loads(req("/api/state", cookie=ck)[1]); assert st["classes"] == ["person", "bag"] and st["total"] == 3 and st["done"] == 0 and st["me"] == "Sam"
items = json.loads(req("/api/list?f=todo", cookie=ck)[1])["items"]; assert [i for i, _ in items] == sorted(ids)
c, b, h = req(f"/api/img/{ids[0]}", cookie=ck); assert c == 200 and b[:2] == b"\xff\xd8" and h["Content-Type"] == "image/jpeg"

# save boxes: validated, clipped, tiny ones dropped, shows up as annotated and credited to the web user
c, b, _ = req(f"/api/boxes/{ids[0]}", {"boxes": [[0, 0.5, 0.5, 0.4, 0.4], [1, 0.99, 0.5, 0.2, 0.2], [0, 0.3, 0.3, 0.0001, 0.0001]]}, ck)
d = json.loads(b); assert c == 200 and len(d["boxes"]) == 2, d
assert d["boxes"][1][1] < 0.99 and abs(d["boxes"][1][3] - 0.11) < 1e-6, d                     # clipped at the image edge
assert P.is_annotated(ids[0]) and P.manifest["images"][ids[0]].get("edited_by") == "web:Sam" and P.editor_id is None
assert calls["chg"] == 1
assert json.loads(req("/api/boxes/" + ids[0], cookie=ck)[1])["annotated"] is True
assert [i for i, _ in json.loads(req("/api/list?f=todo", cookie=ck)[1])["items"]] == sorted(ids[1:])
assert len(json.loads(req("/api/list?f=all", cookie=ck)[1])["items"]) == 3
for bad in ({"boxes": [[5, .5, .5, .1, .1]]}, {"boxes": "x"}, {"boxes": [[0, .5, .5]]}, {"boxes": [[0, "a", .5, .1, .1]]},
            {"boxes": [[0, 1e9, .5, .1, .1]]}, {"boxes": [[0, .5, .5, .1, .1]] * 501}, {}):
    assert req(f"/api/boxes/{ids[1]}", bad, ck)[0] == 400, bad
assert not P.is_annotated(ids[1])

# ids from the outside world
for bad in ("..%2F..%2Fsecret", "nope", "a%00b"):
    assert req(f"/api/img/{bad}", cookie=ck)[0] == 404 and req(f"/api/boxes/{bad}", cookie=ck)[0] == 404, bad
assert req("/api/boxes/nope", {"boxes": []}, ck)[0] == 404

# AI: makes an unreviewed draft, never touches a confirmed image
c, b, _ = req(f"/api/ai/{ids[1]}", {}, ck); d = json.loads(b)
assert c == 200 and len(d["boxes"]) == 2 and d["ai"] and P.is_ai_pending(ids[1]) and not P.is_annotated(ids[1])
c, b, _ = req(f"/api/ai/{ids[0]}", {}, ck); assert json.loads(b)["ai"] is False and len(json.loads(b)["boxes"]) == 2
w.predict = None
c, b, _ = req(f"/api/ai/{ids[2]}", {}, ck); assert c == 409 and b"AI model" in b
w.predict = lambda i: (_ for _ in ()).throw(RuntimeError("Load an AI model on the computer first"))
assert req(f"/api/ai/{ids[2]}", {}, ck)[0] == 409

# publish runs on the computer
assert req("/api/publish", {}, ck)[0] == 200 and calls["pub"] == 1

# stop: the address no longer answers
w.stop()
try: urllib.request.urlopen(base + "/", timeout=2); assert False
except (urllib.error.URLError, ConnectionError, OSError): pass
assert not w.tokens
# a taken port moves on to the next one
a = wm.WebShare(P, port=8795); a.start(); b2 = wm.WebShare(P, port=8795); b2.start(); assert b2.port != a.port; a.stop(); b2.stop()
print("WEB OK")
