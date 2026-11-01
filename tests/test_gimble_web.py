import os, sys, tempfile, json, urllib.request, urllib.error
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac, web_mode as wm

tmp = tempfile.mkdtemp()
P = ac.Project(os.path.join(tmp, "p"), default_classes=["person", "bag"])
ids = [P.add_uploaded_image(Image.new("RGB", (400, 300), (i * 40, 90, 160))) for i in range(3)]
P.manifest["images"][ids[0]]["session_id"] = "s_a"; P.manifest["images"][ids[1]]["session_id"] = "s_a"; P.save_manifest()
calls = {"chg": 0}
def predict(i): return [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.2, 0.2, 0.1, 0.1)]
w = wm.WebShare(P, predict=predict,
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

# an edit is an unconfirmed draft ("?"): validated, clipped, tiny ones dropped, NOT annotated, credited to nobody yet
c, b, _ = req(f"/api/boxes/{ids[0]}", {"boxes": [[0, 0.5, 0.5, 0.4, 0.4], [1, 0.99, 0.5, 0.2, 0.2], [0, 0.3, 0.3, 0.0001, 0.0001]]}, ck)
d = json.loads(b); assert c == 200 and len(d["boxes"]) == 2 and d["annotated"] is False, d
assert d["boxes"][1][1] < 0.99 and abs(d["boxes"][1][3] - 0.11) < 1e-6, d                     # clipped at the image edge
assert not P.is_annotated(ids[0]) and P.is_ai_pending(ids[0]) and P.editor_id is None and calls["chg"] == 1
g = json.loads(req("/api/boxes/" + ids[0], cookie=ck)[1]); assert len(g["boxes"]) == 2 and g["annotated"] is False and g["ai"] is True
assert [i for i, _ in json.loads(req("/api/list?f=todo", cookie=ck)[1])["items"]] == sorted(ids)      # a draft is still "to do"
assert [st for i, st in json.loads(req("/api/list?f=all", cookie=ck)[1])["items"] if i == ids[0]] == [1]
# Confirm makes it real: annotated, credited to the web user, leaves the to-do list
c, b, _ = req(f"/api/confirm/{ids[0]}", {"boxes": d["boxes"]}, ck); d2 = json.loads(b)
assert c == 200 and d2["annotated"] is True and P.is_annotated(ids[0]) and not P.is_ai_pending(ids[0])
assert P.manifest["images"][ids[0]].get("edited_by") == "web:Sam" and P.editor_id is None
assert [i for i, _ in json.loads(req("/api/list?f=todo", cookie=ck)[1])["items"]] == sorted(ids[1:])
assert [i for i, _ in json.loads(req("/api/list?f=done", cookie=ck)[1])["items"]] == [ids[0]]
assert len(json.loads(req("/api/list?f=all", cookie=ck)[1])["items"]) == 3
# editing a confirmed image keeps it confirmed
c, b, _ = req(f"/api/boxes/{ids[0]}", {"boxes": [[0, 0.5, 0.5, 0.2, 0.2]]}, ck); assert json.loads(b)["annotated"] is True and P.is_annotated(ids[0])
# sessions: per-session counts, and the list can be limited to one
ss = {r["id"]: r for r in json.loads(req("/api/sessions", cookie=ck)[1])["sessions"]}
assert ss[""] == {"id": "", "total": 3, "annotated": 1, "draft": 0} and ss["s_a"] == {"id": "s_a", "total": 2, "annotated": 1, "draft": 0}
assert ss["~"]["total"] == 1 and list(ss)[0] == "" and list(ss)[-1] == "~"
assert [i for i, _ in json.loads(req("/api/list?f=all&s=s_a", cookie=ck)[1])["items"]] == sorted(ids[:2])
assert [i for i, _ in json.loads(req("/api/list?f=todo&s=s_a", cookie=ck)[1])["items"]] == [ids[1]]
assert [i for i, _ in json.loads(req("/api/list?f=done&s=s_a", cookie=ck)[1])["items"]] == [ids[0]]
assert [i for i, _ in json.loads(req("/api/list?f=all&s=~", cookie=ck)[1])["items"]] == [ids[2]]
assert json.loads(req("/api/list?f=bogus&s=nope", cookie=ck)[1])["items"] == []
for bad in ({"boxes": [[5, .5, .5, .1, .1]]}, {"boxes": "x"}, {"boxes": [[0, .5, .5]]}, {"boxes": [[0, "a", .5, .1, .1]]},
            {"boxes": [[0, 1e9, .5, .1, .1]]}, {"boxes": [[0, .5, .5, .1, .1]] * 501}, {}):
    assert req(f"/api/boxes/{ids[1]}", bad, ck)[0] == 400, bad
assert not P.is_annotated(ids[1]) and not P.is_ai_pending(ids[1])
assert req("/api/confirm/" + ids[1], {"boxes": "x"}, ck)[0] == 400 and req("/api/confirm/nope", {"boxes": []}, ck)[0] == 404

# ids from the outside world
for bad in ("..%2F..%2Fsecret", "nope", "a%00b"):
    assert req(f"/api/img/{bad}", cookie=ck)[0] == 404 and req(f"/api/boxes/{bad}", cookie=ck)[0] == 404, bad
assert req("/api/boxes/nope", {"boxes": []}, ck)[0] == 404

# AI: makes an unreviewed draft, never touches a confirmed image
c, b, _ = req(f"/api/ai/{ids[1]}", {}, ck); d = json.loads(b)
assert c == 200 and len(d["boxes"]) == 2 and d["ai"] and P.is_ai_pending(ids[1]) and not P.is_annotated(ids[1])
assert all(x[5] == 1 for x in d["boxes"])                                                       # AI boxes come back dashed
g = json.loads(req("/api/boxes/" + ids[1], cookie=ck)[1]); assert [x[5] for x in g["boxes"]] == [1, 1]
# the user adds a solid box of their own and fixes one AI box; the dashes stay where they were
mine = [g["boxes"][0], g["boxes"][1], [1, 0.2, 0.2, 0.1, 0.1, 0]]
g = json.loads(req(f"/api/boxes/{ids[1]}", {"boxes": mine}, ck)[1]); assert [x[5] for x in g["boxes"]] == [1, 1, 0] and not g["annotated"]
# first Confirm: the AI boxes turn solid and the image is NOT finished; it stays a draft
g = json.loads(req(f"/api/confirm/{ids[1]}", {"boxes": g["boxes"]}, ck)[1])
assert g["annotated"] is False and [x[5] for x in g["boxes"]] == [0, 0, 0] and not P.is_annotated(ids[1]) and P.is_ai_pending(ids[1])
assert [x[5] for x in json.loads(req("/api/boxes/" + ids[1], cookie=ck)[1])["boxes"]] == [0, 0, 0]       # and it is remembered
# second Confirm: nothing is dashed any more, so the whole image is confirmed
g = json.loads(req(f"/api/confirm/{ids[1]}", {"boxes": g["boxes"]}, ck)[1])
assert g["annotated"] is True and P.is_annotated(ids[1]) and not P.is_ai_pending(ids[1]) and "web_dash" not in P.manifest["images"][ids[1]]
P.set_boxes(ids[1], [])                                                                          # (put it back for the checks below)
c, b, _ = req(f"/api/ai/{ids[0]}", {}, ck); assert json.loads(b)["ai"] is False and len(json.loads(b)["boxes"]) == 1
w.predict = None
c, b, _ = req(f"/api/ai/{ids[2]}", {}, ck); assert c == 409 and b"AI model" in b
w.predict = lambda i: (_ for _ in ()).throw(RuntimeError("Load an AI model on the computer first"))
assert req(f"/api/ai/{ids[2]}", {}, ck)[0] == 409

# publishing is not possible from the tablet
c, b, _ = req("/api/publish", {}, ck); assert c == 403 and b"computer" in b

# stop: the address no longer answers
w.stop()
try: urllib.request.urlopen(base + "/", timeout=2); assert False
except (urllib.error.URLError, ConnectionError, OSError): pass
assert not w.tokens
# a taken port moves on to the next one
a = wm.WebShare(P, port=8795); a.start(); b2 = wm.WebShare(P, port=8795); b2.start(); assert b2.port != a.port; a.stop(); b2.stop()
print("WEB OK")
