"""
web_mode.py - annotate from a tablet's browser (File > Open in browser).

The desktop program starts a small web server. A tablet on the same Wi-Fi opens the address, types the access code
and name, and annotates with an Apple Pencil: the pen draws and edits boxes, fingers pan and zoom. Everything is
stored by the normal Project on the computer; nothing is saved on the tablet. LiDAR / depth is not part of the web
page. Publishing is NOT possible from the tablet: it is done on the computer (File > Publish to shared folder).
Nothing counts as confirmed until Confirm is pressed; edits before that are saved as an unconfirmed draft ("?").

Safety: off until you start it; a new random 6-digit code every time; wrong codes lock a device out for a few minutes;
plain HTTP, so use it on your own Wi-Fi only. No tkinter here (the GUI passes in what it needs), so it is testable.
"""

import hmac
import io
import json
import os
import re
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import web_page

DEFAULT_PORT = 8765
MAX_IMAGE_SIDE = 1600
MAX_BOXES = 500
MAX_BODY = 1_000_000
LOCK_AFTER, LOCK_SECONDS = 5, 300
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,200}$")


def lan_ip():
    """The address other devices on the Wi-Fi can reach this computer at (127.0.0.1 if it can't be found)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))          # no packet is sent; it just picks the Wi-Fi/LAN adapter
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def clean_boxes(raw, n_classes):
    """Validates boxes from the page: [[cls, xc, yc, w, h], ...] -> [(cls, xc, yc, w, h)] clipped to the image.
    Raises ValueError on anything odd."""
    if not isinstance(raw, list) or len(raw) > MAX_BOXES:
        raise ValueError("bad boxes")
    out = []
    for b in raw:
        if not isinstance(b, (list, tuple)) or len(b) != 5:
            raise ValueError("bad box")
        c = int(b[0])
        if not 0 <= c < n_classes:
            raise ValueError("unknown class")
        xc, yc, w, h = (float(v) for v in b[1:])
        if any(v != v or abs(v) > 10 for v in (xc, yc, w, h)):
            raise ValueError("bad numbers")
        x1, x2 = max(0.0, xc - w / 2), min(1.0, xc + w / 2)
        y1, y2 = max(0.0, yc - h / 2), min(1.0, yc + h / 2)
        if x2 - x1 < 0.002 or y2 - y1 < 0.002:
            continue                                   # a tap, not a box
        out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out


class WebShare:
    """predict(img_id) -> boxes (or raises) / None when no AI model is loaded; on_change() is told after the page
    changed something (so the desktop list can refresh)."""

    def __init__(self, project, predict=None, on_change=None, port=DEFAULT_PORT, code=None):
        self.project, self.predict, self.on_change = project, predict, on_change
        self.port = port
        self.code = code or f"{secrets.randbelow(10**6):06d}"
        self.lock = threading.Lock()
        self.tokens = {}              # cookie token -> name
        self.fails = {}               # ip -> (count, locked_until)
        self.httpd = None
        self.thread = None

    # ---- lifecycle
    @property
    def running(self):
        return self.httpd is not None

    @property
    def url(self):
        return f"http://{lan_ip()}:{self.port}/"

    @property
    def login_url(self):
        return f"{self.url}?c={self.code}"

    def start(self):
        share = self

        class Handler(_Handler):
            owner = share
        last = None
        for port in range(self.port, self.port + 20):
            try:
                self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
                self.port = port
                break
            except OSError as e:
                last = e
        if not self.httpd:
            raise last
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.httpd = None
        self.tokens.clear()

    # ---- login
    def try_login(self, ip, code, name):
        n, until = self.fails.get(ip, (0, 0))
        if until > time.time():
            return None, "Too many tries. Wait a few minutes."
        name = re.sub(r"[^\w .\-]", "", str(name or "")).strip()[:30]
        if not name:
            return None, "Type your name."
        if not hmac.compare_digest(str(code or "").strip(), self.code):
            n += 1
            self.fails[ip] = (n, time.time() + LOCK_SECONDS if n >= LOCK_AFTER else 0)
            return None, "Wrong code."
        self.fails.pop(ip, None)
        tok = secrets.token_urlsafe(24)
        self.tokens[tok] = name
        return tok, None

    # ---- what the page asks for
    def state(self, name):
        pr = self.project
        ids = pr.list_images()
        done = sum(1 for i in ids if pr.is_annotated(i))
        return {"me": name, "classes": list(pr.classes), "total": len(ids), "done": done,
                "ai": self.predict is not None}

    NO_SESSION = "~"           # images that belong to no session (single uploads)

    def sessions(self):
        """[{id, total, annotated, draft}] for every session (and the loose images, id '~'), plus the overall row."""
        pr = self.project
        rows, tot = {}, {"id": "", "total": 0, "annotated": 0, "draft": 0}
        for i, info in pr.manifest["images"].items():
            sid = info.get("session_id") or self.NO_SESSION
            r = rows.setdefault(sid, {"id": sid, "total": 0, "annotated": 0, "draft": 0})
            ann = bool(info.get("annotated"))
            dr = (not ann) and bool(info.get("ai_pending"))
            for x in (r, tot):
                x["total"] += 1
                x["annotated"] += 1 if ann else 0
                x["draft"] += 1 if dr else 0
        return [tot] + [rows[k] for k in sorted(rows, key=lambda k: (k == self.NO_SESSION, k))]

    def list_ids(self, which, session=""):
        """which: 'todo' (not confirmed), 'done' (confirmed) or 'all'. session '' = everything, '~' = loose images."""
        pr = self.project
        out = []
        for i in pr.list_images():
            if session:
                sid = pr.manifest["images"][i].get("session_id") or self.NO_SESSION
                if sid != session:
                    continue
            ann = pr.is_annotated(i)
            if (which == "todo" and ann) or (which == "done" and not ann):
                continue
            out.append([i, 2 if ann else (1 if pr.is_ai_pending(i) else 0)])
        return out

    def _known(self, img_id):
        if not _ID_RE.match(img_id or "") or img_id not in self.project.manifest["images"]:
            raise KeyError(img_id)
        return img_id

    def boxes(self, img_id):
        pr = self.project
        self._known(img_id)
        return {"boxes": [list(b) for b in pr.get_boxes(img_id)], "annotated": pr.is_annotated(img_id),
                "ai": pr.is_ai_pending(img_id)}

    def save_boxes(self, img_id, raw, name):
        """An edit. A confirmed image stays confirmed; anything else is saved as an unconfirmed draft ('?')."""
        return self._store(img_id, raw, name, confirm=False)

    def confirm(self, img_id, raw, name):
        return self._store(img_id, raw, name, confirm=True)

    def _store(self, img_id, raw, name, confirm):
        pr = self.project
        self._known(img_id)
        boxes = clean_boxes(raw, len(pr.classes))
        with self.lock:
            old = pr.editor_id
            pr.editor_id = f"web:{name}"
            try:
                if confirm or pr.is_annotated(img_id):
                    pr.set_boxes(img_id, boxes)
                else:
                    pr.set_ai_predicted_boxes(img_id, boxes)
            finally:
                pr.editor_id = old
        self._changed()
        return {"boxes": [list(b) for b in boxes], "annotated": pr.is_annotated(img_id)}

    def run_ai(self, img_id):
        if self.predict is None:
            raise RuntimeError("Load an AI model on the computer first (File > Insert Assist Model).")
        pr = self.project
        self._known(img_id)
        if pr.is_annotated(img_id):
            return {"boxes": [list(b) for b in pr.get_boxes(img_id)], "ai": False}
        boxes = [tuple(b) for b in (self.predict(img_id) or [])]
        with self.lock:
            pr.set_ai_predicted_boxes(img_id, boxes)
        self._changed()
        return {"boxes": [list(b) for b in boxes], "ai": True}

    def image_bytes(self, img_id):
        from PIL import Image
        self._known(img_id)
        im = Image.open(self.project.image_path(img_id))
        im.load()
        im = im.convert("RGB")
        if max(im.size) > MAX_IMAGE_SIDE:
            im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()

    def _changed(self):
        if self.on_change:
            try:
                self.on_change()
            except Exception:  # noqa: BLE001  (the page's save must not fail because the desktop refresh did)
                pass


class _Handler(BaseHTTPRequestHandler):
    owner = None
    server_version = "GCA"

    def log_message(self, *a):        # quiet
        pass

    # ---- helpers
    def _send(self, code, body=b"", ctype="application/json", headers=()):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._send(code, {"error": msg})

    def _name(self):
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "gca":
                return self.owner.tokens.get(v)
        return None

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("too big")
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    # ---- routes
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, web_page.PAGE, "text/html; charset=utf-8")
        name = self._name()
        if not name:
            return self._err(401, "login")
        o = self.owner
        try:
            if u.path == "/api/state":
                return self._send(200, o.state(name))
            if u.path == "/api/sessions":
                return self._send(200, {"sessions": o.sessions()})
            if u.path == "/api/list":
                q = parse_qs(u.query)
                which = (q.get("f") or ["todo"])[0]
                which = which if which in ("todo", "done", "all") else "todo"
                return self._send(200, {"items": o.list_ids(which, (q.get("s") or [""])[0])})
            m = re.match(r"^/api/(img|boxes)/([^/]+)$", u.path)
            if m:
                if m.group(1) == "img":
                    return self._send(200, o.image_bytes(m.group(2)), "image/jpeg")
                return self._send(200, o.boxes(m.group(2)))
        except KeyError:
            return self._err(404, "No such image.")
        except OSError:
            return self._err(404, "Image file missing.")
        self._err(404, "Not found")

    def do_POST(self):
        u = urlparse(self.path)
        o = self.owner
        try:
            data = self._body()
        except (ValueError, json.JSONDecodeError):
            return self._err(400, "Bad request")
        if u.path == "/api/login":
            tok, err = o.try_login(self.client_address[0], data.get("code"), data.get("name"))
            if err:
                return self._err(403, err)
            return self._send(200, {"ok": True},
                              headers=[("Set-Cookie", f"gca={tok}; Path=/; HttpOnly; SameSite=Strict")])
        name = self._name()
        if not name:
            return self._err(401, "login")
        try:
            m = re.match(r"^/api/(boxes|ai|confirm)/([^/]+)$", u.path)
            if m and m.group(1) == "boxes":
                return self._send(200, o.save_boxes(m.group(2), data.get("boxes"), name))
            if m and m.group(1) == "confirm":
                return self._send(200, o.confirm(m.group(2), data.get("boxes"), name))
            if m and m.group(1) == "ai":
                return self._send(200, o.run_ai(m.group(2)))
            if u.path == "/api/publish":
                return self._err(403, "Publish on the computer (File > Publish to shared folder).")
        except KeyError:
            return self._err(404, "No such image.")
        except (ValueError, TypeError) as e:
            return self._err(400, str(e))
        except RuntimeError as e:
            return self._err(409, str(e))
        except Exception as e:  # noqa: BLE001
            return self._err(500, f"{type(e).__name__}: {e}")
        self._err(404, "Not found")
