"""
share_zip.py - publishing annotated data to a shared folder, and collecting it again.
(The file name is historical: there are no zip files any more - everybody has a plain folder.)

What travels: only ANNOTATED images (the ones that go into training, including confirmed images with no
objects), with their boxes (by class NAME), per-box distance picks and the LiDAR depth picture of the frame.
Recordings, videos, unannotated frames, distorted copies, models and every GUI setting stay local.
Whether a set is for training or validation is NOT stored here - that is the training program's decision.

Layout of the shared folder (every person only ever writes inside their own folder):

    <shared>/Ian/index.json            what is in this folder: one entry per image (boxes, distances, fingerprint)
    <shared>/Ian/images/<uid>.jpg
    <shared>/Ian/depth/<uid>.png       (frames that have LiDAR depth)

How it works
  * Everybody has a name + short id (asked once, remembered in ~/.trash_annotator.json).
  * publish() looks at YOUR folder - not at any note kept in the project - and copies only what is missing or
    different: an empty folder gets everything, an up-to-date one gets nothing, a deleted folder is rebuilt.
    index.json is written last (atomically), so a half-synced folder never shows an image without its files.
    Nothing is ever deleted from the folder by the program.
  * Every shared image has a permanent unique id (uid) and an OWNER = the person who first annotated it.
    The owner never changes, even when somebody else later fixes a box; who edited last, and when, is recorded.
  * merge_folders() reads every person's folder and keeps, per uid, the version edited LAST (log of what it
    replaced). That is how the training program collects everyone's data - nobody is asked anything.
  * To fix old annotations you import them (available_sets / select_entries -> analyze_entries ->
    apply_import), edit, and publish: the fix goes into YOUR folder (the image keeps its owner) and wins by
    being newest. Images you imported but did not change are not copied into your folder.
    Per image, importing compares the folder's version, yours, and the version you had at your last sync
    ("base"): equal -> nothing; only the folder changed -> take it; only yours changed -> keep yours;
    both changed -> the newer edit wins (resolve_latest).

No tkinter here, so everything is testable. The windows live in annotator_gui.py / trainer_gui.py.
"""

import hashlib
import json
import os
import re
import time
import uuid
import shutil

import annotator_core as ac

FORMAT = "gimble-folder-v1"
INDEX = "index.json"

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".trash_annotator.json")
NONE_STATE = "none"          # state of an image that is not annotated


# ---------------------------------------------------------------- who am I (per computer, not per project)

def load_config(path=None):
    try:
        with open(path or CONFIG_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(cfg, path=None):
    path = path or CONFIG_PATH
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)


def new_id(n=12):
    return uuid.uuid4().hex[:n]


def person_name_error(name):
    """Error text, or None if `name` can be used as a person's name (it ends up in set names)."""
    err = ac.Project.validate_session_name(name)
    if err:
        return err.replace("Names", "Your name")
    if "." in (name or ""):
        return "Your name can't contain a dot."
    return None


def identity(cfg):
    """{'id', 'name'} once the person has told the app their name, else None."""
    if cfg.get("name") and cfg.get("id"):
        return {"id": cfg["id"], "name": cfg["name"]}
    return None


def set_identity(cfg, name):
    """Remembers the name (the id is made once and kept, so changing your name later never splits your data)."""
    err = person_name_error(name)
    if err:
        raise ValueError(err)
    cfg["name"] = name.strip()
    cfg.setdefault("id", new_id())
    return identity(cfg)


# ---------------------------------------------------------------- fingerprints

def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_sha1(project, img_id):
    """SHA-1 of the image file (cached in the manifest by size + mtime), or None if the file is missing."""
    info = project.manifest["images"][img_id]
    p = project.image_path(img_id)
    try:
        st = os.stat(p)
    except OSError:
        return None
    stamp = [st.st_size, st.st_mtime_ns]
    if info.get("sha1") and info.get("sha1_stamp") == stamp:
        return info["sha1"]
    info["sha1"] = _sha1_file(p)
    info["sha1_stamp"] = stamp
    return info["sha1"]


def _r(v, n):
    return round(float(v), n)


def canon_boxes(project, img_id):
    """[[class name, xc, yc, w, h], ...] - class by NAME so other people's class lists don't matter."""
    out = []
    for cls, xc, yc, w, h in project.get_boxes(img_id):
        name = project.classes[cls] if 0 <= cls < len(project.classes) else f"class{cls}"
        out.append([name, _r(xc, 5), _r(yc, 5), _r(w, 5), _r(h, 5)])
    return out


def canon_distances(project, img_id, n_boxes):
    out = []
    for pk in project.get_distances(img_id, n_boxes):
        if not pk:
            out.append(None)
        else:
            out.append({k: (_r(v, 4) if isinstance(v, float) else v) for k, v in sorted(pk.items())})
    return out


def state_hash(sha, boxes, dists):
    blob = json.dumps([sha, boxes, dists], sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def local_state(project, img_id):
    """Fingerprint of an image's pixels + boxes + distances; NONE_STATE when it is not annotated."""
    if not project.is_annotated(img_id):
        return NONE_STATE
    sha = image_sha1(project, img_id)
    if sha is None:
        return NONE_STATE
    boxes = canon_boxes(project, img_id)
    return state_hash(sha, boxes, canon_distances(project, img_id, len(boxes)))


# ---------------------------------------------------------------- people and ownership

def people_name(project, pid):
    return project.manifest.get("people", {}).get(pid, pid)


def image_owner(project, img_id, me):
    """Owner id of an image; images that were never shared are yours."""
    return project.manifest["images"][img_id].get("owner_id") or (me["id"] if me else None)


def session_owner(project, sid, me):
    """Owner id of a set (all its images share one owner)."""
    for img_id, info in project.manifest["images"].items():
        if info.get("session_id") == sid:
            return image_owner(project, img_id, me)
    return me["id"] if me else None


def other_people(project, me):
    """{owner id: {'name', 'sessions': [ids], 'images': n}} for everybody except you."""
    out = {}
    for img_id, info in project.manifest["images"].items():
        pid = info.get("owner_id")
        if not pid or (me and pid == me["id"]):
            continue
        row = out.setdefault(pid, {"name": people_name(project, pid), "sessions": [], "images": 0})
        row["images"] += 1
        sid = info.get("session_id")
        if sid and sid not in row["sessions"]:
            row["sessions"].append(sid)
    for row in out.values():
        row["sessions"].sort()
    return out


def _safe(text, limit=40):
    return re.sub(r'[<>:"/\\|?*\s]', "_", str(text))[:limit] or "x"


def _session_for(project, entry, me, people):
    """Local set id for an incoming image: your own sets keep their name, other people's become 'Name.set'."""
    s = entry.get("set")
    if entry["owner_id"] == me["id"]:
        return s or None
    return f"{_safe(people.get(entry['owner_id'], entry['owner_id']))}.{_safe(s or 'uploads', 60)}"


def _local_id_for(entry):
    return f"shr_{_safe(entry.get('set') or 'uploads')}_{int(entry.get('sample_index') or 0):06d}_{entry['uid'][:6]}"


# ---------------------------------------------------------------- publishing

TRAINER_ME = {"id": "trainer", "name": "trainer"}      # the training program is nobody: every set becomes 'Name.set'


def _entry(project, img_id, me):
    info = project.manifest["images"][img_id]
    if not info.get("uid"):
        info["uid"] = new_id()
    if not info.get("owner_id"):
        info["owner_id"] = me["id"]
    boxes = canon_boxes(project, img_id)
    dists = canon_distances(project, img_id, len(boxes))
    sha = image_sha1(project, img_id)
    return {
        "uid": info["uid"], "owner_id": info["owner_id"],
        "set": info.get("set_name") or info.get("session_id"),
        "source": info.get("source"), "sample_index": info.get("sample_index"),
        "time_unix": info.get("time_unix"), "width": info.get("width"), "height": info.get("height"),
        "has_depth": project.has_depth(img_id), "depth_pixels": info.get("depth_pixels", 0),
        "sharpness": info.get("sharpness"), "quarter_turns_cw": info.get("quarter_turns_cw", 0),
        "size": os.path.getsize(project.image_path(img_id)),
        "sha1": sha, "state": state_hash(sha, boxes, dists), "boxes": boxes, "distances": dists,
        "edited_by": info.get("edited_by"), "edited_at": info.get("edited_at"),
    }


def shareable_ids(project):
    return [i for i in project.list_images()
            if project.is_annotated(i) and not project.is_ai_pending(i) and image_sha1(project, i)]


PACK_MARK = "packs-v1"


def migrate_old_sync(project, me):
    """Projects that used the old single-zip sharing carry a 'share_base' on each image meaning 'already in the
    zip'. Packs are new, so for YOUR images (or ones with no owner yet) that memory is wrong: forget it once, so
    the first Publish writes everything. Other people's images keep theirs (they came from packs)."""
    if project.manifest.get("share_format") == PACK_MARK:
        return
    for info in project.manifest["images"].values():
        if info.get("share_base") is not None and (not info.get("owner_id") or info.get("owner_id") == me["id"]):
            info["share_base"] = None
    project.manifest["share_format"] = PACK_MARK
    project.save_manifest()


def changed_ids(project):
    """Annotated images that are new or different from what you last published / imported."""
    out = []
    for i in shareable_ids(project):
        info = project.manifest["images"][i]
        if not info.get("uid") or info.get("share_base") != local_state(project, i):
            out.append(i)
    return out


def _read_index(folder):
    """The folder's index.json as a dict (checked), or None when there is none / it is unreadable."""
    try:
        with open(os.path.join(folder, INDEX), "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("format") == FORMAT and isinstance(d.get("images"), list) else None


def person_dir(shared_dir, me):
    """<shared>/<Name>; if a different person with the same name already uses that folder, <Name>_<id4>."""
    for cand in (os.path.join(shared_dir, _safe(me["name"])),
                 os.path.join(shared_dir, _safe(me["name"]) + "_" + me["id"][:4])):
        idx = _read_index(cand) if os.path.isdir(cand) else None
        if idx is None or idx.get("owner", {}).get("id") == me["id"]:
            return cand
    return cand


def _copy_atomic(src, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def _mine_to_share(project, img_id, me):
    """Images this person puts in their folder: their own, and any image of somebody else's they have EDITED."""
    info = project.manifest["images"][img_id]
    return (not info.get("owner_id")) or info["owner_id"] == me["id"] or info.get("edited_by") == me["id"]


def publish(project, shared_dir, me, progress_cb=None):
    """Brings YOUR folder in the shared folder up to date. Compares what is really in the folder (its index.json
    and the picture files) with your annotated images and copies only what is missing or different.
    Returns {'count', 'new', 'edits', 'path', 'checked'}; count 0 = the folder was already up to date."""
    migrate_old_sync(project, me)
    folder = person_dir(shared_dir, me)
    old = _read_index(folder)
    have = {e["uid"]: e for e in (old or {}).get("images", [])}
    cands = [i for i in shareable_ids(project) if _mine_to_share(project, i, me)]
    todo, fresh = [], []                       # (img_id, entry, was_new)
    for img_id in cands:
        e = _entry(project, img_id, me)
        cur = have.get(e["uid"])
        files_ok = (os.path.exists(os.path.join(folder, "images", f"{e['uid']}.jpg"))
                    and (not e["has_depth"] or os.path.exists(os.path.join(folder, "depth", f"{e['uid']}.png"))))
        if cur is not None and cur.get("state") == e["state"] and files_ok:
            fresh.append((img_id, e))
        else:
            todo.append((img_id, e, cur is None and e["owner_id"] == me["id"]))   # new = my own first annotation; a fix of someone else's image counts as an edit
    for img_id, e in fresh:                    # already there: remember it as synced (this is what import compares with)
        project.manifest["images"][img_id]["share_base"] = e["state"]
    if not todo:
        project.save_manifest()
        return {"count": 0, "new": 0, "edits": 0, "path": folder if old else None, "checked": len(cands)}
    people = dict(project.manifest.get("people", {}))
    people.update((old or {}).get("people", {}))
    people[me["id"]] = me["name"]
    os.makedirs(folder, exist_ok=True)
    for n, (img_id, e, _) in enumerate(todo, 1):
        _copy_atomic(project.image_path(img_id), os.path.join(folder, "images", f"{e['uid']}.jpg"))
        if e["has_depth"]:
            _copy_atomic(project.depth_path(img_id), os.path.join(folder, "depth", f"{e['uid']}.png"))
        have[e["uid"]] = e
        if progress_cb:
            progress_cb(n, len(todo))
    body = {"format": FORMAT, "owner": {"id": me["id"], "name": me["name"]}, "updated_at": time.time(),
            "people": people, "classes": list(project.classes),
            "images": sorted(have.values(), key=lambda x: (x.get("set") or "", x.get("sample_index") or 0, x["uid"]))}
    tmp = os.path.join(folder, INDEX + ".part")          # the index goes last, so nobody sees an image without its files
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(body, f)
    os.replace(tmp, os.path.join(folder, INDEX))
    for img_id, e, _ in todo:
        project.manifest["images"][img_id]["share_base"] = e["state"]
    project.manifest["people"] = people
    project.save_manifest()
    new = sum(1 for t in todo if t[2])
    return {"count": len(todo), "new": new, "edits": len(todo) - new, "path": folder, "checked": len(cands)}


# ---------------------------------------------------------------- reading the shared folder

class _DirSource:
    """A person's folder, read like the zip used to be: read('images/<uid>.jpg'), has('depth/<uid>.png')."""
    def __init__(self, folder):
        self.folder = folder

    def read(self, name):
        with open(os.path.join(self.folder, *name.split("/")), "rb") as f:
            return f.read()

    def has(self, name):
        return os.path.exists(os.path.join(self.folder, *name.split("/")))

    def close(self):
        pass


def list_folders(shared_dir):
    """([{'path', 'index'}] one per person folder with a readable index.json,
        [(path, reason)] for folders that could not be read - e.g. still syncing)."""
    good, bad = [], []
    if not shared_dir or not os.path.isdir(shared_dir):
        return good, bad
    for person in sorted(os.listdir(shared_dir)):
        d = os.path.join(shared_dir, person)
        if not os.path.isdir(d):
            continue
        if not os.path.exists(os.path.join(d, INDEX)):
            if os.path.isdir(os.path.join(d, "images")):
                bad.append((d, "no index.json yet (still syncing?)"))
            continue                               # an unrelated folder: ignore
        idx = _read_index(d)
        if idx is None:
            bad.append((d, "index.json is unreadable (still syncing?)"))
        else:
            good.append({"path": d, "index": idx})
    return good, bad


def merge_folders(folders, progress_cb=None):
    """Reads every person's index and keeps, per image uid, the version edited LAST. An entry whose picture (or
    depth) has not arrived yet is left out and counted in 'pending'.
    Returns {'entries': [entry + '_src'], 'people', 'overridden': [log rows], 'pending': n}; a log row says
    which version was kept and which different one was dropped (identical copies are not logged)."""
    best, overridden, people, pending = {}, [], {}, 0
    for n, fo in enumerate(sorted(folders, key=lambda f: f["index"].get("updated_at") or 0), 1):
        idx = fo["index"]
        people.update(idx.get("people", {}))
        who = idx.get("owner", {}).get("name", os.path.basename(fo["path"]))
        for e in idx["images"]:
            if not os.path.exists(os.path.join(fo["path"], "images", f"{e['uid']}.jpg")) or (
                    e.get("has_depth") and not os.path.exists(os.path.join(fo["path"], "depth", f"{e['uid']}.png"))):
                pending += 1
                continue
            e = dict(e)
            e["_src"], e["_who"] = fo["path"], who
            e["_ts"] = e.get("edited_at") or idx.get("updated_at") or 0
            cur = best.get(e["uid"])
            if cur is None:
                best[e["uid"]] = e
                continue
            newer, older = (e, cur) if e["_ts"] >= cur["_ts"] else (cur, e)
            best[e["uid"]] = newer
            if newer["state"] != older["state"]:
                overridden.append({"uid": e["uid"], "set": e.get("set"), "owner_id": e["owner_id"],
                                   "kept_from": newer["_who"], "kept_at": newer["_ts"],
                                   "dropped_from": older["_who"], "dropped_at": older["_ts"]})
        if progress_cb:
            progress_cb(n, len(folders))
    return {"entries": list(best.values()), "people": people, "overridden": overridden, "pending": pending}


def available_sets(merged):
    """[{'owner_id', 'name', 'set', 'images', 'with_depth'}] - what a person could import to edit."""
    groups = {}
    for e in merged["entries"]:
        key = (e["owner_id"], e.get("set") or "uploads")
        g = groups.setdefault(key, {"owner_id": e["owner_id"], "name": merged["people"].get(e["owner_id"], e["owner_id"]),
                                    "set": key[1], "images": 0, "with_depth": 0})
        g["images"] += 1
        g["with_depth"] += 1 if e.get("has_depth") else 0
    return sorted(groups.values(), key=lambda g: (g["name"].lower(), g["set"]))


def select_entries(merged, picks=None):
    """Entries of the picked (owner_id, set) pairs; None = everything."""
    if picks is None:
        return list(merged["entries"])
    picks = set(picks)
    return [e for e in merged["entries"] if (e["owner_id"], e.get("set") or "uploads") in picks]


# ---------------------------------------------------------------- importing: look first (analyze), then apply

def analyze_entries(project, entries, people, me, progress_cb=None):
    """What importing these entries would do - nothing is changed. Returns
    {'entries', 'people', 'new': [entry], 'update': [(entry, local_id)], 'conflicts': [{...}], 'sync': [...],
     'same': n, 'kept_local': n, 'duplicates': n, 'link': {uid: local_id}, 'new_classes': [...], 'bad_classes': [...]}.
    'update' = only the pack's version changed (or yours is not annotated); 'conflicts' = both changed."""
    migrate_old_sync(project, me)
    by_uid = {info["uid"]: i for i, info in project.manifest["images"].items() if info.get("uid")}
    sizes = {e.get("size") for e in entries}
    hash_all = None in sizes                      # packs without sizes: look at every picture
    by_sha = {}                                   # only pictures of a matching file size can be the same picture
    ids = project.list_images()
    for n, i in enumerate(ids, 1):
        try:
            if hash_all or os.path.getsize(project.image_path(i)) in sizes:
                sha = image_sha1(project, i)
                if sha:
                    by_sha.setdefault(sha, []).append(i)
        except OSError:
            pass
        if progress_cb and n % 500 == 0:
            progress_cb(n, len(ids))

    plan = {"entries": entries, "people": people, "new": [], "update": [], "conflicts": [], "link": {}, "sync": [],
            "same": 0, "kept_local": 0, "duplicates": 0, "new_classes": [], "bad_classes": []}
    seen_classes = []
    for e in entries:
        for b in e["boxes"]:
            if b[0] not in project.classes and b[0] not in seen_classes:
                seen_classes.append(b[0])
        uid = e["uid"]
        lid = by_uid.get(uid)
        if lid is None:
            same_bytes = by_sha.get(e["sha1"], [])
            free = [i for i in same_bytes if not project.manifest["images"][i].get("uid")]
            if free:
                lid = free[0]
                plan["link"][uid] = lid           # the same picture you already had: becomes one shared image
            elif same_bytes:
                plan["duplicates"] += 1           # already here under another person's id: skip, never double up
                continue
        if lid is None:
            plan["new"].append(e)
            continue
        info = project.manifest["images"][lid]
        local = local_state(project, lid)
        inc = e["state"]
        base = info.get("share_base") if info.get("uid") == uid else None
        if local == inc:
            plan["same"] += 1
            plan["sync"].append((lid, e))
        elif local == NONE_STATE and base is None:
            plan["update"].append((e, lid))
        elif base is not None and local == base:
            plan["update"].append((e, lid))
        elif base is not None and inc == base:
            plan["kept_local"] += 1
        else:
            plan["conflicts"].append({
                "uid": uid, "local_id": lid, "entry": e,
                "mine": {"annotated": local != NONE_STATE, "boxes": canon_boxes(project, lid) if local != NONE_STATE else [],
                         "edited_by": people_name(project, info.get("edited_by")) if info.get("edited_by") else None,
                         "edited_at": info.get("edited_at")},
                "theirs": {"annotated": True, "boxes": e["boxes"],
                           "edited_by": people.get(e.get("edited_by") or e["owner_id"], e.get("edited_by") or e["owner_id"]),
                           "edited_at": e.get("edited_at")},
            })
    project.save_manifest()                       # keeps the checksum cache
    for name in seen_classes:
        (plan["bad_classes"] if ac.Project.validate_class_name(name, project.classes) else plan["new_classes"]).append(name)
    return plan


def resolve_latest(plan):
    """For images both you and the pack changed: whoever edited last wins. {uid: 'mine' | 'theirs'}."""
    out = {}
    for c in plan["conflicts"]:
        theirs_ts = c["entry"].get("_ts") or c["entry"].get("edited_at") or 0
        mine_ts = c["mine"]["edited_at"] or 0
        out[c["uid"]] = "theirs" if theirs_ts > mine_ts else "mine"
    return out


def _drop_variants(project, img_id):
    for v in project.manifest["distortions"].pop(img_id, []):
        for p in (os.path.join(project.dist_images_dir, f"{v}.jpg"), os.path.join(project.dist_labels_dir, f"{v}.txt")):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


class _Sources:
    def __init__(self):
        self.open = {}

    def get(self, path):
        if path not in self.open:
            self.open[path] = _DirSource(path)
        return self.open[path]

    def close(self):
        self.open = {}


def _write_entry(project, z, e, local_id, me, people):
    """Creates (local_id None) or overwrites one image from the zip entry. Returns the local id."""
    uid = e["uid"]
    data = z.read(f"images/{uid}.jpg")
    if hashlib.sha1(data).hexdigest() != e["sha1"]:
        raise ValueError(f"The shared image {uid} is damaged or half-synced: it does not match its checksum.")
    is_new = local_id is None
    if is_new:
        local_id = _local_id_for(e)
        n = 1
        while local_id in project.manifest["images"]:
            n += 1
            local_id = _local_id_for(e) + f"_{n}"
        info = {"source": e.get("source") or "shared", "source_name": e.get("set"),
                "sample_index": e.get("sample_index"), "time_unix": e.get("time_unix"),
                "annotated": False, "ai_pending": False, "width": e.get("width"), "height": e.get("height"),
                "has_depth": False, "depth_pixels": e.get("depth_pixels", 0), "sharpness": e.get("sharpness"),
                "quarter_turns_cw": e.get("quarter_turns_cw", 0), "set_name": e.get("set")}
        sid = _session_for(project, e, me, people)
        if sid:
            info["session_id"] = sid
        project.manifest["images"][local_id] = info
    info = project.manifest["images"][local_id]

    path = project.image_path(local_id)           # follows the annotated flag, so set_boxes() below moves it right
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_new or image_sha1(project, local_id) != e["sha1"]:
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    info.pop("sha1", None)
    info.pop("sha1_stamp", None)
    info["width"], info["height"] = e.get("width", info.get("width")), e.get("height", info.get("height"))
    info["quarter_turns_cw"] = e.get("quarter_turns_cw", info.get("quarter_turns_cw", 0))

    if e.get("has_depth") and z.has(f"depth/{uid}.png"):
        with open(project.depth_path(local_id), "wb") as f:
            f.write(z.read(f"depth/{uid}.png"))
        info["has_depth"] = True
        info["depth_pixels"] = e.get("depth_pixels", 0)

    idx = {n: i for i, n in enumerate(project.classes)}
    boxes = [(idx[b[0]], b[1], b[2], b[3], b[4]) for b in e["boxes"]]
    saved_editor = project.editor_id
    project.editor_id = None                      # an import is not an edit by you
    try:
        project.set_boxes(local_id, boxes, distances=e["distances"] if e["distances"] else [])
    finally:
        project.editor_id = saved_editor
    info = project.manifest["images"][local_id]
    info["uid"] = uid
    info.setdefault("owner_id", e["owner_id"])
    if not info.get("set_name"):
        info["set_name"] = e.get("set")
    info["edited_by"] = e.get("edited_by")
    info["edited_at"] = e.get("edited_at")
    if e.get("edited_by") is None:
        info.pop("edited_by", None)
    if not is_new:
        _drop_variants(project, local_id)        # pixels / boxes changed: the distorted copies are stale
    info["share_base"] = local_state(project, local_id)
    return local_id


def apply_import(project, plan, choices, me, progress_cb=None):
    """Does what analyze_entries() found. choices = {uid: 'mine' | 'theirs'} for the conflicts; None = whoever
    edited last wins (resolve_latest). Returns a report dict."""
    if plan["bad_classes"]:
        raise ValueError("The packs have class names this project can't use: " + ", ".join(plan["bad_classes"]))
    if choices is None:
        choices = resolve_latest(plan)
    people = plan["people"]
    for name in plan["new_classes"]:
        project.add_class(name)
    pm = project.manifest.setdefault("people", {})
    for pid, name in people.items():
        if pid != me["id"]:
            pm[pid] = name
    pm[me["id"]] = me["name"]

    report = {"new": 0, "updated": 0, "took_theirs": 0, "kept_mine": 0, "same": plan["same"],
              "kept_local": plan["kept_local"], "duplicates": plan["duplicates"], "linked": len(plan["link"]),
              "added_classes": list(plan["new_classes"]), "new_ids": [], "changed_ids": []}
    work = [(e, None) for e in plan["new"]] + list(plan["update"])
    for c in plan["conflicts"]:
        if choices.get(c["uid"]) == "theirs":
            work.append((c["entry"], c["local_id"]))
            report["took_theirs"] += 1
        else:
            info = project.manifest["images"][c["local_id"]]       # keep mine; it goes out with the next publish
            info["uid"] = c["uid"]
            info.setdefault("owner_id", c["entry"]["owner_id"])
            info["share_base"] = c["entry"]["state"]
            report["kept_mine"] += 1
    zips = _Sources()
    try:
        for n, (e, lid) in enumerate(work, 1):
            lid = lid or plan["link"].get(e["uid"])
            local = _write_entry(project, zips.get(e["_src"]), e, lid, me, people)
            if lid is None:
                report["new"] += 1
                report["new_ids"].append(local)
            else:
                report["updated"] += 1
                report["changed_ids"].append(local)
            if progress_cb:
                progress_cb(n, len(work))
    finally:
        zips.close()
    for lid, e in plan["sync"]:                   # identical already: remember as synced (and take the shared id)
        info = project.manifest["images"].get(lid)
        if info is not None:
            info["uid"] = e["uid"]
            info.setdefault("owner_id", e["owner_id"])
            info["share_base"] = e["state"]
    project.save_manifest()
    return report


def collect_into(project, shared_dir, progress_cb=None):
    """The training program's step: read every person's folder, keep the newest version of each image, and bring
    the workspace project up to date. Returns the import report plus 'folders', 'unreadable', 'overridden',
    'pending', 'total'."""
    folders, bad = list_folders(shared_dir)
    merged = merge_folders(folders)
    plan = analyze_entries(project, merged["entries"], merged["people"], TRAINER_ME)
    report = apply_import(project, plan, None, TRAINER_ME, progress_cb)
    report.update(folders=len(folders), unreadable=bad, overridden=merged["overridden"], pending=merged["pending"],
                  total=len(merged["entries"]))
    return report
