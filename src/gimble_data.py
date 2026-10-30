"""
gimble_data.py - turns the annotator's export (+ the old people-only frames) into training batches.

Sources
-------
1. The Gimble export: data.yaml -> train.txt / val.txt (image paths), label .txt files next to them
   (YOLO format), and distances.csv (one row per box, distance_m empty when no LiDAR distance was
   picked for that box).
2. old_people_data/ (images/ + labels/ with person boxes only). Only used when "person" is one of
   the export's classes. Every class except person is switched OFF for these frames (class_mask), so
   the unlabeled trash/vehicles in them are never taught as background. They are TRAIN only, and
   have no distances.

Each sample gives: image (letterboxed to imgsz x imgsz), boxes (class, normalised xywh), a
distance per box (NaN when unknown) and class_mask (which classes are labeled in this frame).

Augmentation (training only; validation images are never touched):
  hue/saturation/brightness jitter, horizontal flip, small scale/translate. A zoom by factor s makes
  everything look s times closer, so a picked distance becomes d/s - the label stays physically
  true. Frames with distances get a gentler zoom range and no vertical shift (where an object
  sits vertically is itself a distance cue).
"""

import csv
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def names_list(names):
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names or [])


def label_path_for(img_path):
    p = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").replace("/images/", "/labels/")
    return os.path.splitext(p)[0] + ".txt"


def read_labels(path, nc):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                p = ln.split()
                if len(p) >= 5 and p[0].isdigit() and int(p[0]) < nc:
                    rows.append([int(p[0])] + [float(x) for x in p[1:5]])
    except OSError:
        pass
    return rows


def read_list(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def read_distances(csv_path):
    """{image_id: [(xc,yc,w,h,distance_or_nan), ...]} in the order the boxes were exported."""
    out = {}
    if not csv_path or not os.path.exists(csv_path):
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r.get("distance_m", "")
            try:
                dv = float(d) if d != "" else float("nan")
                box = (float(r["xc"]), float(r["yc"]), float(r["w"]), float(r["h"]), dv)
            except (ValueError, KeyError):
                continue
            out.setdefault(r["image"], []).append(box)
    return out


def attach_distances(rows, dist_rows):
    """Distance per label row. Matches by box coordinates (robust), not just order."""
    res = [float("nan")] * len(rows)
    if not dist_rows:
        return res
    used = set()
    for i, (c, xc, yc, w, h) in enumerate(rows):
        for j, (dx, dy, dw, dh, dv) in enumerate(dist_rows):
            if j in used:
                continue
            if abs(dx - xc) < 2e-5 and abs(dy - yc) < 2e-5 and abs(dw - w) < 2e-5 and abs(dh - h) < 2e-5:
                res[i] = dv
                used.add(j)
                break
    return res


class Source:
    """Everything the loaders need, read from one export. Cheap to build, no images loaded."""

    def __init__(self, data_yaml, people_dir=None, use_people=True, classes=None, no_distance=None):
        """classes: names to train (default: every class in the export). Boxes of other classes are
        dropped; the export's order is kept. Frames left without boxes stay as background.
        no_distance: class names whose distances are IGNORED (the class and box are still learned)."""
        self.data_yaml = data_yaml
        self.root = os.path.dirname(os.path.abspath(data_yaml))
        y = load_yaml(data_yaml)
        self.export_names = names_list(y.get("names"))
        if classes:
            unknown = [c for c in classes if c not in self.export_names]
            if unknown:
                raise ValueError(f"Not in this export: {unknown}. The export has: {self.export_names}")
            self.names = [n for n in self.export_names if n in set(classes)]
        else:
            self.names = list(self.export_names)
        self.nc = len(self.names)
        self.no_distance = set(no_distance or ())
        bad = [n for n in self.no_distance if n not in self.names]
        if bad:
            raise ValueError(f"--no-distance names classes that are not being trained: {bad}. Training: {self.names}")
        self._remap = {self.export_names.index(n): i for i, n in enumerate(self.names)}
        self.train_files = read_list(self._p(y.get("train")))
        self.val_files = read_list(self._p(y.get("val")))
        self.dists = read_distances(os.path.join(self.root, "distances.csv"))
        self.people = []          # [(image path, labels)]
        self.people_dir = None
        if use_people and "person" in self.names and people_dir and os.path.isdir(os.path.join(people_dir, "images")):
            self.people_dir = people_dir
            pid = self.names.index("person")
            for fn in sorted(os.listdir(os.path.join(people_dir, "images"))):
                if not fn.lower().endswith(IMG_EXT):
                    continue
                ip = os.path.join(people_dir, "images", fn)
                lp = os.path.join(people_dir, "labels", os.path.splitext(fn)[0] + ".txt")
                rows = [[pid] + r[1:] for r in read_labels(lp, 1)]      # file is person-only (class 0)
                if rows:
                    self.people.append((ip, rows))

    def _p(self, p):
        if not p:
            return None
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def samples(self, split):
        """list of dict(path, rows, dist, mask_all) for 'train' or 'val'."""
        files = self.train_files if split == "train" else self.val_files
        out = []
        for ip in files:
            rows = read_labels(label_path_for(ip), len(self.export_names))
            stem = os.path.splitext(os.path.basename(ip))[0]
            dist = attach_distances(rows, self.dists.get(stem))
            keep = [i for i, r in enumerate(rows) if r[0] in self._remap]
            rows = [[self._remap[rows[i][0]]] + rows[i][1:] for i in keep]
            dd = [dist[i] for i in keep]
            if self.no_distance:
                dd = [float("nan") if self.names[r[0]] in self.no_distance else d for r, d in zip(rows, dd)]
            out.append(dict(path=ip, rows=rows, dist=dd, people_only=False))
        if split == "train":
            for ip, rows in self.people:
                out.append(dict(path=ip, rows=rows, dist=[float("nan")] * len(rows), people_only=True))
        return out

    def summary(self):
        tr, va = self.samples("train"), self.samples("val")

        def cnt(ss):
            c = [0] * self.nc
            nd = 0
            for s in ss:
                for r, d in zip(s["rows"], s["dist"]):
                    c[r[0]] += 1
                    nd += d == d
            return c, nd
        ctr, ndtr = cnt(tr)
        cva, ndva = cnt(va)
        return dict(n_train=len(tr), n_people_only=sum(s["people_only"] for s in tr),
                    n_val=len(va), train_counts=ctr, val_counts=cva,
                    train_dist_boxes=ndtr, val_dist_boxes=ndva)


def letterbox(img, size):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = max(1, round(h * r)), max(1, round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, left, top


class GimbleDataset(Dataset):
    def __init__(self, samples, nc, imgsz=640, train=True, zoom=(0.6, 1.5), zoom_dist=(0.8, 1.25)):
        self.s, self.nc, self.imgsz, self.train = samples, nc, imgsz, train
        self.zoom, self.zoom_dist = zoom, zoom_dist

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        img = cv2.imread(s["path"], cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((480, 640, 3), 114, np.uint8)
            rows, dist = [], []
        else:
            rows, dist = s["rows"], s["dist"]
        H0, W0 = img.shape[:2]
        img, r, left, top = letterbox(img, self.imgsz)
        n = len(rows)
        S = self.imgsz
        cls = np.array([x[0] for x in rows], np.float32).reshape(n, 1)
        xywh = np.array([x[1:5] for x in rows], np.float32).reshape(n, 4)
        # normalised-to-original -> letterboxed pixel xyxy
        px = xywh * np.array([W0, H0, W0, H0], np.float32)
        xyxy = np.concatenate([px[:, :2] - px[:, 2:] / 2, px[:, :2] + px[:, 2:] / 2], 1) * r
        xyxy[:, [0, 2]] += left
        xyxy[:, [1, 3]] += top
        d = np.array(dist, np.float32).reshape(n)

        if self.train:
            img, xyxy, cls, d = self._augment(img, xyxy, cls, d, has_dist=bool(np.isfinite(d).any()))
        n = len(xyxy)
        xywh_n = np.concatenate([(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]], 1) / S if n else np.zeros((0, 4), np.float32)
        mask = np.ones(self.nc, bool)
        if s["people_only"]:
            mask[:] = False
            mask[np.unique(np.array([x[0] for x in s["rows"]], int))] = True
        t = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))
        return dict(img=t, cls=torch.from_numpy(cls.reshape(n, 1)), bboxes=torch.from_numpy(xywh_n.astype(np.float32)),
                    dist=torch.from_numpy(d.astype(np.float32)), class_mask=torch.from_numpy(mask))

    def _augment(self, img, xyxy, cls, d, has_dist):
        S = self.imgsz
        # colour
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-0.015, 0.015) * 180) % 180
        hsv[..., 1] *= random.uniform(0.6, 1.4)
        hsv[..., 2] *= random.uniform(0.6, 1.4)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        # flip
        if random.random() < 0.5:
            img = img[:, ::-1]
            if len(xyxy):
                xyxy = xyxy.copy()
                xyxy[:, [0, 2]] = S - xyxy[:, [2, 0]]
        # zoom + shift
        lo, hi = self.zoom_dist if has_dist else self.zoom
        sc = random.uniform(lo, hi) if random.random() < 0.8 else 1.0
        tx = random.uniform(-0.15, 0.15) * S
        ty = 0.0 if has_dist else random.uniform(-0.15, 0.15) * S
        if sc != 1.0 or tx or ty:
            M = np.array([[sc, 0, (1 - sc) * S / 2 + tx], [0, sc, (1 - sc) * S / 2 + ty]], np.float32)
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (S, S), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            if len(xyxy):
                nb = xyxy * sc
                nb[:, [0, 2]] += M[0, 2]
                nb[:, [1, 3]] += M[1, 2]
                area0 = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
                nb = np.clip(nb, 0, S)
                area1 = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
                keep = (area1 > 0.3 * area0) & ((nb[:, 2] - nb[:, 0]) > 4) & ((nb[:, 3] - nb[:, 1]) > 4)
                # a box that got cut by the border no longer shows the whole object -> its size no
                # longer implies the distance, so drop its distance (class + box still train)
                cut = area1 < 0.85 * area0
                d = np.where(cut, np.nan, d / sc)
                xyxy, cls, d = nb[keep], cls[keep], d[keep]
        return np.ascontiguousarray(img), np.asarray(xyxy, np.float32).reshape(-1, 4), cls, d


def collate(items):
    img = torch.stack([it["img"] for it in items])
    bidx, cls, bb, dist = [], [], [], []
    for i, it in enumerate(items):
        n = len(it["cls"])
        bidx.append(torch.full((n,), float(i)))
        cls.append(it["cls"])
        bb.append(it["bboxes"])
        dist.append(it["dist"])
    return dict(img=img, batch_idx=torch.cat(bidx), cls=torch.cat(cls, 0).reshape(-1, 1),
                bboxes=torch.cat(bb, 0).reshape(-1, 4), dist=torch.cat(dist), class_mask=torch.stack([it["class_mask"] for it in items]))
