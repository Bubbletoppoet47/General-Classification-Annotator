"""
annotator_core.py

All the non-visual logic for the trash-detection annotation tool: box
coordinate math, YOLO label file I/O, the on-disk project structure,
Gimble session import (camera image + LiDAR depth per sample),
photometric distortion generation, and dataset export. Deliberately kept free of any GUI code so it can be tested
headlessly (no display needed) and so the GUI layer (annotator_gui.py)
stays a thin wrapper around a well-tested Project object.

Project folder layout:
    <root>/
        classes.txt                    - one class name per line, index = line number
        manifest.json                  - tracks images, sessions imported, distortions generated
        images/annotated/<id>.jpg      - confirmed source images (what training actually uses)
        images/unannotated/<id>.jpg    - not-yet-confirmed source images (blank or an AI draft)
        labels/annotated/<id>.txt      - YOLO-format boxes for a confirmed image
        labels/unannotated/<id>.txt    - boxes for an unconfirmed image (e.g. an ai_pending
                                          draft), if any - may not exist for a blank image
        depth/<id>.png                 - LiDAR depth for a Gimble sample: 16-bit PNG, millimetres
                                          (z along the camera axis), 0 = no LiDAR return, same
                                          pixel grid as the image (kept in step with rotations).
                                          Flat - no annotated/unannotated split: nothing outside
                                          this tool looks it up by path.
        distances/<id>.json            - per-box LiDAR distance picks for an image, one entry per line
                                          of its label file (same order): null, or
                                          {"distance_m", "u", "v"} = the depth pixel you clicked
                                          (u, v in the image's CURRENT orientation)
        lidar/<id>.npz                 - the sample's raw LiDAR points (float32 x,y,z,intensity,
                                          LiDAR frame, metres); optional, copied at import time
        sessions/<session_id>/         - the Gimble session's meta.json (camera matrix, distortion,
                                          camera<-LiDAR extrinsics, settings) and index.csv
        distorted/images/<id>.jpg      - generated "cheaper camera" variants (always of a
                                          confirmed source - flat, no annotated/unannotated
                                          split, since a distorted variant only ever exists
                                          for an already-annotated image)
        distorted/labels/<id>.txt      - identical copy of the source image's boxes
                                          (photometric-only distortion never moves geometry)

labels/ mirrors images/'s annotated/unannotated split for a reason beyond
tidiness: Ultralytics finds an image's label file by substituting
"images" for "labels" in its path (e.g. .../images/annotated/x.jpg ->
.../labels/annotated/x.txt), preserving whatever comes after "images/".
A flat labels/ directory (no annotated/unannotated split) silently
breaks that lookup for every un-distorted source image - Ultralytics
just treats each one as having zero objects rather than raising an
error, so training still runs and still learns from the (still
correctly-resolving) distorted copies, quietly discarding the cleanest,
least-altered version of every annotated photo the whole time. Keep
label_path()'s subdir logic identical to image_path()'s.
"""

import io
import json
import os
import random
import shutil
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Pillow >= 9.1 moved these constants to Image.Transpose.*, deprecating the
# old Image.* spelling (still present but noisy/eventually-removed) - this
# shim picks whichever one this installed Pillow actually has, so the rest
# of the module can just use ROTATE_90/ROTATE_270 without caring which
# version is on this machine.
try:
    _ROTATE_90 = Image.Transpose.ROTATE_90
    _ROTATE_270 = Image.Transpose.ROTATE_270
except AttributeError:
    _ROTATE_90 = Image.ROTATE_90
    _ROTATE_270 = Image.ROTATE_270


# ---------------------------------------------------------------------------
# Gimble sessions (what gimble_pi.py records)
# ---------------------------------------------------------------------------
#
# <session>/index.csv    idx,time_unix,image,depth,lidar,lidar_points_total,depth_pixels,sharpness
# <session>/images/N.jpg raw camera frame
# <session>/depth/N.png  uint16 millimetres, 0 = no return, same pixel grid as the image
# <session>/lidar/N.npz  points (M,4) float32 in the LiDAR frame (optional)
# <session>/meta.json    camera matrix, distortion, extrinsics, settings

def is_gimble_session(path):
    return (os.path.isdir(path) and os.path.exists(os.path.join(path, 'index.csv'))
            and os.path.isdir(os.path.join(path, 'images')))


def read_gimble_index(session_dir):
    """index.csv rows as dicts (strings), in file order."""
    import csv
    with open(os.path.join(session_dir, 'index.csv'), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


_INVALID_ID_CHARS = set('<>:"/\\|?* ')


def depth_colors(z_m, near=0.5, far=6.0):
    """(N,3) uint8 RGB for depths in metres: near = red, then yellow, green, cyan, far = blue.
    Pure numpy so the GUI needs no OpenCV."""
    t = np.clip((np.asarray(z_m, dtype=np.float32) - near) / max(far - near, 1e-6), 0.0, 1.0)
    stops_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    stops = np.array([[235, 40, 30], [245, 210, 40], [70, 220, 70], [50, 210, 230], [50, 80, 235]], dtype=np.float32)
    out = np.stack([np.interp(t, stops_t, stops[:, c]) for c in range(3)], axis=1)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Box coordinate conversion
# ---------------------------------------------------------------------------

def pixel_box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert a pixel-space rectangle (as drawn on a canvas) to YOLO's
    normalized (x_center, y_center, width, height) format, all in [0, 1].
    Accepts corners in either order and clamps to the image bounds."""
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(x1, img_w))
    x2 = max(0.0, min(x2, img_w))
    y1 = max(0.0, min(y1, img_h))
    y2 = max(0.0, min(y2, img_h))

    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


def yolo_box_to_pixel(x_center, y_center, width, height, img_w, img_h):
    """Inverse of pixel_box_to_yolo - returns (x1, y1, x2, y2) in pixels."""
    cx = x_center * img_w
    cy = y_center * img_h
    w = width * img_w
    h = height * img_h
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def rotate_yolo_box_90(box, clockwise=True):
    """Rotates a single normalized YOLO box (class, xc, yc, w, h) by 90
    degrees to match Image.transpose(ROTATE_270)/(ROTATE_90) rotating the
    image itself the same direction (see Project.rotate_image) - verified
    against PIL's actual pixel output, not just derived on paper: a
    marker in an image's top-left corner lands in the top-right corner
    after a clockwise rotation, and this formula maps a box drawn around
    that marker to the same new corner.

    Swaps width and height (a 90 degree turn swaps which axis is which),
    and remaps the center point:
      clockwise:         new_xc = 1 - yc,   new_yc = xc
      counter-clockwise:  new_xc = yc,       new_yc = 1 - xc
    Applying the same direction four times returns a box to its original
    coordinates exactly (pure arithmetic - no rounding creeps in)."""
    cls, xc, yc, w, h = box
    if clockwise:
        return (cls, 1.0 - yc, xc, h, w)
    return (cls, yc, 1.0 - xc, h, w)


# ---------------------------------------------------------------------------
# YOLO label file I/O
# ---------------------------------------------------------------------------

def save_label_file(path, boxes):
    """boxes: list of (class_id:int, x_center, y_center, width, height),
    all normalized floats. Always writes the file, even if boxes is empty -
    an empty file means 'annotated, nothing here', as distinct from no file
    at all, which the manifest treats as 'not yet annotated'."""
    with open(path, 'w', encoding='utf-8') as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def load_label_file(path):
    if not os.path.exists(path):
        return []
    boxes = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            xc, yc, w, h = (float(p) for p in parts[1:5])
            boxes.append((cls, xc, yc, w, h))
    return boxes


# ---------------------------------------------------------------------------
# Classes file
# ---------------------------------------------------------------------------

def load_classes(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def save_classes(path, classes):
    with open(path, 'w', encoding='utf-8') as f:
        for c in classes:
            f.write(c + '\n')


# ---------------------------------------------------------------------------
# Fixed class list
# ---------------------------------------------------------------------------
#
# The annotation workflow uses a fixed, ordered set of classes (rather than
# free-text classes added on the fly) so every class can be bound to a
# single number-key hotkey (1-9, then 0 for a 10th). A brand-new project
# gets this list automatically (see Project.__init__ below); an existing
# project's classes.txt is never rewritten to match it, so re-opening an
# older/different project never reinterprets its already-saved box class
# ids out from under it.
#
# IMPORTANT if you ever add a class later: append it to the END of this
# list. Every existing label file stores a class as a plain integer index
# into this list - inserting a class in the middle, or reordering, would
# silently relabel every box already drawn with the classes that shifted.
FIXED_CLASSES = [
    'person',
    'tall_vegetation',
    'medium_vegetation',
    'short_vegetation',
    'tall_object',
    'short_object',
    'vehicle',
    'micro_mobility',
    'trash',
]


# ---------------------------------------------------------------------------
# Photometric distortion ("mimic a cheaper camera")
# ---------------------------------------------------------------------------

DEFAULT_DISTORTION_COUNT = 5  # how many distorted variants get auto-generated per annotated image

DEFAULT_DISTORTION_RANGES = {
    'brightness': (0.6, 1.4),
    'contrast': (0.7, 1.3),
    'saturation': (0.6, 1.4),
    'blur_radius': (0.0, 1.5),   # 0 means "no blur applied" this draw
    'blur_chance': 0.5,
    'noise_sigma': (0.0, 12.0),
    'jpeg_quality': (35, 95),
}


def distort_image(pil_img, rng, ranges=None):
    """Applies randomized, bounded photometric jitter only - brightness,
    contrast, saturation, blur, sensor-style noise, and re-compression
    artifacts. Deliberately never crops, rotates, resizes, or warps the
    image, so a box drawn on the source image is still exactly correct on
    every distorted variant - nothing here ever needs its coordinates
    recomputed."""
    r = dict(DEFAULT_DISTORTION_RANGES)
    if ranges:
        r.update(ranges)

    img = pil_img.convert('RGB')

    b_lo, b_hi = r['brightness']
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(b_lo, b_hi))

    c_lo, c_hi = r['contrast']
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(c_lo, c_hi))

    s_lo, s_hi = r['saturation']
    img = ImageEnhance.Color(img).enhance(rng.uniform(s_lo, s_hi))

    if rng.random() < r['blur_chance']:
        radius_lo, radius_hi = r['blur_radius']
        radius = rng.uniform(radius_lo, radius_hi)
        if radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius))

    sigma_lo, sigma_hi = r['noise_sigma']
    sigma = rng.uniform(sigma_lo, sigma_hi)
    if sigma > 0:
        arr = np.asarray(img).astype(np.float32)
        noise = rng.normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    q_lo, q_hi = r['jpeg_quality']
    quality = int(rng.integers(q_lo, q_hi + 1))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img.load()
    return img


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

class Project:
    """Owns one annotation project's on-disk state. All image IDs are
    stable strings; the manifest is the single source of truth for what's
    been imported/annotated/distorted so any operation can be re-run
    without duplicating work - same incremental-cache spirit as the
    language model pipeline."""

    MANIFEST_VERSION = 1

    # Who is doing the editing (set by the annotator once it knows the person's id, see share_zip.py).
    # Every set_boxes() stamps the image with it, quietly, so the conflict window can say who last touched
    # a shared image. Never changes who the image's owner (first annotator) is.
    editor_id = None

    # Source images live in one of these two subfolders of images_dir,
    # never loose in images_dir itself - which of the two is purely a
    # function of current annotation status (see image_path). This makes
    # "which images are actually used in training" something you can see
    # directly in File Explorer, and safe to act on: delete anything sitting
    # in UNANNOTATED_SUBDIR and you know for certain it was never part of an
    # export. Distorted variants and label files are NOT split this way -
    # a distorted variant only ever exists for an annotated source, so
    # there's no ambiguity there to resolve.
    ANNOTATED_SUBDIR = 'annotated'
    UNANNOTATED_SUBDIR = 'unannotated'

    def __init__(self, root_dir, default_classes=None):
        """default_classes: what to bootstrap classes.txt with if this
        project doesn't have one yet (brand-new project). Defaults to
        FIXED_CLASSES - the normal case. Tests pass their own list (or []
        + explicit add_class calls) to exercise class-list mechanics in
        isolation without the production fixed list getting in the way.
        Has no effect at all if classes.txt already exists on disk - an
        existing project's classes are always loaded as-is, never
        replaced."""
        self.root = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.dist_images_dir = os.path.join(root_dir, 'distorted', 'images')
        self.dist_labels_dir = os.path.join(root_dir, 'distorted', 'labels')
        self.depth_dir = os.path.join(root_dir, 'depth')
        self.lidar_dir = os.path.join(root_dir, 'lidar')
        self.sessions_dir = os.path.join(root_dir, 'sessions')
        self.distances_dir = os.path.join(root_dir, 'distances')
        for d in (os.path.join(self.images_dir, self.UNANNOTATED_SUBDIR),
                  os.path.join(self.images_dir, self.ANNOTATED_SUBDIR),
                  os.path.join(self.labels_dir, self.UNANNOTATED_SUBDIR),
                  os.path.join(self.labels_dir, self.ANNOTATED_SUBDIR),
                  self.dist_images_dir, self.dist_labels_dir,
                  self.depth_dir, self.lidar_dir, self.sessions_dir, self.distances_dir):
            os.makedirs(d, exist_ok=True)

        self.classes_path = os.path.join(root_dir, 'classes.txt')
        self.manifest_path = os.path.join(root_dir, 'manifest.json')
        if os.path.exists(self.classes_path):
            self.classes = load_classes(self.classes_path)
        else:
            self.classes = list(FIXED_CLASSES if default_classes is None else default_classes)
            self.save_classes()
        self.manifest = self._load_manifest()

        # Catches up with anything deleted outside the app (e.g. you
        # cleaned up images/unannotated/ in File Explorer) so a manual
        # delete never leaves the app pointing at a file that's gone -
        # see reconcile_deleted_images. Stored so the GUI can tell the
        # user what got cleaned up right after opening a project.
        self.startup_reconciled = self.reconcile_deleted_images()

    # -- persistence -----------------------------------------------------

    def _load_manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, encoding='utf-8') as f:
                m = json.load(f)
        else:
            m = {}
        m.setdefault('version', self.MANIFEST_VERSION)
        # id -> {source, session_id, sample_index, time_unix, annotated, ai_pending, width, height,
        #        has_depth, depth_pixels, sharpness, quarter_turns_cw}
        m.setdefault('images', {})
        m.setdefault('sessions_imported', {})  # session_id -> {path, stride, samples_imported, copied_lidar}
        m.setdefault('distortions', {})      # source id -> [distorted ids]
        m.setdefault('session_roles', {})    # session_id -> 'val' (held out for validation); absent = train
        return m

    def save_manifest(self):
        with open(self.manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)

    def save_classes(self):
        save_classes(self.classes_path, self.classes)

    # -- classes -----------------------------------------------------------

    CLASS_BAD_CHARS = set(' \t\r\n,:;#\'"\\/<>|?*')

    @classmethod
    def validate_class_name(cls, name, existing=(), ignore=None):
        """Error string, or None if `name` is usable as a class name. `existing` = names already taken
        (compared ignoring case); `ignore` = one of them that doesn't count (the class being renamed)."""
        name = (name or '').strip()
        if not name:
            return "The name can't be empty."
        bad = cls.CLASS_BAD_CHARS & set(name)
        if bad:
            other = ' '.join(sorted(c for c in bad if not c.isspace()))
            return ("Class names can't contain spaces" + (f" or these characters: {other}" if other else "")
                    + "  (use _ or - instead of spaces)")
        if len(name) > 40:
            return "The name is too long (40 characters max)."
        for e in existing:
            if e != ignore and e.lower() == name.lower():
                return f"There is already a class called '{e}'."
        return None

    def add_class(self, name):
        """Appends a class at the END (ids of existing classes never change). Raises ValueError for a
        bad name; an exact duplicate is ignored."""
        name = (name or '').strip()
        if name in self.classes:
            return
        err = self.validate_class_name(name, self.classes)
        if err:
            raise ValueError(err)
        self.classes.append(name)
        self.save_classes()

    def class_box_counts(self):
        """Boxes per class across every image in the project (confirmed boxes and AI drafts)."""
        counts = [0] * len(self.classes)
        for img_id in self.manifest['images']:
            for b in self.get_boxes(img_id):
                if 0 <= b[0] < len(counts):
                    counts[b[0]] += 1
        return counts

    def train_class_stats(self):
        """Per class, over the annotated photos (not their distorted copies): how many boxes sit in sets that
        TRAIN, how many in sets held out for VALIDATION, and how many boxes carry a LiDAR distance.
        [{'name', 'train', 'val', 'dist'}] in class order - what the Train window shows next to each checkbox."""
        stats = [{'name': n, 'train': 0, 'val': 0, 'dist': 0} for n in self.classes]
        for img_id in self.list_images():
            if not self.is_annotated(img_id):
                continue
            boxes = self.get_boxes(img_id)
            picks = self.get_distances(img_id, len(boxes))
            side = 'val' if self.is_val_image(img_id) else 'train'
            for b, pk in zip(boxes, picks):
                if 0 <= b[0] < len(stats):
                    stats[b[0]][side] += 1
                    if pk and 'distance_m' in pk:
                        stats[b[0]]['dist'] += 1
        return stats

    TRAIN_SETTINGS_FILE = 'train_settings.json'

    def load_train_settings(self):
        """What the Train window had last time (models, classes by NAME, epochs, parallel/series); {} if none."""
        try:
            with open(os.path.join(self.root, self.TRAIN_SETTINGS_FILE), encoding='utf-8') as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_train_settings(self, settings):
        tmp = os.path.join(self.root, self.TRAIN_SETTINGS_FILE + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, os.path.join(self.root, self.TRAIN_SETTINGS_FILE))

    HOTKEYS_FILE = 'hotkeys.json'

    def load_hotkeys(self):
        """Keyboard shortcuts that differ from the defaults ({action: key text}); {} if none were changed."""
        try:
            with open(os.path.join(self.root, self.HOTKEYS_FILE), encoding='utf-8') as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_hotkeys(self, overrides):
        tmp = os.path.join(self.root, self.HOTKEYS_FILE + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(overrides, f, indent=2)
        os.replace(tmp, os.path.join(self.root, self.HOTKEYS_FILE))

    @staticmethod
    def _parse_yaml_names(path):
        names = {}
        try:
            with open(path, encoding='utf-8') as f:
                in_names = False
                for line in f:
                    if line.startswith('names:'):
                        in_names = True
                    elif in_names and ':' in line:
                        k, v = line.strip().split(':', 1)
                        if k.isdigit():
                            names[int(k)] = v.strip()
        except OSError:
            pass
        return [names[k] for k in sorted(names)]

    def _class_exports(self):
        """[(folder_name, [class names as written in its data.yaml])] for class-filtered exports."""
        out = []
        exp_dir = os.path.join(self.root, 'exports')
        if os.path.isdir(exp_dir):
            for name in sorted(os.listdir(exp_dir)):
                y = os.path.join(exp_dir, name, 'data.yaml')
                if os.path.exists(y):
                    out.append((name, self._parse_yaml_names(y)))
        return out

    def rename_class(self, idx, new_name):
        """Renames class `idx`. Label files store numbers, so no box changes. Class-filtered export
        folders that contain the class are renamed to match; call refresh_exports() afterwards so
        every data.yaml / distances.csv carries the new name. A model trained under the old name sees
        the new name as a new class unless you tell train_yolo.py (--rename old:new)."""
        if not (0 <= idx < len(self.classes)):
            raise ValueError("No such class.")
        old = self.classes[idx]
        new_name = (new_name or '').strip()
        err = self.validate_class_name(new_name, self.classes, ignore=old)
        if err:
            raise ValueError(err)
        if new_name == old:
            return old
        exports = [(f, n) for f, n in self._class_exports() if old in n]
        self.classes[idx] = new_name
        self.save_classes()
        for folder, names in exports:
            names2 = [new_name if n == old else n for n in names]
            src = os.path.join(self.root, 'exports', folder)
            dst = os.path.join(self.root, 'exports', "_".join(names2))
            if dst != src:
                if os.path.exists(dst):
                    shutil.rmtree(src, ignore_errors=True)       # an export with that exact list exists; it gets refreshed
                    continue
                os.replace(src, dst)
            yp = os.path.join(dst, 'data.yaml')                  # keep its class list in step so refresh_exports sees the new name
            try:
                with open(yp, encoding='utf-8') as f:
                    lines = f.read().split('\n')
                in_names = False
                for i, line in enumerate(lines):
                    if line.startswith('names:'):
                        in_names = True
                    elif in_names and ':' in line:
                        k, v = line.strip().split(':', 1)
                        if k.isdigit() and v.strip() == old:
                            lines[i] = f"  {k}: {new_name}"
                with open(yp, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
            except OSError:
                pass
        return old

    def delete_class(self, idx, move_to=None, progress=None):
        """Deletes class `idx`. Boxes of that class are removed, or moved to class `move_to` (an index in
        the CURRENT numbering). Every higher class number shifts down by one, in every label file (confirmed
        boxes, AI drafts, distorted copies) and the per-box distance picks stay attached to their boxes.
        An annotated image that loses ALL its boxes goes back to the to-do queue. Class-filtered exports that
        contained the class are removed (re-export to recreate them); call refresh_exports() afterwards.
        Only the label + distance files that actually change are copied to _class_backups/<time>/ first (the
        distorted copies are not backed up: they are re-made from the label files). progress(done, total)
        is called now and then. Returns dict(removed, moved, unannotated, backup, class)."""
        n = len(self.classes)
        if not (0 <= idx < n):
            raise ValueError("No such class.")
        if n <= 1:
            raise ValueError("Keep at least one class.")
        if move_to is not None and (move_to == idx or not (0 <= move_to < n)):
            raise ValueError("Pick a different class to move the boxes to.")
        name = self.classes[idx]

        def remap(c):
            if c == idx:
                return None if move_to is None else move_to - (1 if move_to > idx else 0)
            return c - 1 if c > idx else c

        # 1. read everything first (nothing is changed yet) and work out what changes
        removed = moved = 0
        plan = []                                    # (img_id, new boxes, new picks)
        ids = list(self.manifest['images'])
        total = len(ids) * 2
        for k, img_id in enumerate(ids):
            if progress and k % 200 == 0:
                progress(k, total)
            boxes = self.get_boxes(img_id)
            if not boxes or all(b[0] < idx for b in boxes):
                continue                              # no box of this class or a higher one: nothing to do
            picks = self.get_distances(img_id, len(boxes))
            nb, np_ = [], []
            for box, pk in zip(boxes, picks):
                c = box[0]
                if c == idx:
                    if move_to is None:
                        removed += 1
                        continue
                    moved += 1
                nb.append((remap(c),) + tuple(box[1:5]))
                np_.append(pk)
            plan.append((img_id, nb, np_))

        # 2. back up just the files that are about to change
        import time as _time
        backup = os.path.join(self.root, '_class_backups', _time.strftime('%Y%m%d_%H%M%S') + f"_delete_{name}")
        os.makedirs(backup, exist_ok=True)
        for img_id, _nb, _np in plan:
            for src, base in ((self.label_path(img_id), self.labels_dir),
                              (self.distance_path(img_id), self.distances_dir)):
                if os.path.exists(src):
                    dst = os.path.join(backup, os.path.basename(base), os.path.relpath(src, base))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(src, dst)
        shutil.copyfile(self.classes_path, os.path.join(backup, 'classes.txt'))
        if os.path.exists(self.manifest_path):
            shutil.copyfile(self.manifest_path, os.path.join(backup, 'manifest.json'))

        # 3. write the new files
        unann = 0
        to_unannotate = []
        for k, (img_id, nb, np_) in enumerate(plan):
            if progress and k % 100 == 0:
                progress(len(ids) + k * len(ids) // max(1, len(plan)), total)
            save_label_file(self.label_path(img_id), nb)
            if os.path.exists(self.distance_path(img_id)) or any(np_):
                self.set_distances(img_id, np_)
            if self.is_annotated(img_id) and not nb:
                to_unannotate.append(img_id)
            else:
                for v in self.manifest['distortions'].get(img_id, []):
                    save_label_file(os.path.join(self.dist_labels_dir, f"{v}.txt"), nb)
        for img_id in to_unannotate:
            self.unannotate_image(img_id)
            unann += 1

        for folder, names in self._class_exports():
            if name in names:
                shutil.rmtree(os.path.join(self.root, 'exports', folder), ignore_errors=True)
        del self.classes[idx]
        self.save_classes()
        self.save_manifest()
        if progress:
            progress(total, total)
        return {'removed': removed, 'moved': moved, 'unannotated': unann, 'backup': backup, 'class': name}

    # -- images ------------------------------------------------------------

    def _next_upload_id(self):
        n = 1
        existing = set(self.manifest['images'])
        while f"img_{n}" in existing:
            n += 1
        return f"img_{n}"

    def add_uploaded_image(self, pil_img, filename_hint=None):
        img_id = self._next_upload_id()
        pil_img = pil_img.convert('RGB')
        pil_img.save(os.path.join(self.images_dir, self.UNANNOTATED_SUBDIR, f"{img_id}.jpg"), quality=95)
        self.manifest['images'][img_id] = {
            'source': 'upload',
            'source_name': filename_hint,
            'has_depth': False,
            'annotated': False,
            'ai_pending': False,
            'width': pil_img.width,
            'height': pil_img.height,
        }
        self.save_manifest()
        return img_id

    # -- Gimble samples: depth / LiDAR sidecars --------------------------------

    def depth_path(self, img_id):
        return os.path.join(self.depth_dir, f"{img_id}.png")

    def lidar_path(self, img_id):
        return os.path.join(self.lidar_dir, f"{img_id}.npz")

    def has_depth(self, img_id):
        return bool(self.manifest['images'].get(img_id, {}).get('has_depth')) \
            and os.path.exists(self.depth_path(img_id))

    def distance_path(self, img_id):
        return os.path.join(self.distances_dir, f"{img_id}.json")

    def _sidecar_paths(self, img_id):
        return [self.depth_path(img_id), self.lidar_path(img_id), self.distance_path(img_id)]

    # -- per-box distance picks ---------------------------------------------------

    def get_distances(self, img_id, n_boxes=None):
        """List of per-box distance picks, aligned with get_boxes(img_id): each entry is None or
        {"distance_m", "u", "v"}. Padded with None / truncated to n_boxes when given, so a stale
        or missing sidecar can never misalign with the label file."""
        picks = []
        p = self.distance_path(img_id)
        if os.path.exists(p):
            try:
                with open(p, encoding='utf-8') as f:
                    picks = json.load(f).get('boxes', [])
            except (OSError, ValueError):
                picks = []
        if n_boxes is None:
            n_boxes = len(self.get_boxes(img_id))
        picks = list(picks)[:n_boxes]
        return picks + [None] * (n_boxes - len(picks))

    def set_distances(self, img_id, picks):
        with open(self.distance_path(img_id), 'w', encoding='utf-8') as f:
            json.dump({'boxes': list(picks)}, f)

    def load_depth_mm(self, img_id):
        """(H, W) uint16 depth in millimetres aligned with the image (0 = no LiDAR return),
        or None if this image has no depth."""
        if not self.has_depth(img_id):
            return None
        with Image.open(self.depth_path(img_id)) as im:
            return np.array(im).astype(np.uint16)

    def depth_points(self, img_id):
        """The image's LiDAR returns as three parallel arrays (u_px, v_px, z_metres), or None."""
        d = self.load_depth_mm(img_id)
        if d is None:
            return None
        v, u = np.nonzero(d)
        return u, v, d[v, u].astype(np.float32) / 1000.0

    def list_images(self):
        return sorted(self.manifest['images'])

    def image_path(self, img_id):
        subdir = self.ANNOTATED_SUBDIR if self.is_annotated(img_id) else self.UNANNOTATED_SUBDIR
        return os.path.join(self.images_dir, subdir, f"{img_id}.jpg")

    def label_path(self, img_id):
        # Mirrors image_path()'s annotated/unannotated split exactly - see
        # the module docstring for why this has to match, not just look
        # tidy: Ultralytics derives an image's label path by substituting
        # "images" for "labels" in place, so this label file's subdir
        # must always match whichever subdir the image itself is
        # currently sitting in.
        subdir = self.ANNOTATED_SUBDIR if self.is_annotated(img_id) else self.UNANNOTATED_SUBDIR
        return os.path.join(self.labels_dir, subdir, f"{img_id}.txt")

    def get_boxes(self, img_id):
        return load_label_file(self.label_path(img_id))

    def set_boxes(self, img_id, boxes, distances=None):
        """The human-confirmed path: whether these boxes came from drawing
        from scratch, editing an AI pre-fill, or just clicking Confirm on
        an AI pre-fill with no changes, calling this is what makes them
        trusted ground truth. Always clears ai_pending - once a human has
        looked at an image, it's no longer "pending AI review" regardless
        of how it got here."""
        was_annotated = self.is_annotated(img_id)
        # distances: per-box picks in the same order as boxes (see get_distances); None leaves
        # whatever is stored alone (still re-aligned to the new box count on read)
        if distances is not None:
            self.set_distances(img_id, distances)

        # label_path() resolves under the CURRENT (pre-flip) annotation
        # state, so capture it before touching anything, then always
        # (re)write the final content at the ANNOTATED-subdir path - a
        # rename first if a file already sits at the old (unannotated)
        # location (e.g. an ai_pending draft's boxes), so this never
        # needs delete permission, only rename permission, matching
        # unannotate_image()'s reasoning below.
        old_label_path = self.label_path(img_id)
        new_label_path = os.path.join(self.labels_dir, self.ANNOTATED_SUBDIR, f"{img_id}.txt")
        if old_label_path != new_label_path and os.path.exists(old_label_path):
            os.replace(old_label_path, new_label_path)
        save_label_file(new_label_path, boxes)

        if not was_annotated:
            # physically move the file into images/annotated/ - see
            # unannotate_image() below for the reverse direction
            old_path = os.path.join(self.images_dir, self.UNANNOTATED_SUBDIR, f"{img_id}.jpg")
            new_path = os.path.join(self.images_dir, self.ANNOTATED_SUBDIR, f"{img_id}.jpg")
            if os.path.exists(old_path):
                os.replace(old_path, new_path)
        info = self.manifest['images'][img_id]
        info['annotated'] = True
        info['ai_pending'] = False
        info['edited_at'] = time.time()
        if self.editor_id:
            info['edited_by'] = self.editor_id
        self.save_manifest()

    def is_annotated(self, img_id):
        return self.manifest['images'].get(img_id, {}).get('annotated', False)

    def unannotate_image(self, img_id):
        """Reverses set_boxes(): moves img_id back into images/unannotated/
        and clears its annotated (and ai_pending) flags, so it re-enters
        the normal pending-work queue exactly as if it had never been
        reviewed. Use this to back out a bad annotation - e.g. a stray
        invalid class id that slipped in - rather than hand-patching the
        label file in place; a wrong box is often a sign the whole image
        deserves a fresh look, not just a corrected number.

        Clears the label file (an unannotated image has no boxes on disk
        - see the module docstring) and best-effort deletes every
        distorted variant generated from this source, since a distorted
        variant's label is only ever a copy of its source's boxes and
        would otherwise sit around as stale, orphaned training data
        pointing at boxes that no longer exist anywhere. No-op if img_id
        is already unannotated (or unknown).

        Ordering here is deliberate and safety-critical, not incidental.
        image_path() and reconcile_deleted_images() both trust the
        manifest's annotated flag to know which folder (images/annotated/
        vs images/unannotated/) a file actually lives in - so the flag
        flip and its save are done immediately after the physical file
        move, before anything that can fail (deleting files can fail with
        a permissions error on some setups, e.g. a synced folder without
        delete rights granted; renaming within the same drive can't).
        Getting this order wrong once actually happened during
        development: a PermissionError partway through left the file
        already moved to unannotated/ but the manifest still saying
        annotated=True, and the NEXT time the project was opened,
        reconcile_deleted_images looked for the file at the (now stale)
        annotated/ path, didn't find it, and concluded the image had been
        deleted outside the app - wiping its manifest entry completely
        even though the file was sitting right there. The label file is
        also cleared by rewriting it empty (save_label_file always WRITES
        rather than removing), not by os.remove, for the same reason - a
        blocked delete must never leave an unannotated image's box data
        looking non-empty. Only the now-orphaned distortion files, whose
        loss is cosmetic disk clutter rather than a correctness problem,
        are allowed to best-effort fail."""
        if not self.is_annotated(img_id):
            return

        old_path = os.path.join(self.images_dir, self.ANNOTATED_SUBDIR, f"{img_id}.jpg")
        new_path = os.path.join(self.images_dir, self.UNANNOTATED_SUBDIR, f"{img_id}.jpg")
        if os.path.exists(old_path):
            os.replace(old_path, new_path)

        # Same rename-then-rewrite approach as set_boxes(): move whatever
        # sits at the annotated-subdir label path over to the
        # unannotated-subdir path (a rename, never a delete), then
        # overwrite it empty there - never leave it behind at the old
        # (now-wrong) subdir.
        old_label_path = os.path.join(self.labels_dir, self.ANNOTATED_SUBDIR, f"{img_id}.txt")
        new_label_path = os.path.join(self.labels_dir, self.UNANNOTATED_SUBDIR, f"{img_id}.txt")
        if os.path.exists(old_label_path):
            os.replace(old_label_path, new_label_path)
        save_label_file(new_label_path, [])
        if os.path.exists(self.distance_path(img_id)):
            self.set_distances(img_id, [])     # rewrite empty, never a delete (see above)

        self.manifest['images'][img_id]['annotated'] = False
        self.manifest['images'][img_id]['ai_pending'] = False
        self.save_manifest()

        variant_ids = self.manifest['distortions'].pop(img_id, [])
        for variant_id in variant_ids:
            for p in (os.path.join(self.dist_images_dir, f"{variant_id}.jpg"),
                      os.path.join(self.dist_labels_dir, f"{variant_id}.txt")):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
        self.save_manifest()

    def is_ai_pending(self, img_id):
        return self.manifest['images'].get(img_id, {}).get('ai_pending', False)

    def set_ai_predicted_boxes(self, img_id, boxes):
        """Writes model-predicted boxes for img_id as an UNREVIEWED draft.
        Unlike set_boxes(), this deliberately does not mark the image
        annotated and does not move its file into images/annotated/ - the
        image stays in the normal pending-work queue (now pre-filled
        instead of blank, flagged with ai_pending so the GUI can show it
        differently) until a human actually looks at it and calls
        set_boxes(), whether by editing something or just hitting
        Confirm. Nothing this writes is trusted training data on its own,
        however confident the model's predictions look."""
        save_label_file(self.label_path(img_id), boxes)
        if os.path.exists(self.distance_path(img_id)):
            self.set_distances(img_id, [])     # a fresh draft has no picks
        self.manifest['images'][img_id]['ai_pending'] = True
        self.save_manifest()

    def delete_image(self, img_id):
        """Permanently removes img_id from the project: its image file,
        its label file (if any), every distorted variant generated from
        it (images + labels), and its manifest entry. Unlike
        unannotate_image() (which keeps the image around, just resets it
        to "unreviewed" for a fresh annotation pass) this is a real,
        irreversible delete - for a frame that isn't worth keeping at
        all (blurry, a near-duplicate, nothing of interest in view), not
        a bad annotation worth redoing.

        Works regardless of whether img_id is currently annotated or
        not - tries both possible subdirectories for its image/label
        files rather than trusting is_annotated() first, since that's
        exactly the flag this call is about to make moot anyway.

        Each file removal is best-effort (same tolerance
        reconcile_deleted_images and unannotate_image's distortion
        cleanup already use) - a locked/permission-denied file doesn't
        abort the whole delete or leave the manifest half-updated; the
        manifest entry is removed regardless, since that's what actually
        keeps a frame out of review, AI-assist, and export. Silently
        no-ops if img_id isn't a known image (e.g. it was already
        deleted, or reconcile_deleted_images() already cleaned it up)."""
        if img_id not in self.manifest['images']:
            return
        self._delete_image_no_save(img_id)
        self.save_manifest()

    def _delete_image_no_save(self, img_id):
        paths = [
            os.path.join(self.images_dir, self.ANNOTATED_SUBDIR, f"{img_id}.jpg"),
            os.path.join(self.images_dir, self.UNANNOTATED_SUBDIR, f"{img_id}.jpg"),
            os.path.join(self.labels_dir, self.ANNOTATED_SUBDIR, f"{img_id}.txt"),
            os.path.join(self.labels_dir, self.UNANNOTATED_SUBDIR, f"{img_id}.txt"),
        ] + self._sidecar_paths(img_id)
        for variant_id in self.manifest['distortions'].pop(img_id, []):
            paths.append(os.path.join(self.dist_images_dir, f"{variant_id}.jpg"))
            paths.append(os.path.join(self.dist_labels_dir, f"{variant_id}.txt"))

        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

        del self.manifest['images'][img_id]

    def rotate_image(self, img_id, clockwise=True):
        """Rotates img_id's image file 90 degrees in place (a real pixel
        rotation rewritten to disk, not a display-only transform - this
        is what a frame that was captured sideways or upside-down needs
        to train correctly), and rotates its box coordinates to match
        (see rotate_yolo_box_90) so existing annotations stay pinned to
        the same real-world objects instead of ending up scattered
        wherever the old coordinates happen to land on the new
        orientation.

        clockwise=True (the default) rotates clockwise; four calls in
        the same direction return an image - and its boxes - to their
        original orientation exactly, modulo JPEG recompression on the
        image file itself (each save is a fresh encode, same as every
        other place this project writes a .jpg).

        Any distorted variants of img_id are discarded rather than
        rotated in place - left alone, they'd keep depicting the OLD,
        wrong orientation, which is exactly what this call exists to
        fix. The normal auto-distort path (ensure_all_distortions, or
        the GUI's _auto_distort after this) regenerates fresh ones from
        the now-corrected source, same as after any other box edit -
        no separate "rotate the distortions too" logic to get subtly
        wrong.

        Works regardless of annotated/ai_pending/confirmed status, and
        updates the manifest's recorded width/height to match (they
        swap on every 90 degree turn). No-op if img_id isn't a known
        image."""
        if img_id not in self.manifest['images']:
            return

        img_path = self.image_path(img_id)
        pil_img = Image.open(img_path).convert('RGB')
        method = _ROTATE_270 if clockwise else _ROTATE_90
        rotated = pil_img.transpose(method)
        rotated.save(img_path, quality=95)

        info = self.manifest['images'][img_id]
        info['width'], info['height'] = rotated.size
        # depth lives in the image's pixel grid, so it has to turn with it; record the net
        # quarter-turns so the camera matrix in the session's meta.json can still be mapped
        # onto this image later
        info['quarter_turns_cw'] = (info.get('quarter_turns_cw', 0) + (1 if clockwise else -1)) % 4
        old_h = pil_img.height
        if os.path.exists(self.distance_path(img_id)):
            # a picked depth pixel (u, v) turns with the image: clockwise, the pixel in old row r,
            # column c lands in new row c, column (old_h - 1 - r); counter-clockwise is the inverse
            picks = self.get_distances(img_id)
            old_w = pil_img.width
            turned = []
            for pk in picks:
                if pk is None:
                    turned.append(None)
                    continue
                u, v = pk['u'], pk['v']
                nu, nv = ((old_h - 1 - v, u) if clockwise else (v, old_w - 1 - u))
                turned.append({**pk, 'u': int(nu), 'v': int(nv)})
            self.set_distances(img_id, turned)
        if self.has_depth(img_id):
            depth = self.load_depth_mm(img_id)
            depth = np.rot90(depth, k=-1 if clockwise else 1)
            Image.fromarray(np.ascontiguousarray(depth).astype(np.uint16)).save(self.depth_path(img_id))

        label_path = self.label_path(img_id)
        had_label_file = os.path.exists(label_path)
        boxes = load_label_file(label_path)
        if boxes or had_label_file:
            save_label_file(label_path, [rotate_yolo_box_90(b, clockwise) for b in boxes])

        for variant_id in self.manifest['distortions'].pop(img_id, []):
            for p in (os.path.join(self.dist_images_dir, f"{variant_id}.jpg"),
                      os.path.join(self.dist_labels_dir, f"{variant_id}.txt")):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass

        self.save_manifest()

    def reconcile_deleted_images(self):
        """Call this after image files might have been deleted outside the
        app - e.g. you cleaned up images/unannotated/ in File Explorer to
        free space, knowing those weren't used in training. Any manifest
        entry whose image file no longer exists on disk gets fully
        forgotten: its label file, any distorted variants (images and
        labels) generated from it, and its manifest entries all get
        cleaned up together, so a manual delete never leaves the app
        pointing at something that's gone. Runs automatically every time a
        Project is opened (see __init__); safe and cheap to call again
        anytime (e.g. mid-session, without restarting the app). Returns
        the list of image ids that were removed this way."""
        removed = []
        for img_id in list(self.manifest['images']):
            if os.path.exists(self.image_path(img_id)):
                continue
            removed.append(img_id)
            lp = self.label_path(img_id)
            if os.path.exists(lp):
                os.remove(lp)
            for sp in self._sidecar_paths(img_id):
                if os.path.exists(sp):
                    os.remove(sp)
            for variant_id in self.manifest.get('distortions', {}).pop(img_id, []):
                for p in (os.path.join(self.dist_images_dir, f"{variant_id}.jpg"),
                          os.path.join(self.dist_labels_dir, f"{variant_id}.txt")):
                    if os.path.exists(p):
                        os.remove(p)
            del self.manifest['images'][img_id]
        if removed:
            self.save_manifest()
        return removed

    # -- Gimble session import ------------------------------------------------

    def list_sessions(self):
        """[(session_id, sample_count)] for sessions that currently have samples in the project."""
        counts = {}
        for info in self.manifest['images'].values():
            sid = info.get('session_id')
            if sid:
                counts[sid] = counts.get(sid, 0) + 1
        return sorted(counts.items())

    def import_gimble_session(self, session_dir, stride=1, copy_lidar=True, progress_cb=None):
        """Imports a session folder written by gimble_pi.py: every `stride`-th sample becomes an
        image in images/unannotated/ with id "gim_<session>_<idx:06d>", and its LiDAR depth
        (depth/<id>.png, plus lidar/<id>.npz when copy_lidar) travels with it so the depth
        stays available for review and later training - the project stays self-contained even
        if the USB drive is unplugged.

        Safe to re-run on the same session (e.g. after recording more samples into it, or with
        a smaller stride): samples already in the project are skipped, never re-copied - so a
        sample you've already annotated is never touched. The session's meta.json / index.csv
        (camera matrix, distortion, camera<-LiDAR extrinsics, capture settings) are copied to
        sessions/<session>/.

        Returns dict: new_ids, skipped_existing, missing (sample indices whose image file wasn't
        there), no_depth (indices imported without usable depth), session_id."""
        session_dir = os.path.abspath(session_dir)
        if not is_gimble_session(session_dir):
            raise ValueError(
                f"Not a Gimble session folder (needs index.csv and images/): {session_dir}")
        stride = max(1, int(stride))
        session_id = os.path.basename(session_dir.rstrip("\\/"))
        bad = _INVALID_ID_CHARS & set(session_id)
        if bad or not session_id:
            raise ValueError(f"Session folder name has characters that can't go in an image id: "
                             f"{' '.join(sorted(bad)) or '(empty)'}")

        rows = read_gimble_index(session_dir)
        sess_out = os.path.join(self.sessions_dir, session_id)
        os.makedirs(sess_out, exist_ok=True)
        for name in ('meta.json', 'index.csv'):
            src = os.path.join(session_dir, name)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(sess_out, name))

        picked = [r for n, r in enumerate(rows) if n % stride == 0]
        new_ids, skipped, missing, no_depth = [], 0, [], []
        for done, row in enumerate(picked, 1):
            idx = int(row['idx'])
            img_id = f"gim_{session_id}_{idx:06d}"
            if progress_cb:
                progress_cb(done, len(picked))
            if img_id in self.manifest['images']:
                skipped += 1
                continue
            src_img = os.path.join(session_dir, row['image'])
            if not os.path.exists(src_img):
                missing.append(idx)
                continue
            dst_img = os.path.join(self.images_dir, self.UNANNOTATED_SUBDIR, f"{img_id}.jpg")
            with Image.open(src_img) as im:
                w, h = im.size
                if im.format == 'JPEG':
                    shutil.copyfile(src_img, dst_img)      # byte-identical, no re-encode
                else:
                    im.convert('RGB').save(dst_img, quality=95)

            has_depth = False
            src_depth = os.path.join(session_dir, row.get('depth') or '')
            if row.get('depth') and os.path.exists(src_depth):
                with Image.open(src_depth) as dm:
                    ok = dm.size == (w, h)
                if ok:
                    shutil.copyfile(src_depth, self.depth_path(img_id))
                    has_depth = True
            if not has_depth:
                no_depth.append(idx)
            if copy_lidar and row.get('lidar'):
                src_lidar = os.path.join(session_dir, row['lidar'])
                if os.path.exists(src_lidar):
                    shutil.copyfile(src_lidar, self.lidar_path(img_id))

            def _num(key, cast, default):
                try:
                    return cast(row[key])
                except (KeyError, TypeError, ValueError):
                    return default

            self.manifest['images'][img_id] = {
                'source': 'gimble',
                'source_name': session_id,
                'session_id': session_id,
                'sample_index': idx,
                'time_unix': _num('time_unix', float, None),
                'annotated': False,
                'ai_pending': False,
                'width': w,
                'height': h,
                'has_depth': has_depth,
                'depth_pixels': _num('depth_pixels', int, 0) if has_depth else 0,
                'sharpness': _num('sharpness', float, None),
                'quarter_turns_cw': 0,
            }
            new_ids.append(img_id)

        prev = self.manifest['sessions_imported'].get(session_id, {'samples_imported': 0})
        self.manifest['sessions_imported'][session_id] = {
            'path': session_dir,
            'stride': stride,
            'samples_imported': prev['samples_imported'] + len(new_ids),
            'copied_lidar': bool(copy_lidar),
        }
        self.save_manifest()
        return {'new_ids': new_ids, 'skipped_existing': skipped, 'missing': missing,
                'no_depth': no_depth, 'session_id': session_id}

    # -- Video import -----------------------------------------------------------

    VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.webm', '.mpg', '.mpeg')

    @staticmethod
    def video_info(video_path):
        """(frame_count, fps, width, height) of a video file. Needs OpenCV (pip install opencv-python)."""
        try:
            import cv2
        except ImportError:
            raise RuntimeError("Importing videos needs OpenCV:  pip install opencv-python")
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Couldn't open the video: {video_path}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            return n, fps, w, h
        finally:
            cap.release()

    @staticmethod
    def default_video_session_name(video_path):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        name = ''.join('_' if (c in _INVALID_ID_CHARS) else c for c in stem).strip('_.') or 'video'
        return name[:80]

    def find_video_session(self, video_path):
        """Name of the set this video file was already imported as (even if renamed since), or None."""
        vp = os.path.normcase(os.path.abspath(video_path))
        for sid, rec in self.manifest['sessions_imported'].items():
            if rec.get('kind') == 'video' and rec.get('path') and os.path.normcase(rec['path']) == vp:
                return sid
        return None

    def import_video(self, video_path, stride=10, session_name=None, progress_cb=None):
        """Imports a video file as a set (session): every `stride`-th frame becomes a JPEG in
        images/unannotated/ with id "vid_<session>_<frame:06d>". Videos carry no LiDAR depth, so
        these frames can be boxed and labeled but not given a distance. Everything else
        (rename, delete, train/validation, working set, export) treats the set like any other.

        Safe to re-run on the same video (e.g. with a smaller stride): frames already in the
        project are skipped, so annotations are never touched. The session name defaults to the
        video's file name; a name already used by a Gimble session is refused.

        Returns dict: new_ids, skipped_existing, session_id, frames_seen, fps."""
        try:
            import cv2
        except ImportError:
            raise RuntimeError("Importing videos needs OpenCV:  pip install opencv-python")
        video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            raise ValueError(f"Video not found: {video_path}")
        stride = max(1, int(stride))
        session_id = (session_name or self.default_video_session_name(video_path)).strip()
        err = self.validate_session_name(session_id)
        if err:
            raise ValueError(err)
        prev = self.manifest['sessions_imported'].get(session_id)
        in_use = any(i.get('session_id') == session_id for i in self.manifest['images'].values())
        if (prev is not None or in_use) and (prev or {}).get('kind') != 'video':
            raise ValueError(f"A set named '{session_id}' already exists (it is not a video). "
                             "Pick another name.")
        if prev is not None and prev.get('path') and os.path.normcase(prev['path']) != os.path.normcase(video_path):
            raise ValueError(f"A video set named '{session_id}' already exists from a different file "
                             f"({prev['path']}). Pick another name.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Couldn't open the video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        sess_out = os.path.join(self.sessions_dir, session_id)
        os.makedirs(sess_out, exist_ok=True)
        new_ids, skipped, frame_no = [], 0, -1
        try:
            while True:
                ok = cap.grab()
                if not ok:
                    break
                frame_no += 1
                if frame_no % stride:
                    continue
                if progress_cb:
                    progress_cb(frame_no + 1, total)
                img_id = f"vid_{session_id}_{frame_no:06d}"
                if img_id in self.manifest['images']:
                    skipped += 1
                    continue
                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    continue
                h, w = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(
                    os.path.join(self.images_dir, self.UNANNOTATED_SUBDIR, f"{img_id}.jpg"), quality=95)
                sharp = float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                self.manifest['images'][img_id] = {
                    'source': 'video',
                    'source_name': session_id,
                    'session_id': session_id,
                    'sample_index': frame_no,
                    'video_time_s': (frame_no / fps) if fps > 0 else None,
                    'time_unix': None,
                    'annotated': False,
                    'ai_pending': False,
                    'width': w,
                    'height': h,
                    'has_depth': False,
                    'depth_pixels': 0,
                    'sharpness': sharp,
                    'quarter_turns_cw': 0,
                }
                new_ids.append(img_id)
        finally:
            cap.release()
        if not new_ids and not skipped:
            raise ValueError("No frames could be read from that video.")
        with open(os.path.join(sess_out, 'video_info.json'), 'w', encoding='utf-8') as f:
            json.dump({'video': video_path, 'fps': fps, 'frames_in_video': frame_no + 1, 'stride': stride}, f, indent=1)
        old = (prev or {}).get('samples_imported', 0)
        self.manifest['sessions_imported'][session_id] = {
            'path': video_path, 'kind': 'video', 'stride': stride, 'fps': fps,
            'samples_imported': old + len(new_ids), 'copied_lidar': False}
        self.save_manifest()
        return {'new_ids': new_ids, 'skipped_existing': skipped, 'session_id': session_id,
                'frames_seen': frame_no + 1, 'fps': fps}

    # -- Gimble session management ------------------------------------------------

    def delete_images(self, img_ids):
        """delete_image() for many ids with a single manifest write (a session can be hundreds
        of samples). Returns how many were removed."""
        n = 0
        for img_id in list(img_ids):
            if img_id in self.manifest['images']:
                self._delete_image_no_save(img_id)
                n += 1
        if n:
            self.save_manifest()
        return n

    def session_summary(self):
        """One dict per Gimble session that has samples in the project, sorted by name:
        session_id, samples, annotated, with_depth, boxes_with_distance, source_path."""
        out = {}
        for img_id, info in self.manifest['images'].items():
            sid = info.get('session_id')
            if not sid:
                continue
            row = out.setdefault(sid, {'session_id': sid, 'samples': 0, 'annotated': 0,
                                       'with_depth': 0, 'source_path': '',
                                       'role': self.session_role(sid)})
            row['samples'] += 1
            row['annotated'] += 1 if info.get('annotated') else 0
            row['with_depth'] += 1 if self.has_depth(img_id) else 0
        for sid, row in out.items():
            row['source_path'] = self.manifest['sessions_imported'].get(sid, {}).get('path', '')
        return [out[k] for k in sorted(out)]

    def session_role(self, session_id):
        return 'val' if self.manifest['session_roles'].get(session_id) == 'val' else 'train'

    def is_val_image(self, img_id):
        sid = self.manifest['images'].get(img_id, {}).get('session_id')
        return bool(sid) and self.session_role(sid) == 'val'

    def set_session_role(self, session_id, role):
        """role 'val' holds the whole set out for validation; 'train' (default) trains on it.
        Validation images are never augmented, so making a set 'val' deletes its distorted
        copies. (Switching back to 'train' regenerates them automatically.) Call
        refresh_exports() afterwards. Returns how many distorted copies were removed."""
        if role not in ('train', 'val'):
            raise ValueError("role must be 'train' or 'val'")
        if session_id not in {i.get('session_id') for i in self.manifest['images'].values()}:
            raise ValueError(f"No imported session named '{session_id}'.")
        removed = 0
        if role == 'val':
            self.manifest['session_roles'][session_id] = 'val'
            for img_id in self.session_image_ids(session_id):
                for v in self.manifest['distortions'].pop(img_id, []):
                    for p in (os.path.join(self.dist_images_dir, f"{v}.jpg"),
                              os.path.join(self.dist_labels_dir, f"{v}.txt")):
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except OSError:
                            pass
                    removed += 1
        else:
            self.manifest['session_roles'].pop(session_id, None)
        self.save_manifest()
        return removed

    def session_progress(self):
        """{session_id: (total_samples, annotated_samples)} straight from the manifest (no disk
        access, cheap enough to call after every edit)."""
        out = {}
        for info in self.manifest['images'].values():
            sid = info.get('session_id')
            if sid:
                t, a = out.get(sid, (0, 0))
                out[sid] = (t + 1, a + (1 if info.get('annotated') else 0))
        return out

    # -- thinning a set --------------------------------------------------------------

    THUMB_SIZE = (32, 24)

    def _frame_order_key(self, img_id):
        info = self.manifest['images'][img_id]
        idx = info.get('sample_index')
        return (0, idx, img_id) if isinstance(idx, (int, float)) else (1, 0, img_id)

    def session_frames_in_order(self, session_id):
        """The set's image ids in recording order (frame / sample number)."""
        return sorted(self.session_image_ids(session_id), key=self._frame_order_key)

    def frame_thumbnails(self, session_id, progress_cb=None):
        """{img_id: small grayscale float32 array} for every frame of the set (cheap to compare)."""
        out = {}
        ids = self.session_frames_in_order(session_id)
        for n, img_id in enumerate(ids, 1):
            if progress_cb:
                progress_cb(n, len(ids))
            try:
                with Image.open(self.image_path(img_id)) as im:
                    im.draft('L', (self.THUMB_SIZE[0] * 4, self.THUMB_SIZE[1] * 4))
                    out[img_id] = np.asarray(im.convert('L').resize(self.THUMB_SIZE, Image.BILINEAR), dtype=np.float32)
            except OSError:
                continue
        return out

    def plan_thin(self, session_id, every_n=None, difference_pct=None, thumbs=None):
        """Which frames of a set a thinning would remove - nothing is changed here.
        Annotated frames are NEVER removed (they carry your work).
          every_n=N           keep every Nth frame of the recording (frames 1, N+1, 2N+1, ...)
          difference_pct=P    walk the recording and drop each frame that differs from the last KEPT frame
                              by less than P percent (mean brightness difference of a tiny thumbnail), i.e.
                              near-duplicates; annotated frames count as kept references. Needs `thumbs`
                              (from frame_thumbnails).
        Returns dict(remove=[ids], keep=[ids], total, annotated)."""
        ids = self.session_frames_in_order(session_id)
        if (every_n is None) == (difference_pct is None):
            raise ValueError("Choose either 'keep every Nth' or a difference threshold.")
        remove = []
        if every_n is not None:
            n = max(1, int(every_n))
            for pos, i in enumerate(ids):
                if pos % n and not self.is_annotated(i):
                    remove.append(i)
        else:
            if thumbs is None:
                raise ValueError("thumbs are needed for a difference threshold.")
            ref = None
            for i in ids:
                t = thumbs.get(i)
                if self.is_annotated(i) or t is None or ref is None:
                    ref = t if t is not None else ref
                    continue
                diff = float(np.abs(t - ref).mean()) / 255.0 * 100.0
                if diff < float(difference_pct):
                    remove.append(i)
                else:
                    ref = t
        rm = set(remove)
        return {'remove': remove, 'keep': [i for i in ids if i not in rm], 'total': len(ids),
                'annotated': sum(1 for i in ids if self.is_annotated(i))}

    def thin_session(self, session_id, ids):
        """Permanently removes the given frames of a set (as chosen by plan_thin). Annotated frames are
        refused even if listed. Call refresh_exports() afterwards. Returns how many were removed."""
        mine = set(self.session_image_ids(session_id))
        todo = [i for i in ids if i in mine and not self.is_annotated(i)]
        return self.delete_images(todo)

    def session_image_ids(self, session_id):
        return sorted(i for i, info in self.manifest['images'].items()
                      if info.get('session_id') == session_id)

    @staticmethod
    def validate_session_name(name):
        """Returns an error string, or None if `name` can be used as a session name."""
        name = (name or '').strip()
        if not name:
            return "The name can't be empty."
        bad = _INVALID_ID_CHARS & set(name)
        if bad:
            return ("Names can't contain: " + ' '.join(repr(c) if c == ' ' else c
                                                        for c in sorted(bad))
                    + "  (use _ or - instead of spaces)")
        if len(name) > 80:
            return "The name is too long (80 characters max)."
        if name in ('.', '..'):
            return "That name isn't allowed."
        return None

    def rename_session(self, old, new):
        """Renames an imported session (Gimble or video) everywhere: image ids gim_<old>_NNNNNN ->
        gim_<new>_NNNNNN (vid_ likewise) (images, labels, depth, LiDAR, distance picks, distorted variants),
        the manifest entries, and sessions/<old>. Returns {'renamed': n, 'id_map': {old: new}}.
        Exports are NOT rewritten here - call refresh_exports() afterwards so train.txt / val.txt
        / class-filtered exports pick up the new names. Raises ValueError for a bad or taken
        name (nothing is changed in that case)."""
        new = (new or '').strip()
        if old not in {i.get('session_id') for i in self.manifest['images'].values()}:
            raise ValueError(f"No imported session named '{old}'.")
        err = self.validate_session_name(new)
        if err:
            raise ValueError(err)
        if new == old:
            raise ValueError("That is already the session's name.")
        existing = {i.get('session_id') for i in self.manifest['images'].values()}
        if new in existing or new in self.manifest['sessions_imported'] \
                or os.path.exists(os.path.join(self.sessions_dir, new)):
            raise ValueError(f"A session named '{new}' already exists.")

        id_map = {}
        for img_id, info in self.manifest['images'].items():
            if info.get('session_id') == old:
                id_map[img_id] = img_id
                for tag in ('gim', 'vid'):                  # Gimble samples / video frames
                    pre = f"{tag}_{old}_"
                    if img_id.startswith(pre):
                        id_map[img_id] = f"{tag}_{new}_" + img_id[len(pre):]
                        break
        clash = [n for o, n in id_map.items() if n != o and n in self.manifest['images']]
        if clash:
            raise ValueError(f"Renaming would collide with existing image id {clash[0]}.")

        def mv(a, b):
            if a != b and os.path.exists(a):
                os.replace(a, b)

        for o, n in id_map.items():
            if o != n:
                for sub in (self.ANNOTATED_SUBDIR, self.UNANNOTATED_SUBDIR):
                    mv(os.path.join(self.images_dir, sub, f"{o}.jpg"),
                       os.path.join(self.images_dir, sub, f"{n}.jpg"))
                    mv(os.path.join(self.labels_dir, sub, f"{o}.txt"),
                       os.path.join(self.labels_dir, sub, f"{n}.txt"))
                for a, b in zip(self._sidecar_paths(o), self._sidecar_paths(n)):
                    mv(a, b)
                variants = self.manifest['distortions'].pop(o, None)
                if variants is not None:
                    new_variants = []
                    for v in variants:
                        nv = n + v[len(o):] if v.startswith(o) else v
                        mv(os.path.join(self.dist_images_dir, f"{v}.jpg"),
                           os.path.join(self.dist_images_dir, f"{nv}.jpg"))
                        mv(os.path.join(self.dist_labels_dir, f"{v}.txt"),
                           os.path.join(self.dist_labels_dir, f"{nv}.txt"))
                        new_variants.append(nv)
                    self.manifest['distortions'][n] = new_variants
            info = self.manifest['images'].pop(o)
            info['session_id'] = new
            if info.get('source_name') == old:
                info['source_name'] = new
            self.manifest['images'][n] = info

        mv(os.path.join(self.sessions_dir, old), os.path.join(self.sessions_dir, new))
        if old in self.manifest['sessions_imported']:
            self.manifest['sessions_imported'][new] = self.manifest['sessions_imported'].pop(old)
        if old in self.manifest['session_roles']:
            self.manifest['session_roles'][new] = self.manifest['session_roles'].pop(old)
        self.save_manifest()
        return {'renamed': len(id_map), 'id_map': id_map}

    def delete_session(self, session_id):
        """Permanently removes every sample of an imported session (images, labels, depth,
        LiDAR, distance picks, distorted variants) plus its sessions/<id> folder and import
        record. The original recording on the drive is not touched. Call refresh_exports()
        afterwards. Returns the number of samples removed."""
        ids = self.session_image_ids(session_id)
        n = self.delete_images(ids)
        shutil.rmtree(os.path.join(self.sessions_dir, session_id), ignore_errors=True)
        self.manifest['sessions_imported'].pop(session_id, None)
        self.manifest['session_roles'].pop(session_id, None)
        self.save_manifest()
        return n

    # -- exports on disk ------------------------------------------------------------

    def existing_exports(self):
        """[{'name': '(all classes)' or <folder>, 'classes': None | [names], 'root': dir,
        'yaml': path}] for every data.yaml this project has written."""
        found = []

        def parse_names(path):
            names = {}
            try:
                with open(path, encoding='utf-8') as f:
                    in_names = False
                    for line in f:
                        if line.startswith('names:'):
                            in_names = True
                        elif in_names and ':' in line:
                            k, v = line.strip().split(':', 1)
                            if k.isdigit():
                                names[int(k)] = v.strip()
            except OSError:
                pass
            return [names[k] for k in sorted(names)]

        root_yaml = os.path.join(self.root, 'data.yaml')
        if os.path.exists(root_yaml):
            found.append({'name': '(all classes)', 'classes': None,
                          'root': self.root, 'yaml': root_yaml})
        exp_dir = os.path.join(self.root, 'exports')
        if os.path.isdir(exp_dir):
            for name in sorted(os.listdir(exp_dir)):
                y = os.path.join(exp_dir, name, 'data.yaml')
                if os.path.exists(y):
                    names = [c for c in parse_names(y) if c in self.classes]
                    found.append({'name': name, 'classes': names,
                                  'root': os.path.join(exp_dir, name), 'yaml': y})
        return found

    def refresh_exports(self, progress_cb=None):
        """Re-runs every export that already exists on disk with its last settings, so
        train.txt / val.txt / distances.csv (and copies in class-filtered exports) reflect
        renamed or deleted sessions. Returns the list of export names refreshed."""
        st = self.manifest.get('export_settings', {})
        done = []
        for ex in self.existing_exports():
            self.export_dataset(include_distortions=st.get('include_distortions', True),
                                distortion_count=st.get('distortion_count', DEFAULT_DISTORTION_COUNT),
                                progress_cb=progress_cb, classes=ex['classes'])
            done.append(ex['name'])
        return done

    # -- distortion ------------------------------------------------------

    def generate_distortions(self, img_id, n_variants, ranges=None, seed=None):
        """Creates n_variants photometrically-distorted copies of img_id,
        each with a label file identical to the source's (copied, not
        recomputed - correct precisely because these transforms never
        move anything). Skips creating new variant IMAGES beyond what
        already exists unless you ask for more than already exist - but
        every call, regardless of whether any new variant gets created,
        first re-syncs every EXISTING variant's label file to the
        source's current boxes. This matters because annotation happens
        incrementally: a box can be added, deleted, or relabeled after
        distortions were already generated for that image, and a stale
        label file would silently train on the wrong answer. Re-copying a
        label file is cheap (a few lines of text), so doing it on every
        call - even when nothing else changes - costs effectively
        nothing."""
        if self.is_val_image(img_id):
            return []       # validation images are never augmented
        existing = self.manifest['distortions'].get(img_id, [])
        src_boxes = self.get_boxes(img_id)
        for variant_id in existing:
            save_label_file(os.path.join(self.dist_labels_dir, f"{variant_id}.txt"), src_boxes)

        need = n_variants - len(existing)
        if need <= 0:
            return existing

        rng = np.random.default_rng(seed)
        src_img = Image.open(self.image_path(img_id)).convert('RGB')

        new_ids = list(existing)
        for i in range(len(existing), n_variants):
            variant_id = f"{img_id}_dist{i}"
            distorted = distort_image(src_img, rng, ranges)
            distorted.save(os.path.join(self.dist_images_dir, f"{variant_id}.jpg"), quality=90)
            save_label_file(os.path.join(self.dist_labels_dir, f"{variant_id}.txt"), src_boxes)
            new_ids.append(variant_id)

        self.manifest['distortions'][img_id] = new_ids
        self.save_manifest()
        return new_ids

    def ensure_all_distortions(self, n_variants=DEFAULT_DISTORTION_COUNT, ranges=None, progress_cb=None):
        """Sweeps every HUMAN-CONFIRMED (annotated) image with at least one
        box and makes sure it has n_variants distorted copies, re-syncing
        labels along the way (see generate_distortions). Skips images with
        zero boxes entirely - there's nothing useful to distort an empty
        annotation into.

        Deliberately checks is_annotated(), not just "has boxes": an
        unreviewed AI draft (ai_pending) has boxes on disk too, but it
        isn't trustworthy ground truth yet and the model's own live-assist
        re-prediction can replace those boxes at any moment (see
        annotator_gui.py's _trigger_live_predict) - distorting it would be
        wasted image/disk work now and stale labels later. Only a box set
        that's actually been confirmed via set_boxes() is worth the cost.

        Safe and cheap to call anytime: an image that already meets the
        target just gets its existing labels re-synced, no new image
        processing. This is what makes distortion "automatic" - the GUI
        calls this after every box edit instead of exposing a manual
        button, and export_dataset() below calls it once more as a final
        safety net. Pass progress_cb(done, total) to report progress on a
        sweep that can process hundreds of images."""
        targets = [img_id for img_id in self.list_images()
                   if self.is_annotated(img_id) and self.get_boxes(img_id) and not self.is_val_image(img_id)]
        for n, img_id in enumerate(targets, 1):
            self.generate_distortions(img_id, n_variants, ranges=ranges)
            if progress_cb:
                progress_cb(n, len(targets))

    # -- export ------------------------------------------------------------

    def export_dataset(self, include_distortions=True,
                        ensure_distortions=True, distortion_count=DEFAULT_DISTORTION_COUNT,
                        progress_cb=None, classes=None, val_frac=None, seed=None):
        """Writes data.yaml plus train.txt/val.txt (Ultralytics-style path
        lists) covering every annotated source image.

        The train/val split is chosen per SET, not per image: every annotated frame of a session
        marked 'val' (Project.set_session_role) goes to val.txt, everything else to train.txt.
        Neighbouring frames of one recording are near-duplicates, so any split INSIDE a
        recording would put twins on both sides and inflate the validation score. Validation
        frames are never augmented; distorted copies exist only for training frames.
        (val_frac / seed are accepted but ignored - kept so older callers don't break.)

        Also ensures distortion coverage (see ensure_all_distortions)
        before writing anything, by default - a final safety net so
        export always includes every annotated image's variants even if
        the background auto-distort in the GUI never got a chance to run
        for one (e.g. the app was closed right after drawing a box). Pass
        ensure_distortions=False to skip this and export exactly whatever
        already exists on disk. Pass progress_cb(done, total) to report
        progress on that sweep - the GUI uses this to keep the window
        responsive and show real feedback instead of appearing to hang.

        classes: optional list of class NAMES to train on right now, e.g.
        ['person']. Every annotated image is still included either way -
        an image with no boxes in the selected class(es) becomes a valid
        empty-label "definitely nothing here" negative example, not a
        wasted one - but each image's boxes are filtered down to just the
        selected classes and remapped to a fresh 0-indexed id space used
        only by this export. The filtered copies live under
        exports/<class names>/ instead of overwriting anything - your
        full multi-class annotations in labels/ are never touched, so a
        narrow export now (person only) and a wider one later (once
        you've annotated more of the other classes) both always work from
        the same underlying ground truth. None (default) exports every
        class exactly as before, referencing your original files in place
        with no copying."""
        if include_distortions and ensure_distortions:
            self.ensure_all_distortions(distortion_count, progress_cb=progress_cb)

        self.manifest['export_settings'] = {
            'include_distortions': include_distortions, 'distortion_count': distortion_count}
        self.save_manifest()

        # Validation is chosen per SET (see set_session_role): every annotated frame of a set
        # marked 'val' is validation, everything else trains. Neighbouring frames of a recording
        # are near-duplicates, so splitting inside a recording would leak and inflate the scores.
        annotated = [i for i in self.list_images() if self.is_annotated(i)]
        val_sources = {i for i in annotated if self.is_val_image(i)}

        if classes is None:
            out_classes = self.classes
            remap = {i: i for i in range(len(self.classes))}
            export_root = self.root
            exp_images_dir = None  # None means "reference original files in place"
            exp_labels_dir = None
        else:
            unknown = [c for c in classes if c not in self.classes]
            if unknown:
                raise ValueError(f"Unknown class name(s) for export: {unknown} - choices are: {self.classes}")
            selected_idx = [i for i, name in enumerate(self.classes) if name in classes]
            out_classes = [self.classes[i] for i in selected_idx]
            remap = {old: new for new, old in enumerate(selected_idx)}
            export_name = "_".join(out_classes) if out_classes else "empty"
            export_root = os.path.join(self.root, 'exports', export_name)
            exp_images_dir = os.path.join(export_root, 'images')
            exp_labels_dir = os.path.join(export_root, 'labels')
            os.makedirs(exp_images_dir, exist_ok=True)
            os.makedirs(exp_labels_dir, exist_ok=True)

        def export_one(img_id, src_image_path, src_boxes):
            """Returns the path to use in train.txt/val.txt for this
            image, writing a copy + filtered label first if this is a
            class-filtered export."""
            if exp_images_dir is None:
                return os.path.abspath(src_image_path)
            dst_image_path = os.path.join(exp_images_dir, f"{img_id}.jpg")
            # Skip re-copying an image that's already an up-to-date copy -
            # same incremental-cache spirit as generate_distortions() only
            # creating what's missing, so re-running a filtered export
            # after adding a few more annotated images doesn't re-copy
            # everything that was already there.
            if not os.path.exists(dst_image_path) or os.path.getmtime(dst_image_path) < os.path.getmtime(src_image_path):
                shutil.copyfile(src_image_path, dst_image_path)
            filtered = [(remap[cls], xc, yc, w, h) for (cls, xc, yc, w, h) in src_boxes if cls in remap]
            save_label_file(os.path.join(exp_labels_dir, f"{img_id}.txt"), filtered)
            return os.path.abspath(dst_image_path)

        train_paths, val_paths = [], []
        dist_rows = []
        kept_names = set()

        def add_dist_rows(split, out_id, src_id, boxes, picks):
            for (cls, xc, yc, w, h), pk in zip(boxes, picks):
                if cls not in remap:
                    continue
                pk = pk or {}
                dist_rows.append([
                    split, out_id, remap[cls], self.classes[cls] if cls < len(self.classes) else cls,
                    f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}",
                    f"{pk['distance_m']:.3f}" if 'distance_m' in pk else '',
                    pk.get('u', ''), pk.get('v', ''),
                    src_id, f"depth/{src_id}.png" if self.has_depth(src_id) else ''])

        for img_id in annotated:
            split = 'val' if img_id in val_sources else 'train'
            bucket = val_paths if split == 'val' else train_paths
            src_boxes = self.get_boxes(img_id)
            picks = self.get_distances(img_id, len(src_boxes))
            bucket.append(export_one(img_id, self.image_path(img_id), src_boxes))
            kept_names.add(img_id)
            add_dist_rows(split, img_id, img_id, src_boxes, picks)
            if include_distortions:
                for variant_id in self.manifest['distortions'].get(img_id, []):
                    # distortions never move box geometry or change class
                    # ids, so the source's own (already-fetched) boxes are
                    # exactly this variant's boxes too - no need to re-read
                    # its label file from disk. Same for the LiDAR distance picks.
                    bucket.append(export_one(
                        variant_id, os.path.join(self.dist_images_dir, f"{variant_id}.jpg"), src_boxes))
                    kept_names.add(variant_id)
                    add_dist_rows(split, variant_id, img_id, src_boxes, picks)

        if exp_images_dir is not None:
            # drop copies of images that are no longer part of the project (deleted / renamed)
            for folder, ext in ((exp_images_dir, '.jpg'), (exp_labels_dir, '.txt')):
                for fn in os.listdir(folder):
                    stem, e = os.path.splitext(fn)
                    if e == ext and stem not in kept_names:
                        try:
                            os.remove(os.path.join(folder, fn))
                        except OSError:
                            pass

        import csv
        with open(os.path.join(export_root, 'distances.csv'), 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['split', 'image', 'class_id', 'class', 'xc', 'yc', 'w', 'h',
                         'distance_m', 'point_u', 'point_v', 'source_image', 'source_depth_png'])
            wr.writerows(dist_rows)

        with open(os.path.join(export_root, 'train.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_paths) + ('\n' if train_paths else ''))
        with open(os.path.join(export_root, 'val.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_paths) + ('\n' if val_paths else ''))

        yaml_path = os.path.join(export_root, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"train: {os.path.abspath(os.path.join(export_root, 'train.txt'))}\n")
            f.write(f"val: {os.path.abspath(os.path.join(export_root, 'val.txt'))}\n")
            f.write(f"nc: {len(out_classes)}\n")
            f.write("names:\n")
            for i, name in enumerate(out_classes):
                f.write(f"  {i}: {name}\n")

        return {
            'train_count': len(train_paths),
            'val_count': len(val_paths),
            'n_source_images': len(annotated),
            'n_val_sources': len(val_sources),
            'val_sessions': sorted(r['session_id'] for r in self.session_summary() if r['role'] == 'val'),
            'train_sessions': sorted({self.manifest['images'][i].get('session_id') or '(uploads)' for i in annotated if i not in val_sources}),
            'no_validation': len(val_sources) == 0,
            'data_yaml': yaml_path,
            'distances_csv': os.path.join(export_root, 'distances.csv'),
            'n_distance_boxes': sum(1 for r in dist_rows if r[8] != ''),
            'classes': out_classes,
        }
