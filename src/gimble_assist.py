"""
gimble_assist.py - lets the annotator pre-label with YOUR from-scratch model.

Same interface as model_assist.ModelAssist (predict_boxes + class_names), so the annotator treats the two
the same way. The from-scratch model lives in ~/yolo_runs/gimble_model/weights/best.pt and is rewritten
after every training round; reload_if_changed() notices that and picks up the new version automatically.
This module only ever READS that file. (The original helper model is ModelAssist's job and is never touched.)
"""
import os

import numpy as np

import gimble_model as gm


class GimbleAssist:
    def __init__(self, checkpoint_path, device=None):
        import torch
        self.path = checkpoint_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.stamp = None
        self.meta = {}
        self.imgsz = 640
        self.class_names = []
        self._load()

    # ---- loading -------------------------------------------------------------------------
    def _load(self):
        st = os.stat(self.path)
        model, ck = gm.load_checkpoint(self.path)
        model.to(self.device).eval()
        self.model = model
        self.imgsz = int(ck.get("imgsz", 640))
        self.class_names = list(model.names)
        self.stamp = (st.st_mtime_ns, st.st_size)
        self.meta = {k: ck.get(k) for k in ("round", "epoch", "map50", "map", "dist_med_rel")}

    def reload_if_changed(self):
        """Loads the checkpoint again if training has replaced it. Returns True when a newer version was
        loaded. A file that is mid-write / unreadable just keeps the current model for now."""
        try:
            st = os.stat(self.path)
            if (st.st_mtime_ns, st.st_size) == self.stamp:
                return False
            self._load()
            return True
        except Exception:  # noqa: BLE001
            return False

    def describe(self):
        m = self.meta
        parts = []
        if m.get("round") is not None:
            parts.append(f"round {m['round']}")
        if m.get("epoch") is not None:
            parts.append(f"epoch {m['epoch']}")
        if m.get("map") is not None:
            parts.append(f"mAP50-95 {float(m['map']):.2f}")
        return ", ".join(parts) or "from-scratch model"

    # ---- prediction ----------------------------------------------------------------------
    def predict_detections(self, image_path, conf_threshold=0.25):
        """[(class_id, xc, yc, w, h, conf, distance_m)] with box values normalised to the original image."""
        import cv2
        import torch
        import gimble_data as gd
        data = np.fromfile(image_path, dtype=np.uint8)          # works with any characters in the path
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return []
        H0, W0 = img.shape[:2]
        lb, r, left, top = gd.letterbox(img, self.imgsz)
        x = torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)))[None].float().div(255).to(self.device)
        det = self.model.infer(x, conf=conf_threshold)[0].cpu().numpy()
        out = []
        for x1, y1, x2, y2, conf, cls, dist in det:
            x1, x2 = np.clip([(x1 - left) / r, (x2 - left) / r], 0, W0)
            y1, y2 = np.clip([(y1 - top) / r, (y2 - top) / r], 0, H0)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            out.append((int(cls), float((x1 + x2) / 2 / W0), float((y1 + y2) / 2 / H0),
                        float((x2 - x1) / W0), float((y2 - y1) / H0), float(conf), float(dist)))
        return out

    def predict_boxes(self, image_path, conf_threshold=0.25):
        """Same return shape as ModelAssist.predict_boxes: [(class_id, xc, yc, w, h)]."""
        return [d[:5] for d in self.predict_detections(image_path, conf_threshold)]


class HelperAssist:
    """A trained HELPER slot (stock Ultralytics checkpoint, e.g. ~/yolo_runs/helper_m/weights/best.pt) for the
    annotator. Wraps model_assist.ModelAssist (which stays untouched) and adds what the slots need: it notices
    when training has replaced best.pt and loads the new version, and it can describe itself."""

    def __init__(self, checkpoint_path, device=None):
        self.path = checkpoint_path
        self.device = device
        self.stamp = None
        self._inner = None
        self._load()

    def _load(self):
        import model_assist
        st = os.stat(self.path)
        inner = model_assist.ModelAssist(self.path, device=self.device)
        self._inner, self.stamp = inner, (st.st_mtime_ns, st.st_size)

    @property
    def class_names(self):
        return self._inner.class_names

    def reload_if_changed(self):
        try:
            st = os.stat(self.path)
            if (st.st_mtime_ns, st.st_size) == self.stamp:
                return False
            self._load()
            return True
        except Exception:  # noqa: BLE001
            return False

    def describe(self):
        """From the run's results.csv (best epoch by mAP50-95); generic text if that is unavailable."""
        import csv
        run_dir = os.path.dirname(os.path.dirname(self.path))
        try:
            with open(os.path.join(run_dir, "results.csv"), newline="", encoding="utf-8") as f:
                rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
            best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
            return f"epoch {int(float(best['epoch']))}/{len(rows)}, mAP50-95 {float(best['metrics/mAP50-95(B)']):.2f}"
        except Exception:  # noqa: BLE001
            return "helper model"

    def predict_boxes(self, image_path, conf_threshold=0.25):
        return self._inner.predict_boxes(image_path, conf_threshold=conf_threshold)
