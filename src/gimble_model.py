"""
gimble_model.py - one network that outputs class, box AND distance for every object.

Built on Ultralytics' YOLO26 detector (random weights, nothing pretrained) plus a small
distance head. The detector part is exactly the stock YOLO26 and keeps its own loss; the
distance head reads the SAME backbone/neck features, so both tasks share one body:

    image -> backbone/neck -> detection head -> class + box          (normal YOLO)
                          \\-> distance head (RoI-pooled features + box geometry) -> metres

Why the distance head is per box (not per pixel): you annotate boxes and attach a LiDAR
distance when you have one. Size, shape and position of an object in the picture carry the
distance, so the head sees (a) neck features pooled inside the box and (b) the box geometry.
Boxes with no distance simply contribute no distance loss (masked), so every other labeled
box still trains class and position.

Two other pieces live here:
  * class-ignore masking: for images where only some classes were labeled (the old people-only
    frames) the classification loss of the unlabeled classes is switched off, so unlabeled trash
    or vehicles in those frames are NOT taught as background;
  * checkpoint save/load for this model (it is not a stock Ultralytics .pt).

Tested against Ultralytics 8.4.x. Different major versions may change the head/loss internals;
check_ultralytics() warns when the installed version is untested.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TESTED_ULTRALYTICS = "8.4"
MODEL_FORMAT = "gimble-det-dist-v1"
D_MIN, D_MAX = 0.2, 60.0          # metres - distances are modelled in log space inside this range
IGNORE_LOGIT = -20.0              # logit that makes BCE(target=0) loss and gradient ~0


def check_ultralytics():
    import ultralytics
    ver = ultralytics.__version__
    if not ver.startswith(TESTED_ULTRALYTICS):
        print(f"WARNING: Ultralytics {ver} installed; this code was tested with {TESTED_ULTRALYTICS}.x. "
              "If training errors out inside the detection head/loss, install a matching version.")
    return ver


class DistanceHead(nn.Module):
    """Predicts log-distance for boxes from RoI-pooled neck features + box geometry."""

    def __init__(self, chs, strides, pool=5, mid=32, hidden=256):
        super().__init__()
        self.strides = [float(s) for s in strides]
        self.pool = pool
        self.reduce = nn.ModuleList(nn.Sequential(nn.Conv2d(c, mid, 1), nn.SiLU()) for c in chs)
        nfeat = mid * pool * pool * len(chs) + 6
        self.mlp = nn.Sequential(nn.Linear(nfeat, hidden), nn.SiLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        self.mlp[-1].bias.data.fill_(math.log(6.0))      # start at "about 6 m"

    def forward(self, feats, rois, img_hw):
        """feats: list of (B,C,H,W) neck maps. rois: (N,5) [batch_idx, x1,y1,x2,y2] in input-image pixels.
        Returns (N,) predicted log-distance (natural log of metres)."""
        from torchvision.ops import roi_align
        if rois.shape[0] == 0:
            return rois.new_zeros(0)
        rois = rois.float()
        pooled = []
        for f, s, red in zip(feats, self.strides, self.reduce):
            p = roi_align(f.float(), rois, output_size=self.pool, spatial_scale=1.0 / s,
                          sampling_ratio=2, aligned=True)
            pooled.append(red(p).flatten(1))
        H, W = img_hw
        x1, y1, x2, y2 = rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
        w, h = (x2 - x1).clamp(min=1.0), (y2 - y1).clamp(min=1.0)
        geo = torch.stack([
            (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,           # where the box centre is
            y2 / H,                                          # where its bottom edge (feet) is
            torch.log(h / H), torch.log(w / W),              # size (log): distance ~ 1/size
            torch.log(h / w),                                # shape
        ], 1)
        return self.mlp(torch.cat(pooled + [geo], 1)).squeeze(1)


class GimbleNet(nn.Module):
    """YOLO26 detector + DistanceHead sharing one body."""

    def __init__(self, names, cfg="yolo26n.yaml"):
        super().__init__()
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import IterableSimpleNamespace
        self.cfg = cfg
        self.names = list(names)
        self.det = DetectionModel(cfg, nc=len(self.names), verbose=False)
        self.det.names = {i: n for i, n in enumerate(self.names)}
        self.det.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=100)
        head = self.det.model[-1]
        chs = [m[0][0].in_channels if isinstance(m[0], nn.Sequential) else m[0].conv.in_channels for m in head.cv2]
        self.dist = DistanceHead(chs, head.stride.tolist())
        self.nc = len(self.names)

    @property
    def stride(self):
        return self.det.stride

    # ---- class changes ------------------------------------------------------------------
    def class_cls_convs(self):
        """The final 1x1 conv of each classification branch (one output channel per class)."""
        head = self.det.model[-1]
        convs = []
        for name in ("cv3", "one2one_cv3"):
            for seq in getattr(head, name, []):
                convs.append(seq[-1] if isinstance(seq, nn.Sequential) else seq)
        return convs

    def adopt(self, old):
        """Copy everything `old` (a GimbleNet) learned into this net, matching classes BY NAME: classes both
        have keep their learned output rows, new classes start fresh, classes only `old` had are left
        behind. Everything that isn't per-class (backbone, neck, box head, distance head) is copied whole.
        Returns (kept, new, dropped) name lists."""
        sd_new = self.state_dict()
        copied = {}
        for k, v in old.state_dict().items():
            if k in sd_new and sd_new[k].shape == v.shape:
                copied[k] = v
        self.load_state_dict(copied, strict=False)
        kept = [n for n in self.names if n in old.names]
        new = [n for n in self.names if n not in old.names]
        dropped = [n for n in old.names if n not in self.names]
        # per-class output rows, by name
        olds, news = old.class_cls_convs(), self.class_cls_convs()
        with torch.no_grad():
            for mo, mn in zip(olds, news):
                for n in kept:
                    i_old, i_new = old.names.index(n), self.names.index(n)
                    mn.weight[i_new].copy_(mo.weight[i_old])
                    if mn.bias is not None and mo.bias is not None:
                        mn.bias[i_new].copy_(mo.bias[i_old])
        return kept, new, dropped

    # ---- training -----------------------------------------------------------------------
    def set_epochs(self, epochs):
        self.det.args.epochs = int(epochs)

    def forward_train(self, batch, dist_gain=2.0, jitter=0.05):
        """batch: dict img (B,3,H,W) float 0-1, batch_idx (N,), cls (N,), bboxes (N,4) normalised xywh,
        dist (N,) metres or NaN, class_mask (B,nc) bool = classes that ARE labeled in that image.
        Returns (total_loss, items dict of floats). total_loss already includes the batch-size factor
        Ultralytics uses, so it can be backpropagated directly."""
        img = batch["img"]
        B, _, H, W = img.shape
        preds = self.det(img)                                   # training mode -> dict
        cmask = batch["class_mask"]
        if not bool(cmask.all()):
            ignore = (~cmask)[:, :, None]                       # (B, nc, 1) - broadcast over anchors
            for k in ("one2many", "one2one"):
                if k in preds:
                    s = preds[k]["scores"]
                    preds[k]["scores"] = torch.where(ignore, torch.full_like(s, IGNORE_LOGIT), s)
        if getattr(self.det, "criterion", None) is None:
            self.det.criterion = self.det.init_criterion()
        det_loss, det_items = self.det.criterion(preds, batch)
        det_loss = det_loss.sum()

        # ---- distance: only boxes that carry one ----
        feats = (preds["one2many"] if "one2many" in preds else preds)["feats"]
        d = batch["dist"]
        ok = torch.isfinite(d)
        dist_loss = det_loss.new_zeros(())
        n_ok = int(ok.sum())
        if n_ok:
            xywh = batch["bboxes"][ok] * torch.tensor([W, H, W, H], device=img.device)
            if jitter > 0:                                       # detector boxes are never perfect
                jit = (torch.rand_like(xywh) * 2 - 1) * jitter * xywh[:, 2:].repeat(1, 2)
                xywh = xywh + jit
            xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
            rois = torch.cat([batch["batch_idx"][ok].view(-1, 1).float(), xyxy], 1)
            pred = self.dist(feats, rois, (H, W))
            target = torch.log(d[ok].clamp(D_MIN, D_MAX))
            dist_loss = F.smooth_l1_loss(pred, target, beta=0.1)
            dist_loss = dist_loss * dist_gain * B
        total = det_loss + dist_loss
        if isinstance(det_items, dict):
            items = {k: float(v) for k, v in det_items.items()}
        else:                                   # some Ultralytics versions return a tensor: box, cls, dfl, ...
            vals = det_items.detach().flatten().tolist()
            items = dict(zip(("box_loss", "cls_loss", "dfl_loss"), vals))
        items["dist_loss"] = float(dist_loss.detach()) / max(B, 1) / max(dist_gain, 1e-9)
        items["n_dist"] = n_ok
        return total, items

    # ---- inference ----------------------------------------------------------------------
    @torch.no_grad()
    def infer(self, img, conf=0.25, max_det=300):
        """img: (B,3,H,W) float 0-1. Returns list (per image) of tensors (n, 7):
        x1,y1,x2,y2,conf,cls,distance_m in input-image pixels."""
        self.eval()
        self.det.end2end = True
        out = self.det(img)
        y, preds = out if isinstance(out, tuple) else (out, None)
        feats = preds["one2one"]["feats"]
        B, _, H, W = img.shape
        res = []
        rois, owners, keep_list = [], [], []
        for b in range(B):
            det = y[b]
            det = det[det[:, 4] >= conf][:max_det]
            keep_list.append(det)
            rois.append(torch.cat([torch.full((len(det), 1), float(b), device=img.device), det[:, :4]], 1))
        rois = torch.cat(rois, 0)
        logd = self.dist(feats, rois, (H, W)) if len(rois) else rois.new_zeros(0)
        dm = torch.exp(logd).clamp(D_MIN, D_MAX)
        i = 0
        for det in keep_list:
            n = len(det)
            res.append(torch.cat([det[:, :6], dm[i:i + n, None]], 1))
            i += n
        return res

    @torch.no_grad()
    def distance_for_boxes(self, img, boxes_per_image):
        """Distance (m) for given boxes (list of (n,4) xyxy tensors in input pixels). Used to score the
        distance head on ground-truth boxes."""
        self.eval()
        self.det.end2end = True
        out = self.det(img)
        preds = out[1] if isinstance(out, tuple) else out
        feats = preds["one2one"]["feats"]
        B, _, H, W = img.shape
        rois = torch.cat([torch.cat([torch.full((len(b), 1), float(i), device=img.device), b], 1)
                          for i, b in enumerate(boxes_per_image)], 0)
        if not len(rois):
            return [torch.zeros(0) for _ in boxes_per_image]
        dm = torch.exp(self.dist(feats, rois, (H, W))).clamp(D_MIN, D_MAX)
        out, i = [], 0
        for b in boxes_per_image:
            out.append(dm[i:i + len(b)].cpu())
            i += len(b)
        return out


# ---- checkpoints --------------------------------------------------------------------------

def save_checkpoint(path, model, **extra):
    import ultralytics
    ck = dict(format=MODEL_FORMAT, cfg=model.cfg, names=model.names,
              state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
              ultralytics=ultralytics.__version__, **extra)
    torch.save(ck, path)


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ck, dict) or ck.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path} is not a {MODEL_FORMAT} checkpoint")
    model = GimbleNet(ck["names"], cfg=ck["cfg"])
    model.load_state_dict(ck["state_dict"])
    return model, ck
