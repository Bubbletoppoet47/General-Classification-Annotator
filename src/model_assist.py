"""
model_assist.py

Model-assisted pre-labeling: runs a trained YOLO checkpoint over images and
returns its predictions in the same (class_id, x_center, y_center, width,
height) format annotator_core.py already uses, so they can be written
straight into a project's label files as an unreviewed starting point.

Deliberately its own module, separate from annotator_core.py: ultralytics
pulls in torch/torchvision, a much heavier dependency than anything else
this tool needs, and one you only need once you actually have a trained
checkpoint to use. Importing it is deferred to inside ModelAssist.__init__
so the rest of the tool (drawing boxes, exporting a dataset, everything
that doesn't need a model) never pays that cost or requires it installed.

IMPORTANT - what this module does NOT do: it never decides that a
prediction is correct. That's exactly why annotator_core.py's
set_ai_predicted_boxes() (the only place these results should be written)
never marks an image "annotated" - a predicted box is a draft, not ground
truth, however confident it looks. A human still has to look at it.
"""


class ModelAssist:
    """Wraps a trained YOLO checkpoint for pre-labeling. Loads the model
    once (loading weights has real overhead) and reuses it across many
    predict_boxes() calls - construct one instance per pre-labeling batch,
    not one per image."""

    def __init__(self, checkpoint_path, device=None):
        """device=None lets ultralytics pick automatically (CUDA if
        available, else CPU). Inference is light enough that this is
        normally safe to leave on auto - pass 'cpu' explicitly if you want
        to guarantee it never touches the GPU (e.g. training is running
        at the same time)."""
        from ultralytics import YOLO  # heavy import - see module docstring
        self.model = YOLO(checkpoint_path)
        self.device = device
        self.class_names = list(self.model.names.values()) if hasattr(self.model, 'names') else None

    def predict_boxes(self, image_path, conf_threshold=0.25):
        """Returns [(class_id, x_center, y_center, width, height), ...] for
        predictions at or above conf_threshold, in the same normalized
        YOLO format annotator_core.py's label files already use - so the
        result can be passed directly to Project.set_ai_predicted_boxes().
        Returns an empty list if the model found nothing above threshold,
        which is a perfectly normal, useful result (an empty pre-fill
        still saves you from having to open the image and confirm "no,
        there's really nothing here" from a blank canvas)."""
        results = self.model.predict(image_path, device=self.device, verbose=False)
        r = results[0]
        boxes = []
        if r.boxes is None or len(r.boxes) == 0:
            return boxes
        xywhn = r.boxes.xywhn.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        for (xc, yc, w, h), c, p in zip(xywhn, cls, conf):
            if p >= conf_threshold:
                boxes.append((int(c), float(xc), float(yc), float(w), float(h)))
        return boxes
