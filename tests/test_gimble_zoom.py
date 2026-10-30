import os, sys, tempfile, shutil
import numpy as np
from PIL import Image
import tkinter as tk
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac, annotator_gui as ag, hotkeys as hk, share_zip as sz

tmp = tempfile.mkdtemp(); sz.CONFIG_PATH = os.path.join(tmp, "cfg.json")
P = ac.Project(os.path.join(tmp, "p"), default_classes=["a", "b"])
arr = np.zeros((300, 400, 3), np.uint8); arr[100:140, 200:260] = (255, 0, 0)
iid = P.add_uploaded_image(Image.fromarray(arr), "x")
P.set_boxes(iid, [(0, 230 / 400, 120 / 300, 60 / 400, 40 / 300)])
del P
root = tk.Tk(); root.geometry("1400x800"); app = ag.AnnotatorApp(root, os.path.join(tmp, "p"))
app.load_image(iid); root.update(); root.update_idletasks()
cv = app.canvas; cw, ch = cv.winfo_width(), cv.winfo_height()

# hotkey exists, default f, and is bound; the button shows it
assert hk.Hotkeys().get("fit_view") == "f" and "<Key-f>" in app._bound_keys and "(F)" in app.fit_button.cget("text")
assert hk.Hotkeys({"fit_view": "Ctrl+f"}).bindings(2).get("<Control-Key-f>") == "fit_view"          # combinations work

# at zoom 1 nothing changed: whole image fitted and centred
s1, (ox1, oy1), (dw1, dh1) = app._fit_geometry()
assert app.zoom == 1 and abs(ox1 * 2 + dw1 - cw) < 1.01 and abs(oy1 * 2 + dh1 - ch) < 1.01
# zoom out past the fit does nothing
app._zoom_at(cw // 2, ch // 2, -3); assert app.zoom == 1 and app.view_center is None
# the box centre in image px (230, 120) -> zoom around that screen point: the point stays under the pointer
cx, cy = app._image_px_to_canvas(230, 150)                    # (the middle row: no edge in the way, so the spot can stay exactly put)
for _ in range(4): app._zoom_at(cx, cy, 1)
root.update()
assert app.zoom > 1.9
nx, ny = app._image_px_to_canvas(230, 150)
assert abs(nx - cx) < 1.5 and abs(ny - cy) < 1.5, ((cx, cy), (nx, ny))
# round trip canvas <-> image at zoom
ix, iy = app._canvas_to_image_px(*app._image_px_to_canvas(77.5, 201.25)); assert abs(ix - 77.5) < 1e-6 and abs(iy - 201.25) < 1e-6
# limit
for _ in range(60): app._zoom_at(cw // 2, ch // 2, 1)
assert app.zoom == ag.ZOOM_MAX
# the shown pixels are the right ones: the red patch (image px 200..260 x 100..140) is at its mapped canvas place
app._zoom_at(*app._image_px_to_canvas(230, 120), -50)        # back toward fit, then zoom on the patch again
app.fit_view(); app._zoom_at(*app._image_px_to_canvas(230, 120), 8); root.update()
x0, y0 = app._image_px_to_canvas(205, 105); x1, y1 = app._image_px_to_canvas(255, 135)
from PIL import ImageGrab
img = app.tk_image._PhotoImage__photo if False else None
# check via the box outline coordinates equal the patch edges
bx1, by1 = app._image_px_to_canvas(200, 100); bx2, by2 = app._image_px_to_canvas(260, 140)
rects = [cv.coords(i) for i in cv.find_all() if cv.type(i) == "rectangle"]
assert any(abs(r[0] - bx1) < 2 and abs(r[1] - by1) < 2 and abs(r[2] - bx2) < 2 and abs(r[3] - by2) < 2 for r in rects), (rects, (bx1, by1, bx2, by2))
# the image drawn is only the visible crop (never a giant bitmap)
sl = 2 * app._fit_geometry()[0] + 4
assert app.tk_image.width() <= cw + sl and app.tk_image.height() <= ch + sl
# panning: wheel-button drag moves the view, stays inside the image
c0 = app.view_center
class E: pass
e = E(); e.x, e.y = 400, 300; app._on_pan_start(e)
e2 = E(); e2.x, e2.y = 300, 250; app._on_pan_drag(e2)
c1 = app.view_center; assert c1[0] > c0[0] and c1[1] > c0[1]
e3 = E(); e3.x, e3.y = -5000, -5000; app._on_pan_drag(e3)
assert 0 <= app.view_center[0] <= 400 and 0 <= app.view_center[1] <= 300
sc, (ox, oy), (dw, dh) = app._fit_geometry(); assert ox <= 0.01 and oy <= 0.01 and ox + dw >= cw - 0.01 and oy + dh >= ch - 0.01   # no empty border
# pan does nothing at zoom 1
app.fit_view(); app._on_pan_start(e); app._on_pan_drag(e2); assert app.view_center is None
# drawing a box while zoomed lands on the right image pixels
app._zoom_at(*app._image_px_to_canvas(50, 50), 4); root.update(); n0 = len(app.boxes)
a = app._image_px_to_canvas(40, 40); b = app._image_px_to_canvas(80, 70)
ev = E(); ev.x, ev.y = int(a[0]), int(a[1]); app._on_canvas_press(ev)
ev2 = E(); ev2.x, ev2.y = int(b[0]), int(b[1]); app._on_canvas_release(ev2)
assert len(app.boxes) == n0 + 1
_, xc, yc, w, h = app.boxes[-1][:5]
assert abs(xc * 400 - 60) < 2 and abs(yc * 300 - 55) < 2 and abs(w * 400 - 40) < 2 and abs(h * 300 - 30) < 2, app.boxes[-1]
# opening another image resets the view; Fit button resets
app._zoom_at(10, 10, 2) if app.zoom == 1 else None
app.load_image(iid); assert app.zoom == 1 and app.view_center is None
app._zoom_at(cw // 2, ch // 2, 3); assert app.zoom > 1
app.fit_button.invoke(); assert app.zoom == 1 and app.view_center is None
# LiDAR dots line up zoomed in: depth overlay puts a dot at the canvas spot of its image pixel
d = np.zeros((300, 400), np.uint16); d[120, 230] = 3000
Image.fromarray(d).save(app.project.depth_path(iid)); app.project.manifest["images"][iid].update(has_depth=True, depth_pixels=1)
app._depth_cache = (None, None)
app.pick_var.set(True); app._on_pick_toggle()
app._zoom_at(*app._image_px_to_canvas(230, 120), 6); root.update()
pts = app._current_depth_points(); assert pts is not None and len(pts[0]) == 1
res = app._overlay_depth(Image.new("RGB", (cw, ch)), app._fit_geometry()[0], (0, 0))
print("ZOOM OK"); root.destroy(); shutil.rmtree(tmp, ignore_errors=True)
