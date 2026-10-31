"""Drives the real page in a headless browser: log in, draw with the pen, pan with a finger, undo, delete, confirm."""
import os, sys, tempfile, json
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("SKIPPED (needs playwright)"); sys.exit(0)
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import annotator_core as ac, web_mode as wm

tmp = tempfile.mkdtemp()
P = ac.Project(os.path.join(tmp, "p"), default_classes=["person", "bag"])
ids = [P.add_uploaded_image(Image.new("RGB", (800, 600), (i * 60, 90, 160))) for i in range(2)]
w = wm.WebShare(P, predict=lambda i: [(1, 0.5, 0.5, 0.3, 0.3)], port=8801, code="654321"); w.start()
base = f"http://127.0.0.1:{w.port}/"
errors = []
with sync_playwright() as pw:
    try:
        br = pw.chromium.launch()
    except Exception as e:
        print("SKIPPED (no browser:", str(e)[:60], ")"); w.stop(); sys.exit(0)
    pg = br.new_page(viewport={"width": 1180, "height": 820})
    pg.on("pageerror", lambda e: errors.append(str(e))); pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.goto(base + "?c=654321")
    assert pg.input_value("#code") == "654321"                                  # the QR link fills the code in
    pg.fill("#name", "Pat"); pg.click("#go")
    pg.wait_for_selector("#app", state="visible"); pg.wait_for_function("S.img !== null")
    assert pg.evaluate("S.classes.length") == 2 and "1 / 2" in pg.inner_text("#info")
    r = pg.evaluate("(()=>{const r=document.getElementById('cv').getBoundingClientRect();return [r.left,r.top,r.width,r.height]})()")
    cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
    # pen (mouse) drag draws a box and saves it
    pg.mouse.move(cx - 100, cy - 80); pg.mouse.down(); pg.mouse.move(cx, cy, steps=5); pg.mouse.move(cx + 100, cy + 80, steps=5); pg.mouse.up()
    pg.wait_for_function("S.boxes.length === 1"); pg.wait_for_timeout(500)
    assert P.is_annotated(ids[0]) and len(P.get_boxes(ids[0])) == 1 and P.manifest["images"][ids[0]]["edited_by"] == "web:Pat"
    # a finger drag pans, never draws
    ox = pg.evaluate("S.ox")
    pg.evaluate("""(()=>{const c=document.getElementById('c'),r=c.getBoundingClientRect(),m=(t,x,y)=>c.dispatchEvent(new PointerEvent(t,{pointerId:7,pointerType:'touch',clientX:x,clientY:y,bubbles:true}));
      m('pointerdown',r.left+200,r.top+200);m('pointermove',r.left+260,r.top+230);m('pointerup',r.left+260,r.top+230)})()""")
    assert pg.evaluate("S.ox") - ox == 60 and pg.evaluate("S.boxes.length") == 1
    # two-finger pinch zooms
    s0 = pg.evaluate("S.s")
    pg.evaluate("""(()=>{const c=document.getElementById('c'),r=c.getBoundingClientRect(),m=(t,id,x,y)=>c.dispatchEvent(new PointerEvent(t,{pointerId:id,pointerType:'touch',clientX:x,clientY:y,bubbles:true}));
      m('pointerdown',1,r.left+300,r.top+300);m('pointerdown',2,r.left+400,r.top+300);m('pointermove',1,r.left+250,r.top+300);m('pointermove',2,r.left+450,r.top+300);m('pointermove',1,r.left+200,r.top+300);m('pointermove',2,r.left+500,r.top+300);
      m('pointerup',1,r.left+250,r.top+300);m('pointerup',2,r.left+450,r.top+300)})()""")
    assert pg.evaluate("S.s") > s0 * 1.3 and pg.evaluate("S.boxes.length") == 1
    pg.click("#fit")
    # undo removes it (and saves), a new box then delete
    pg.click("#undo"); pg.wait_for_function("S.boxes.length === 0"); pg.wait_for_timeout(500); assert P.get_boxes(ids[0]) == []
    pg.mouse.move(cx - 50, cy - 50); pg.mouse.down(); pg.mouse.move(cx + 50, cy + 50, steps=5); pg.mouse.up()
    pg.wait_for_function("S.boxes.length === 1"); pg.click("#del"); pg.wait_for_function("S.boxes.length === 0")
    # AI on a confirmed image is skipped; on a new one it fills a draft, and Confirm saves it
    pg.click("#ai"); pg.wait_for_function("document.getElementById('toast').textContent.includes('skipped')")
    pg.click("#conf"); pg.wait_for_function("S.items.length === 1"); pg.wait_for_timeout(300)       # (To do list: image 1 left)
    assert "1 / 1" in pg.inner_text("#info")
    pg.click("#ai"); pg.wait_for_function("S.boxes.length === 1"); assert P.is_ai_pending(ids[1]) and not P.is_annotated(ids[1])
    pg.click("#conf"); pg.wait_for_function("S.items.length === 0"); pg.wait_for_timeout(300)
    assert P.is_annotated(ids[1]) and len(P.get_boxes(ids[1])) == 1 and "All done" in pg.inner_text("#info")
    br.close()
w.stop()
errors = [e for e in errors if "401" not in e]      # (the first "are you logged in?" check)
assert not errors, errors
print("WEB BROWSER OK")
