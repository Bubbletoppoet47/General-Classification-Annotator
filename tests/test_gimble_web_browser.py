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
ids = [P.add_uploaded_image(Image.new("RGB", (800, 600), (i * 60, 90, 160))) for i in range(3)]
for i in ids[:2]: P.manifest["images"][i]["session_id"] = "s1"
P.save_manifest()
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
    assert pg.evaluate("S.classes.length") == 2 and "1 / 3" in pg.inner_text("#info") and pg.locator("#pub").count() == 0   # no Publish on the tablet
    # Sessions: the list shows annotated / total per session; picking one limits the page to it
    pg.click("#sess"); pg.wait_for_selector("#panel", state="visible")
    rows = pg.locator("#plist .row").all_inner_texts()
    assert len(rows) == 3 and "All sessions" in rows[0] and "0 / 3" in rows[0] and "s1" in rows[1] and "0 / 2" in rows[1] and "Single images" in rows[2], rows
    pg.locator("#plist .row", has_text="s1").click(); pg.wait_for_function("S.items.length === 2 && S.sess === 's1'")
    assert pg.inner_text("#sess") == "s1" and not pg.is_visible("#panel")
    r = pg.evaluate("(()=>{const r=document.getElementById('cv').getBoundingClientRect();return [r.left,r.top,r.width,r.height]})()")
    cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
    # pen (mouse) drag draws a box; it is saved as a DRAFT ("?"), not confirmed
    pg.mouse.move(cx - 100, cy - 80); pg.mouse.down(); pg.mouse.move(cx, cy, steps=5); pg.mouse.move(cx + 100, cy + 80, steps=5); pg.mouse.up()
    pg.wait_for_function("S.boxes.length === 1 && S.status === 1"); pg.wait_for_timeout(400)
    assert not P.is_annotated(ids[0]) and P.is_ai_pending(ids[0]) and len(P.get_boxes(ids[0])) == 1
    assert "?" in pg.inner_text("#info") and "\u2713" not in pg.inner_text("#info")
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
    # undo removes it (saved as a draft again), then a new box and Delete
    pg.click("#undo"); pg.wait_for_function("S.boxes.length === 0"); pg.wait_for_timeout(400); assert P.get_boxes(ids[0]) == [] and not P.is_annotated(ids[0])
    pg.mouse.move(cx - 50, cy - 50); pg.mouse.down(); pg.mouse.move(cx + 50, cy + 50, steps=5); pg.mouse.up()
    pg.wait_for_function("S.boxes.length === 1"); pg.click("#del"); pg.wait_for_function("S.boxes.length === 0")
    # draw once more and Confirm: now it is annotated (credited to Pat), the check mark shows, and the next image opens
    pg.mouse.move(cx - 60, cy - 60); pg.mouse.down(); pg.mouse.move(cx + 60, cy + 60, steps=5); pg.mouse.up()
    pg.wait_for_function("S.boxes.length === 1"); pg.wait_for_timeout(300)
    pg.click("#conf"); pg.wait_for_function("S.items.length === 1 && S.idx === 0 && S.status === 0"); pg.wait_for_timeout(300)
    assert P.is_annotated(ids[0]) and len(P.get_boxes(ids[0])) == 1 and P.manifest["images"][ids[0]]["edited_by"] == "web:Pat"
    assert "1 / 1" in pg.inner_text("#info")
    # AI draft: "?" (not a check mark) until Confirm; then Confirm and Next
    pg.click("#ai"); pg.wait_for_function("S.boxes.length === 1 && S.status === 1")
    assert P.is_ai_pending(ids[1]) and not P.is_annotated(ids[1]) and "?" in pg.inner_text("#info") and "\u2713" not in pg.inner_text("#info")
    pg.click("#conf"); pg.wait_for_function("S.items.length === 0"); pg.wait_for_timeout(300)
    assert P.is_annotated(ids[1]) and not P.is_ai_pending(ids[1]) and "All done" in pg.inner_text("#info")
    # filters inside the session: Done shows both, All shows both; AI on a confirmed image is skipped
    pg.click("#filter"); pg.wait_for_function("S.filter === 'done' && S.items.length === 2")
    assert pg.inner_text("#filter").startswith("Done") and "\u2713" in pg.inner_text("#info")
    pg.click("#ai"); pg.wait_for_function("document.getElementById('toast').textContent.includes('skipped')")
    pg.click("#filter"); pg.wait_for_function("S.filter === 'all' && S.items.length === 2")
    pg.click("#sess"); pg.wait_for_selector("#panel", state="visible")
    rows = pg.locator("#plist .row").all_inner_texts(); assert "2 / 2" in rows[1] and "2 / 3" in rows[0], rows
    br.close()
w.stop()
errors = [e for e in errors if "401" not in e]      # (the first "are you logged in?" check)
assert not errors, errors
print("WEB BROWSER OK")
