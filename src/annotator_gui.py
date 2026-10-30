"""
annotator_gui.py

Desktop bounding-box annotation tool for building a trash-detection
dataset: draw boxes on photos from the Gimble rig (camera + LiDAR, see
gimble_pi.py), import a recorded session folder (each sample brings its
LiDAR depth along, viewable as an overlay on the photo) or a video, and
PUBLISH what you annotated into your own folder inside a shared folder. Classes are
fixed and bound to number-key hotkeys (Edit > Classes, Edit > Hotkeys).

This program only annotates and publishes. Training (collecting everyone's
folders, validation sets, augmentation, the Train window) lives in the
separate training program: trainer_gui.py / trainer_cli.py.

Menus: File (imports, Publish, Import data to edit, shared folder, Exit),
Edit (classes, hotkeys, name, AI confidence, theme), Manage (sessions, sync).

All the actual logic (box math, file formats, the project's on-disk
state, Gimble session import, distortion, export/splitting) lives in
annotator_core.py and is unit-tested there - this file is deliberately
just the visual layer: it draws things and turns clicks into calls on a
well-tested Project object.

Run with:
    python annotator_gui.py [project_folder]

If no folder is given, it defaults to ./Dataset next to this
script (created if it doesn't exist yet) - a fresh project, separate
from the older video-based TrashDetection project folder, which is left
untouched.
"""

import io
import json
import math
import os
import sys
import threading
import time
from tkinter import (
    Tk, Toplevel, Canvas, Listbox, StringVar, BooleanVar,
    filedialog, messagebox, simpledialog, SINGLE, END, BOTH, LEFT, RIGHT,
    TOP, BOTTOM, X, Y, NW, Menu, PhotoImage,
)
from tkinter import ttk
from tkinter.ttk import Treeview

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import annotator_core as ac
import model_registry as mr
import model_library as ml
import app_update as au
import hotkeys as hk
import share_zip as sz

try:
    import sv_ttk
    HAVE_SV_TTK = True
except ImportError:
    HAVE_SV_TTK = False


CANVAS_BG = "#1e1e1e"
DEPTH_NEAR_M, DEPTH_FAR_M = 0.5, 6.0     # colour scale of the LiDAR depth overlay
ZOOM_MAX = 8.0                          # scroll-wheel zoom limit (1 = whole image)
ZOOM_STEP = 1.25                        # one wheel notch
PICK_RADIUS_PX = 40                      # how far (screen px) from the cursor a LiDAR point can be picked
COPY_LIDAR_ON_IMPORT = True              # also copy each sample's raw LiDAR points into the project
BOX_COLORS = [
    "#e74c3c", "#2ecc71", "#3498db", "#f1c40f", "#9b59b6",
    "#1abc9c", "#e67e22", "#95a5a6", "#fd79a8", "#00b894",
]


def color_for_class(class_id):
    return BOX_COLORS[class_id % len(BOX_COLORS)]


# AI Assist always loads this ONE fixed file, never anything from a
# training run's own (possibly still-being-written) folder. It's
# maintained separately by update_model_backup.py, which you run by
# hand whenever a training run finishes and you want its result to
# become what this tool (and webcam_test.py, same convention) load -
# see that script for how it decides what counts as "the" checkpoint
# and the validation it does before replacing this file. Nothing here
# ever auto-detects "the latest" training run on its own anymore, on
# purpose - that's exactly how an old/incompatible checkpoint (or one
# from a crashed run) could end up loaded without anyone deciding that
# was actually what they wanted.
BACKUP_PATH = os.path.join(os.path.expanduser("~"), "yolo_runs", "backup", "current_best.pt")
BACKUP_META_PATH = os.path.join(os.path.dirname(BACKUP_PATH), "current_best.meta.json")
# The from-scratch model (train_yolo.py). The annotator only READS it; training rewrites it every round and
# the annotator reloads it when it changes. The original helper above is never written by this tool.
RUNS_DIR = os.path.join(os.path.expanduser("~"), "yolo_runs")
# The AI model box: these two, then one entry per trained model slot ("Medium helper", "Nano scratch", ...) found
# under RUNS_DIR by model_registry.discover() - re-checked every time the box is opened.
AI_CHOICES = ["No AI model", "Original helper model"]
# Models you insert from the shared Models folder (File > Insert Assist/Scratch Model) are copied here, on this computer,
# so the annotator does not depend on Drive being synced while you work: <INSERTED_DIR>/Assist and /Scratch.
INSERTED_DIR = os.path.join(RUNS_DIR, "inserted")


def _describe_backup():
    """Short human-readable line about the current backup checkpoint,
    read from the sidecar update_model_backup.py writes alongside it -
    e.g. "run trash_detection, epoch 58/73, mAP50 0.773". Falls back to
    something generic if the sidecar's missing or unreadable, which is
    fine - it's just for display, never anything load-bearing."""
    try:
        with open(BACKUP_META_PATH) as f:
            meta = json.load(f)
        parts = []
        if "run" in meta:
            parts.append(f"run {meta['run']}")
        if "epoch" in meta and "total_epochs" in meta:
            parts.append(f"epoch {meta['epoch']}/{meta['total_epochs']}")
        if "map50" in meta:
            parts.append(f"mAP50 {meta['map50']}")
        return ", ".join(parts) if parts else "backup checkpoint"
    except Exception:
        return "backup checkpoint"


class AnnotatorApp:
    def __init__(self, root, project_root):
        self.root = root
        self.project = ac.Project(project_root)

        self.current_image_id = None
        self.current_pil_image = None
        self.tk_image = None
        self.boxes = []              # list of (class_id, xc, yc, w, h) for the current image
        self.box_dist = []           # parallel to boxes: None or {distance_m, u, v} (picked LiDAR point)
        self.active_session = None   # None = all sets; else the session id whose frames are listed
        self._session_keys = [None]  # parallel to the set picker's entries
        self._bg_job = False         # an import / export / refresh is running in the background
        self._sessions_refresh = None   # callback that redraws the Manage Sessions table, while it's open
        self.selected_box_index = None
        self.draw_start = None       # (canvas_x, canvas_y) while dragging a new box
        self.temp_rect_id = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.zoom = 1.0              # 1 = whole image fitted to the window; scroll wheel zooms in (up to ZOOM_MAX)
        self.view_center = None      # image pixel shown in the middle of the window when zoomed in
        self._pan_last = None
        self.current_class_id = 0
        self.model_assist = None        # loaded ModelAssist / GimbleAssist instance, or None until picked in the AI model box
        self._ai_kind = None
        self.model_conf = 0.25
        self.ai_class_vars = {}         # class_id -> BooleanVar, built once in _build_ui
        self._predict_busy = False      # a live prediction is running in the background
        self._predict_queued_img_id = None   # most recent request to run once the current one finishes
        self._depth_cache = (None, None)     # (img_id, (u, v, z) arrays or None) for the overlay

        self.hotkeys = hk.Hotkeys(self.project.load_hotkeys())
        self.cfg = sz.load_config()                  # per computer: your name + id, where the shared folder is
        self.me = sz.identity(self.cfg)              # None until you have told the app your name (asked on first share)
        self.access_people = set()                   # other people's data you have opened with Access this run
        if self.me:
            self.project.editor_id = self.me['id']
        self._bound_keys = set()
        self._inserted = {}                           # toolbar label -> entry for models inserted from the Models folder
        self._model_submenus = []
        self._detecting_models = False
        self._scan_inserted()
        self._build_ui()
        self._bind_hotkeys()
        self._refresh_class_list()
        self._refresh_image_list()
        self._update_status()

        if self.project.startup_reconciled:
            n = len(self.project.startup_reconciled)
            messagebox.showinfo(
                "Cleaned up deleted files",
                f"Noticed {n} image(s) that were deleted outside the app since it was "
                f"last open, and cleaned up their tracking (labels, distortions) to match."
            )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _set_window_icon(self):
        """Use the logo as the window icon. Silently does nothing if the picture is missing."""
        try:
            import sys
            base = getattr(sys, "_MEIPASS", None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            path = os.path.join(base, "assets", "icon_256.png")
            if os.path.exists(path):
                self._icon_img = PhotoImage(file=path)
                self.root.iconphoto(True, self._icon_img)
        except Exception:
            pass

    def _build_ui(self):
        self.root.title("Trash Detection Annotator")
        self._set_window_icon()
        self.root.geometry("1280x820")
        self.root.minsize(900, 600)

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(side=TOP, fill=X)
        self._build_menubar()
        # One switch: LiDAR dots on the photo AND distance picking (hover a box -> click a dot).
        # show_depth_var always mirrors pick_var; the two used to be separate toggles.
        self.show_depth_var = BooleanVar(value=False)
        self.pick_var = BooleanVar(value=False)
        self.pick_check = ttk.Checkbutton(toolbar, variable=self.pick_var, command=self._on_pick_toggle)
        self.pick_check.pack(side=LEFT, padx=(4, 4))
        self._update_pick_check_text()
        self.fit_button = ttk.Button(toolbar, command=self.fit_view)
        self.fit_button.pack(side=LEFT, padx=(8, 4))
        self._update_fit_button_text()
        ttk.Separator(toolbar, orient="vertical").pack(side=LEFT, fill=Y, padx=8)
        ttk.Label(toolbar, text="AI model:").pack(side=LEFT, padx=(4, 2))
        self.ai_choice_var = StringVar(value=AI_CHOICES[0])
        self._ai_entries = {}
        self.ai_choice_box = ttk.Combobox(toolbar, textvariable=self.ai_choice_var, values=AI_CHOICES,
                                          state="readonly", width=24, postcommand=self._refresh_ai_choices)
        self.ai_choice_box.pack(side=LEFT, padx=2)
        self.ai_choice_box.bind("<<ComboboxSelected>>", lambda e: self.load_ai_model())
        self._refresh_ai_choices()
        self.root.after(300, self._identify_old_models_in_background)
        ttk.Button(toolbar, text="Confirm Image", command=self.confirm_current_as_is).pack(side=LEFT, padx=4)
        self.ai_model_status_var = StringVar(value="No model loaded")
        ttk.Label(toolbar, textvariable=self.ai_model_status_var).pack(side=LEFT, padx=(8, 0))

        body = ttk.Frame(self.root)
        body.pack(side=TOP, fill=BOTH, expand=True)

        # --- left sidebar: classes ---
        # Scrollable, not a plain fixed frame: this column has grown every
        # time a feature landed here (AI Assist, Clear & Retry, Selected
        # box) and a plain frame silently clips whatever doesn't fit the
        # window instead of erroring - which is exactly what happened the
        # first time (the Selected box section scrolled off the bottom,
        # unreachable, with no on-screen sign anything was wrong).
        # A canvas+scrollbar means new controls added here in the future
        # are always reachable no matter how tall this column gets, rather
        # than silently vanishing again.
        left_outer = ttk.Frame(body, width=220)
        left_outer.pack(side=LEFT, fill=Y)
        left_outer.pack_propagate(False)
        self.left_canvas = left_canvas = Canvas(left_outer, highlightthickness=0, bd=0)
        left_scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        left_scrollbar.pack(side=RIGHT, fill=Y)

        left = ttk.Frame(left_canvas, padding=10)
        left_window = left_canvas.create_window((0, 0), window=left, anchor=NW)

        def _sync_left_scrollregion(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        left.bind("<Configure>", _sync_left_scrollregion)

        def _sync_left_width(event):
            left_canvas.itemconfig(left_window, width=event.width)
        left_canvas.bind("<Configure>", _sync_left_width)

        # mousewheel only scrolls this column while the cursor is actually
        # over it (bound/unbound on enter/leave) - a global bind_all would
        # otherwise hijack scrolling everywhere else in the window (e.g.
        # the image list on the right)
        def _on_left_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        left_canvas.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel))
        left_canvas.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))

        ttk.Label(left, text="Classes", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.class_listbox = Listbox(left, selectmode=SINGLE, exportselection=False,
                                      activestyle="none", highlightthickness=0, bd=0,
                                      font=("Consolas", 10))
        self.class_listbox.pack(fill=BOTH, expand=True, pady=(6, 6))
        self.class_listbox.bind("<<ListboxSelect>>", self._on_class_select)
        ttk.Label(left, text="Press a class's key (shown in [ ]) to set\n"
                              "the active class - if a box is selected,\n"
                              "it relabels that box too. Add, rename or\n"
                              "delete classes with Edit > Classes; change\n"
                              "any key with Edit > Hotkeys (both saved\n"
                              "with the project).",
                  wraplength=200, justify=LEFT).pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill=X, pady=10)
        ttk.Label(left, text="AI Assist", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(left, text="Check classes to get live model\n"
                              "predictions on the CURRENT photo only.\n"
                              "Toggling re-evaluates it instantly.\n"
                              "Never touches confirmed images or\n"
                              "boxes you've drawn yourself.",
                  wraplength=200, justify=LEFT).pack(anchor="w", pady=(2, 6))
        self.ai_box = ttk.Frame(left)
        self.ai_box.pack(fill=X, anchor="w")
        self._build_ai_assist_checkboxes(self.ai_box)
        ttk.Button(left, text="Clear Boxes & Retry AI",
                   command=self.clear_and_retry_ai).pack(fill=X, pady=(8, 0))
        ttk.Label(left, text="Wipes this image's boxes - confirmed or\n"
                              "still a draft - and re-predicts it with\n"
                              "whatever's loaded/checked above. Handy\n"
                              "once you've trained a better checkpoint\n"
                              "and want an older annotation reconsidered.",
                  wraplength=200, justify=LEFT).pack(anchor="w", pady=(4, 0))

        ttk.Separator(left, orient="horizontal").pack(fill=X, pady=10)
        ttk.Label(left, text="Selected box", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Button(left, text="Delete Selected Box", command=self.delete_selected_box).pack(fill=X, pady=(6, 0))
        ttk.Button(left, text="Clear Distance of Box", command=self.clear_selected_distance).pack(fill=X, pady=(6, 0))
        ttk.Label(left, text="LiDAR dots + pick distance: one switch in the toolbar "
                              "shows the dots and turns on picking. Hover a box - the nearest LiDAR point inside "
                              "it lights up with its distance - and click to "
                              "attach that distance to the box. Right-click "
                              "clears it. Drag still draws boxes.",
                  wraplength=200, justify=LEFT).pack(anchor="w", pady=(6, 0))
        ttk.Label(left, text="Drag: draw a new box, even on top of\n"
                              "existing ones - crowded scenes never\n"
                              "block you. Click: select a box (for\n"
                              "relabeling with the class keys, or this\n"
                              "button/the delete key). Double-click or\n"
                              "right-click a box: delete it instantly.",
                  wraplength=200, justify=LEFT).pack(anchor="w", pady=(10, 0))

        # --- center: canvas ---
        center = ttk.Frame(body, padding=10)
        center.pack(side=LEFT, fill=BOTH, expand=True)
        self.canvas = Canvas(center, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<MouseWheel>", self._on_canvas_wheel)               # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1))   # Linux
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, -1))
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)                # hold the wheel and drag to move around
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", lambda e: setattr(self, '_pan_last', None))
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", lambda e: self.canvas.delete("hover"))
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        # --- right sidebar: image navigator ---
        right = ttk.Frame(body, padding=10, width=260)
        right.pack(side=RIGHT, fill=Y)
        right.pack_propagate(False)
        ttk.Label(right, text="Images", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(right, text="Shows images still needing annotation.\n"
                              "Annotated ones drop off once you leave them.\n"
                              "[?] = AI pre-labeled, not yet reviewed by you -\n"
                              "boxes show dashed until you fix them or click\n"
                              "Confirm Image. On disk: images/unannotated/ vs\n"
                              "images/annotated/ - safe to delete from\n"
                              "unannotated/ by hand, it's never used in\n"
                              "training. Deleted something? Hit Sync Deleted\n"
                              "Files (or just reopen the project). Each Gimble\n"
                              "sample's LiDAR depth lives in depth/ and follows\n"
                              "the image through rotate/delete.",
                  wraplength=240, justify=LEFT).pack(anchor="w", pady=(2, 6))

        ttk.Label(right, text="Working set", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.session_var = StringVar(value="")
        self.session_combo = ttk.Combobox(right, textvariable=self.session_var, state="readonly")
        self.session_combo.pack(fill=X, pady=(2, 2))
        self.session_combo.bind("<<ComboboxSelected>>", self._on_session_pick)
        ttk.Label(right, text="[A] = every frame of the set is annotated.",
                  wraplength=240, justify=LEFT).pack(anchor="w", pady=(0, 6))

        self.show_annotated_var = BooleanVar(value=False)
        ttk.Checkbutton(right, text="Show annotated too (review/fix)",
                         variable=self.show_annotated_var,
                         command=self._refresh_image_list).pack(anchor="w", pady=(0, 6))
        nav_row = ttk.Frame(right)
        nav_row.pack(fill=X, pady=(4, 6))
        ttk.Button(nav_row, text="< Prev", command=self.prev_image).pack(side=LEFT, expand=True, fill=X)
        ttk.Button(nav_row, text="Next >", command=self.next_image).pack(side=LEFT, expand=True, fill=X)
        ttk.Button(right, text="Rotate 90° ⟳", command=self.rotate_current_frame).pack(fill=X, pady=(0, 6))
        ttk.Button(right, text="Delete Frame...", command=self.delete_current_frame).pack(fill=X, pady=(0, 6))
        self.image_listbox = Listbox(right, selectmode=SINGLE, exportselection=False,
                                      activestyle="none", highlightthickness=0, bd=0,
                                      font=("Consolas", 10))
        self.image_listbox.pack(fill=BOTH, expand=True)
        self.image_listbox.bind("<<ListboxSelect>>", self._on_image_select)

        # --- status bar ---
        self.status_var = StringVar(value="No project loaded")
        status_bar = ttk.Frame(self.root, padding=(10, 4))
        status_bar.pack(side=BOTTOM, fill=X)
        ttk.Label(status_bar, textvariable=self.status_var).pack(side=LEFT)
        self.progress_var = StringVar(value="")
        ttk.Label(status_bar, textvariable=self.progress_var).pack(side=RIGHT)

    def _build_menubar(self):
        bar = Menu(self.root, tearoff=0)
        file_m = Menu(bar, tearoff=0)
        file_m.add_command(label="Import Gimble Session...", command=self.import_gimble_session)
        file_m.add_command(label="Import Video...", command=self.import_video)
        file_m.add_separator()
        self.assist_menu = Menu(file_m, tearoff=0)
        self.scratch_menu = Menu(file_m, tearoff=0)
        file_m.add_cascade(label="Insert Assist Model", menu=self.assist_menu)
        file_m.add_cascade(label="Insert Scratch Model", menu=self.scratch_menu)
        file_m.configure(postcommand=self._rebuild_model_menus)           # always shows what is in the Models folder right now
        file_m.add_separator()
        file_m.add_command(label="Publish to shared folder", command=self.publish_to_shared)
        file_m.add_command(label="Import data to edit...", command=self.import_to_edit)
        file_m.add_command(label="Shared folder location...", command=self.choose_shared_dir)
        file_m.add_command(label="Models folder location...", command=self.choose_models_dir)
        file_m.add_separator()
        file_m.add_command(label="Check for updates...", command=self.check_for_updates)
        if not au.is_frozen():
            file_m.add_command(label="Update folder location...", command=self.choose_update_dir)
        file_m.add_separator()
        file_m.add_command(label="Exit", command=self.root.destroy)
        edit_m = Menu(bar, tearoff=0)
        edit_m.add_command(label="Classes...", command=self.manage_classes)
        edit_m.add_command(label="Hotkeys...", command=self.edit_hotkeys)
        edit_m.add_separator()
        edit_m.add_command(label="My name...", command=self.change_my_name)
        edit_m.add_command(label="AI confidence...", command=self.set_ai_confidence)
        edit_m.add_separator()
        edit_m.add_command(label="Toggle theme", command=self._toggle_theme)
        manage_m = Menu(bar, tearoff=0)
        manage_m.add_command(label="Sessions...", command=self.manage_sessions)
        manage_m.add_command(label="Sync Deleted Files", command=self.sync_deleted_files)
        manage_m.add_separator()
        manage_m.add_command(label="Dataset location...", command=self.choose_dataset)
        bar.add_cascade(label="File", menu=file_m)
        bar.add_cascade(label="Edit", menu=edit_m)
        bar.add_cascade(label="Manage", menu=manage_m)
        self.root.config(menu=bar)
        self.menubar, self.file_menu, self.edit_menu, self.manage_menu = bar, file_m, edit_m, manage_m

    # ------------------------------------------------------------------
    # The shared Models folder: File > Insert Assist Model / Insert Scratch Model
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Updates: File > Check for updates (copies newer program files from Annotator_Share, then relaunches)
    # ------------------------------------------------------------------

    APP_DIR = au.app_dir(__file__)

    def _update_dir(self):
        d = self.cfg.get('update_dir')
        if d and os.path.isdir(d):
            return d
        guess = os.path.join(self.APP_DIR, "Annotator_Share")           # the usual place when it sits beside the program
        return guess if os.path.isdir(guess) else None

    def choose_update_dir(self):
        """Where the Annotator_Share folder is. Remembered on this computer."""
        d = filedialog.askdirectory(title="Pick the Annotator_Share folder (the one with annotator_gui.py inside)",
                                    initialdir=self.cfg.get('update_dir') or None)
        if not d:
            return None
        self.cfg['update_dir'] = os.path.normpath(d)
        self._save_cfg()
        return self.cfg['update_dir']

    def check_for_updates(self):
        if au.is_frozen():
            self._check_github_updates()
            return
        if self._bg_blocked():
            return
        share = self._update_dir() or self.choose_update_dir()
        if not share:
            return
        res = au.check(share, self.APP_DIR)
        if not res['ok']:
            messagebox.showinfo("Check for updates", res['error'])
            return
        names = res['changed'] + res['new']
        if not [n for n in names if n != "UPDATES.txt"]:
            messagebox.showinfo("Check for updates", "You are up to date.")
            return
        text = "These files will be updated:\n  " + "\n  ".join(names)
        if res['note']:
            text += "\n\nWhat's new:\n" + res['note'][:1500]
        if res['older']:
            text += "\n\nCareful: the update folder looks OLDER than your copy, so this would go back a version."
        text += ("\n\nYour old files are kept in the _previous folder. Your projects, sessions and settings are not "
                 "touched.\n\nUpdate now? The annotator will close")
        text += " (you restart it yourself after installing packages)." if res['requirements'] else " and reopen."
        if not messagebox.askyesno("Update available", text):
            return
        try:
            au.apply(share, self.APP_DIR, names)
        except Exception as e:
            messagebox.showerror("Couldn't update", f"{e}\n\nNothing was changed.")
            return
        self._finish_update(res['requirements'])

    def _check_github_updates(self):
        """Packaged program: ask GitHub for the newest release in the background, then offer the download page.
        (A running .exe can't replace itself, so the new version is downloaded and run from the page.)"""
        self.root.config(cursor="watch")
        box = {}

        def worker():
            box['res'] = au.github_check()

        def poll():
            if 'res' not in box:
                self.root.after(100, poll)
                return
            self.root.config(cursor="")
            self._show_github_result(box['res'])

        threading.Thread(target=worker, daemon=True).start()
        poll()

    def _show_github_result(self, res):
        if not res['ok']:
            messagebox.showinfo("Check for updates", res['error'])
            return
        if not res['newer']:
            messagebox.showinfo("Check for updates", f"You are up to date (version {res['current']}).")
            return
        text = f"Version {res['latest'].lstrip('vV')} is available. You have {res['current']}."
        if res['notes']:
            text += "\n\nWhat's new:\n" + res['notes'][:1500]
        text += ("\n\nOpen the download page? Download the new file, close this program, and run the new one. "
                 "Your project, sessions and settings are not touched.")
        if messagebox.askyesno("Update available", text):
            import webbrowser
            webbrowser.open(res['url'])

    def _finish_update(self, requirements):
        if requirements:
            messagebox.showinfo("Updated", "Updated. This update changes the packages the annotator needs.\n\n"
                                "Open a console in the annotator folder and run:\n    pip install -r requirements.txt\n\n"
                                "then start the annotator again.")
            self.root.destroy()
            return
        self._relaunch(self.project.root)

    def choose_dataset(self):
        """Manage > Dataset location: open a different project folder (e.g. one that was renamed or moved).
        Remembered for this copy of the program; the annotator then reopens on it."""
        if self._bg_blocked():
            return
        d = filedialog.askdirectory(title="Pick your dataset folder (the one with manifest.json and images inside)",
                                    initialdir=os.path.dirname(self.project.root))
        if not d:
            return
        d = os.path.normpath(d)
        if _same(d, self.project.root):
            messagebox.showinfo("Dataset location", "That is the dataset already open.")
            return
        if not os.path.exists(os.path.join(d, "manifest.json")):
            if not messagebox.askyesno("Dataset location",
                                       f"There is no annotator dataset in:\n{d}\n\n"
                                       "(no manifest.json). Start a NEW empty dataset there?"):
                return
        if not messagebox.askyesno("Dataset location",
                                   f"Open this dataset?\n{d}\n\nThe annotator will close and reopen on it. "
                                   "The one that is open now is not changed."):
            return
        os.makedirs(d, exist_ok=True)
        key = os.path.normcase(os.path.abspath(self.APP_DIR))
        self.cfg.setdefault('projects', {})[key] = d
        self._save_cfg()
        self._relaunch(d)

    def _relaunch(self, project_root):
        import subprocess
        try:
            if au.is_frozen():
                subprocess.Popen([sys.executable, project_root], cwd=self.APP_DIR,
                                 env=dict(os.environ, PYINSTALLER_RESET_ENVIRONMENT="1"))
            else:
                subprocess.Popen([sys.executable, os.path.join(self.APP_DIR, "annotator_gui.py"), project_root],
                                 cwd=self.APP_DIR)
        except OSError as e:
            messagebox.showinfo("Reopen", f"It couldn't reopen by itself ({e}). Please start it again.")
        self.root.destroy()

    def choose_models_dir(self):
        """Where the Models folder is (it holds Assist and Scratch; you make those folders yourself). Remembered."""
        d = filedialog.askdirectory(title="Pick the Models folder (the one with Assist and Scratch inside)",
                                    initialdir=self.cfg.get('models_dir') or None)
        if not d:
            return None
        self.cfg['models_dir'] = os.path.normpath(d)
        self._save_cfg()
        return self.cfg['models_dir']

    def _scan_inserted(self):
        """Models inserted in earlier runs are still in INSERTED_DIR - keep them in the toolbar box."""
        self._inserted = {}
        for key in ml.FOLDERS:
            d = os.path.join(INSERTED_DIR, ml.FOLDERS[key])
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                pt = os.path.join(d, f)
                info = ml.read_info(pt) if f.lower().endswith('.pt') else None
                if info and info.get('size'):
                    v = {'path': pt, 'file': f, 'key': key, 'size': info['size'], 'ts': info.get('created') or os.path.getmtime(pt)}
                    self._inserted[ml.short_label(v)] = self._entry_for(v, pt)

    @staticmethod
    def _entry_for(v, local_path):
        return {'kind': 'scratch' if v['key'] == 'scratch' else 'helper', 'size': v['size'], 'label': ml.short_label(v),
                'dir': os.path.dirname(local_path), 'path': local_path}

    def _rebuild_model_menus(self):
        """Fills both nested lists from the Models folder as it is right now (runs every time File is opened)."""
        for m in self._model_submenus:
            try:
                m.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._model_submenus = []
        for key, menu in (('assist', self.assist_menu), ('scratch', self.scratch_menu)):
            menu.delete(0, 'end')
            models_dir = self.cfg.get('models_dir')
            msg = ml.missing_message(models_dir, key)
            if msg:
                menu.add_command(label=("Set the Models folder first (File > Models folder location...)" if not models_dir else
                                        f"Missing folder: {ml.folder_path(models_dir, key)}"), state='disabled')
                continue
            versions = ml.list_versions(models_dir, key)
            unknown = [v for v in versions if not v['size']]
            if unknown:
                self._detect_pasted_models_in_background()
            news = ml.newest_per_size(versions)
            if not news and not unknown:
                menu.add_command(label=f"No models in {ml.folder_path(models_dir, key)} yet - paste one in", state='disabled')
            for v in news:
                menu.add_command(label="Newest:  " + ml.describe(v), command=lambda v=v: self.insert_model(v))
            olds = ml.older_per_size(versions)
            if olds:
                menu.add_separator()
                for size in ml.SIZE_ORDER:
                    if size in olds:
                        sub = Menu(menu, tearoff=0)
                        self._model_submenus.append(sub)
                        for v in olds[size]:
                            sub.add_command(label=ml.describe(v), command=lambda v=v: self.insert_model(v))
                        menu.add_cascade(label=f"Older {mr.SIZE_NAMES[size]} versions", menu=sub)
            if unknown:
                menu.add_separator()
                menu.add_command(label=f"Reading the size of {len(unknown)} pasted model(s) - open this menu again in a moment",
                                 state='disabled')

    def _detect_pasted_models_in_background(self):
        """A model you pasted by hand has no info file yet: read its size once (loads torch) and save the info beside it."""
        if self._detecting_models:
            return
        self._detecting_models = True
        models_dir = self.cfg.get('models_dir')

        def worker():
            try:
                for key in ml.FOLDERS:
                    ml.list_versions(models_dir, key, detect=True)
            except Exception:  # noqa: BLE001
                pass
            self.root.after(0, lambda: setattr(self, '_detecting_models', False))
        threading.Thread(target=worker, daemon=True).start()

    def insert_model(self, v):
        """Copies the chosen version to this computer, adds it to the toolbar's AI model box and loads it."""
        if self._bg_blocked():
            return
        self._bg_job = True
        self.progress_var.set(f"Copying {ml.short_label(v)}...")

        def worker():
            try:
                local = ml.local_copy(v, INSERTED_DIR)
                err = None
            except Exception as e:  # noqa: BLE001
                local, err = None, str(e)
            self.root.after(0, lambda: self._insert_done(v, local, err))
        threading.Thread(target=worker, daemon=True).start()

    def _insert_done(self, v, local, err):
        self._bg_job = False
        self.progress_var.set("")
        if err:
            messagebox.showerror("Couldn't insert the model", err)
            return
        entry = self._entry_for(v, local)
        self._inserted[entry['label']] = entry
        self._refresh_ai_choices()
        self.load_ai_model(entry['label'])

    def _announce(self, headline):
        """A short message in the status corner that fades after a few seconds."""
        self.progress_var.set(headline)
        self.root.after(4000, lambda: self.progress_var.set("") if not self._bg_job else None)

    def _toggle_theme(self):
        if not HAVE_SV_TTK:
            return
        current = sv_ttk.get_theme()
        sv_ttk.set_theme("light" if current == "dark" else "dark")

    def _build_ai_assist_checkboxes(self, parent, keep_checked=()):
        """One checkbox per class. Rebuilt whenever the class list changes (see _after_class_change);
        classes named in keep_checked stay checked. All start unchecked otherwise: nothing gets
        AI-touched until you explicitly opt a class in."""
        for w in parent.winfo_children():
            w.destroy()
        self.ai_class_vars = {}
        for i, name in enumerate(self.project.classes):
            var = BooleanVar(value=name in keep_checked)
            self.ai_class_vars[i] = var
            key = self._hotkey_label(i)
            prefix = f"[{key}] " if key else ""
            cb = ttk.Checkbutton(parent, text=f"{prefix}{name}", variable=var,
                                  command=self._on_ai_assist_toggle)
            cb.pack(anchor="w", pady=1)

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def _hotkey_label(self, idx):
        """Shown hotkey for class index idx ('' if it has none). Defaults: 1-9 for the first nine
        classes, 0 for the tenth; change them with the Hotkeys window."""
        return self.hotkeys.label(f"class:{idx}")

    def _hotkey_actions(self):
        return {
            "delete_box": self.delete_selected_box,
            "toggle_pick": self._toggle_pick,
            "toggle_depth": self._toggle_depth,
            "prev_image": self.prev_image,
            "next_image": self.next_image,
            "fit_view": self.fit_view,
        }

    def _bind_hotkeys(self):
        """(Re)binds every shortcut from self.hotkeys; run again whenever a key or the class list changes."""
        for seq in self._bound_keys:
            self.root.unbind(seq)
        self._bound_keys = set()
        fixed = self._hotkey_actions()
        for seq, action in self.hotkeys.bindings(len(self.project.classes)).items():
            if action.startswith("class:"):
                idx = int(action[6:])
                self.root.bind(seq, lambda e, i=idx: self._set_active_class(i))
            else:
                self.root.bind(seq, lambda e, fn=fixed[action]: fn())
            self._bound_keys.add(seq)

    def _pick_keys_text(self):
        keys = [self.hotkeys.label(a) for a in ("toggle_pick", "toggle_depth")]
        keys = [k for k in keys if k]
        return "/".join(keys)

    def _update_pick_check_text(self):
        keys = self._pick_keys_text()
        self.pick_check.configure(text="LiDAR dots + pick distance" + (f" ({keys})" if keys else ""))

    def _update_fit_button_text(self):
        key = self.hotkeys.label("fit_view")
        self.fit_button.configure(text="Fit to screen" + (f" ({key})" if key else ""))

    def _hotkeys_changed(self):
        """Save, rebind and refresh everything that shows a key."""
        try:
            self.project.save_hotkeys(self.hotkeys.to_dict())
        except OSError as e:
            messagebox.showerror("Hotkeys", f"Could not save the hotkeys:\n{e}")
        self._bind_hotkeys()
        self._update_pick_check_text()
        self._update_fit_button_text()
        checked = {self.project.classes[i] for i, v in self.ai_class_vars.items()
                   if v.get() and i < len(self.project.classes)}
        self._build_ai_assist_checkboxes(self.ai_box, keep_checked=checked)
        self._refresh_class_list()
        if getattr(self, '_classes_refresh', None):
            try:
                self._classes_refresh()
            except Exception:  # noqa: BLE001 - the classes window may have just closed
                pass

    def _apply_hotkey(self, action, binding):
        """Gives `action` the key `binding` (None clears it). Returns a one-line note for the Hotkeys window."""
        n = len(self.project.classes)
        lost = self.hotkeys.set(action, binding, n)
        self._hotkeys_changed()
        titles = dict(self.hotkeys.actions(self.project.classes))
        text = f"{titles.get(action, action)}: " + (hk.label(binding) if binding else "no key")
        if lost:
            text += "   (taken away from: " + ", ".join(titles.get(a, a) for a in lost) + ")"
        return text

    def edit_hotkeys(self):
        """Window to change the keyboard shortcuts: pick a row, press 'Change...', press the new key."""
        if getattr(self, '_hotkeys_win', None) is not None:
            try:
                self._hotkeys_win.lift()
                return
            except Exception:  # noqa: BLE001
                self._hotkeys_win = None
        win = self._hotkeys_win = Toplevel(self.root)
        win.title("Hotkeys")
        win.geometry("480x520")
        win.minsize(400, 360)
        win.transient(self.root)

        ttk.Label(win, text="Keyboard shortcuts (saved with the project)",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        tree = Treeview(win, columns=("key",), selectmode="browse", height=14)
        tree.heading("#0", text="Action")
        tree.heading("key", text="Key")
        tree.column("#0", width=300)
        tree.column("key", width=110, anchor="center")
        tree.pack(fill=BOTH, expand=True, padx=12, pady=4)
        note = ttk.Label(win, text="Pick an action, then press Change... and hit the new key (Ctrl / Shift "
                                   "combinations work too). A key can only do one thing - giving one that is "
                                   "already in use takes it from the other action.",
                         wraplength=440, justify=LEFT)
        note.pack(anchor="w", padx=12, pady=(0, 4))

        def refresh(select=None):
            sel = select or (tree.selection()[0] if tree.selection() else None)
            tree.delete(*tree.get_children())
            for action, title in self.hotkeys.actions(self.project.classes):
                tree.insert("", END, iid=action, text=title, values=(self.hotkeys.label(action) or "(none)",))
            if sel and tree.exists(sel):
                tree.selection_set(sel)
                tree.see(sel)
        self._hotkeys_refresh = refresh
        refresh()

        def chosen():
            sel = tree.selection()
            if not sel:
                note.config(text="Pick an action in the list first.")
                return None
            return sel[0]

        def change():
            action = chosen()
            if action is None:
                return
            titles = dict(self.hotkeys.actions(self.project.classes))
            cap = Toplevel(win)
            cap.title("New key")
            cap.geometry("320x120")
            cap.transient(win)
            ttk.Label(cap, text=f"Press the new key for\n{titles[action]}\n(Esc = cancel)",
                      justify="center").pack(expand=True, padx=12, pady=12)

            def on_key(e):
                if e.keysym == "Escape":
                    cap.destroy()
                    return "break"
                b = hk.from_event(e.keysym, e.state)
                if b is None:             # only Shift / Ctrl so far - wait for the real key
                    return "break"
                cap.destroy()
                msg = self._apply_hotkey(action, b)
                refresh(select=action)
                note.config(text=msg)
                return "break"
            cap.bind("<KeyPress>", on_key)
            cap.focus_force()
            try:
                cap.grab_set()
            except Exception:  # noqa: BLE001
                pass
            self._hotkey_capture_win = cap

        def clear():
            action = chosen()
            if action is None:
                return
            note.config(text=self._apply_hotkey(action, None))
            refresh(select=action)

        def reset_one():
            action = chosen()
            if action is None:
                return
            self._apply_hotkey(action, self.hotkeys.default(action))
            note.config(text=f"{dict(self.hotkeys.actions(self.project.classes)).get(action, action)}: "
                             f"back to {self.hotkeys.label(action) or 'no key'}")
            refresh(select=action)

        def reset_all():
            if not messagebox.askyesno("Hotkeys", "Put every shortcut back to its default?", parent=win):
                return
            self.hotkeys.reset_all()
            self._hotkeys_changed()
            refresh()
            note.config(text="All shortcuts are back to their defaults.")

        tree.bind("<Double-Button-1>", lambda e: change())
        tree.bind("<Return>", lambda e: change())
        btns = ttk.Frame(win)
        btns.pack(fill=X, padx=12, pady=12)
        ttk.Button(btns, text="Change...", command=change).pack(side=LEFT, padx=(0, 6))
        ttk.Button(btns, text="Clear", command=clear).pack(side=LEFT, padx=6)
        ttk.Button(btns, text="Default", command=reset_one).pack(side=LEFT, padx=6)
        ttk.Button(btns, text="All defaults", command=reset_all).pack(side=LEFT, padx=6)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=RIGHT)

        def on_close(e):
            if e.widget is win:
                self._hotkeys_win = None
                self._hotkeys_refresh = None
        win.bind("<Destroy>", on_close)

    def _refresh_class_list(self):
        self.class_listbox.delete(0, END)
        for i, name in enumerate(self.project.classes):
            key = self._hotkey_label(i)
            prefix = f"[{key}] " if key else "    "
            self.class_listbox.insert(END, f"{prefix}{name}")
            self.class_listbox.itemconfig(i, fg=color_for_class(i))
        if self.project.classes:
            idx = min(self.current_class_id, len(self.project.classes) - 1)
            self.class_listbox.selection_clear(0, END)
            self.class_listbox.selection_set(idx)
            self.current_class_id = idx

    def _set_active_class(self, idx):
        """Sets the class that new boxes will be drawn with. If a box is
        currently selected on the canvas, also relabels that box in place
        - one keypress covers both "start drawing as X" and "recategorize
        this box as X"."""
        if not (0 <= idx < len(self.project.classes)):
            return
        self.current_class_id = idx
        self.class_listbox.selection_clear(0, END)
        self.class_listbox.selection_set(idx)
        self.class_listbox.see(idx)

        if self.selected_box_index is not None and self.selected_box_index < len(self.boxes):
            _, xc, yc, w, h = self.boxes[self.selected_box_index]
            self.boxes[self.selected_box_index] = (idx, xc, yc, w, h)
            self._save_current_boxes()

        self._redraw()
        self._update_status()

    def _on_class_select(self, event):
        sel = self.class_listbox.curselection()
        if sel:
            self._set_active_class(sel[0])

    def _after_class_change(self, headline, boxes_changed=False, was_checked=()):
        """Everything that shows classes follows the project's list: sidebar list, AI checkboxes
        (checked classes stay checked by NAME), the open image's boxes, and any existing exports."""
        n = len(self.project.classes)
        self.current_class_id = max(0, min(self.current_class_id, n - 1))
        self._bind_hotkeys()
        if getattr(self, '_hotkeys_refresh', None):
            self._hotkeys_refresh()
        self._refresh_class_list()
        self._build_ai_assist_checkboxes(self.ai_box, keep_checked=set(was_checked))
        if boxes_changed:
            self._depth_cache = (None, None)
            if self.current_image_id is not None and self.current_image_id in self.project.manifest['images']:
                self.load_image(self.current_image_id)
            else:
                self.current_image_id = None
                self.current_pil_image = None
                self._set_boxes([])
                self._refresh_image_list()
            self.selected_box_index = None
        self._redraw()
        self._update_status()
        if getattr(self, '_classes_refresh', None):
            self._classes_refresh()
        self._announce(headline)

    def manage_classes(self):
        """Add / rename / delete classes. Saved in the project (classes.txt) so they're still there next
        session. New classes go at the END, so existing class numbers never change; deleting one renumbers
        the later ones everywhere (Project.delete_class) after keeping a backup of the label files."""
        if getattr(self, '_classes_win', None) is not None:
            try:
                self._classes_win.lift()
                return
            except Exception:
                self._classes_win = None
        win = self._classes_win = Toplevel(self.root)
        win.title("Classes")
        win.geometry("420x440")
        win.minsize(360, 320)
        win.transient(self.root)

        ttk.Label(win, text="Classes in this project (saved with the project)",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        lb = Listbox(win, selectmode=SINGLE, exportselection=False, activestyle="none",
                     font=("Consolas", 10), highlightthickness=0)
        lb.pack(fill=BOTH, expand=True, padx=12, pady=4)
        note = ttk.Label(win, text="", justify=LEFT, wraplength=380)
        note.pack(anchor="w", padx=12)
        btns = ttk.Frame(win)
        btns.pack(fill=X, padx=12, pady=12, side=BOTTOM)

        def refresh():
            counts = self.project.class_box_counts()
            sel = lb.curselection()
            lb.delete(0, END)
            for i, name in enumerate(self.project.classes):
                key = self._hotkey_label(i)
                lb.insert(END, f"{('[' + key + ']') if key else '   '} {name:<24} {counts[i]} box(es)")
                lb.itemconfig(i, fg=color_for_class(i))
            if sel and sel[0] < len(self.project.classes):
                lb.selection_set(sel[0])
        self._classes_refresh = refresh
        refresh()

        def chosen():
            sel = lb.curselection()
            if not sel:
                note.config(text="Pick a class in the list first.")
                return None
            note.config(text="")
            return sel[0]

        def checked_names():
            return {self.project.classes[i] for i, v in self.ai_class_vars.items() if v.get() and i < len(self.project.classes)}

        def do_add():
            if self._bg_blocked():
                return
            name = simpledialog.askstring("Add class", "Name for the new class (no spaces - use _ or -):", parent=win)
            if name is None:
                return
            err = self.project.validate_class_name(name, self.project.classes)
            if err:
                messagebox.showerror("Can't add class", err, parent=win)
                return
            was = checked_names()
            self.project.add_class(name.strip())
            self._after_class_change(f"Added class '{name.strip()}'.", was_checked=was)
            lb.selection_clear(0, END)
            lb.selection_set(END)

        def do_rename():
            idx = chosen()
            if idx is None or self._bg_blocked():
                return
            old = self.project.classes[idx]
            new = simpledialog.askstring("Rename class", f"New name for '{old}':", initialvalue=old, parent=win)
            if new is None or new.strip() == old:
                return
            err = self.project.validate_class_name(new, self.project.classes, ignore=old)
            if err:
                messagebox.showerror("Can't rename", err, parent=win)
                return
            was = checked_names()
            self.project.rename_class(idx, new.strip())
            was = {new.strip() if n == old else n for n in was}
            self._after_class_change(
                f"Renamed '{old}' to '{new.strip()}'. (A model already trained as '{old}' needs "
                f"--rename {old}:{new.strip()} when you train again.)", was_checked=was)

        def do_delete():
            idx = chosen()
            if idx is None or self._bg_blocked():
                return
            if len(self.project.classes) <= 1:
                messagebox.showinfo("Can't delete", "Keep at least one class.", parent=win)
                return
            name = self.project.classes[idx]
            n_boxes = self.project.class_box_counts()[idx]
            move_to = None
            if n_boxes:
                ans = self._ask_delete_class(win, idx, n_boxes)
                if ans is None:
                    return
                move_to = ans[0]
            elif not messagebox.askyesno("Delete class", f"Delete the class '{name}'? It has no boxes.",
                                         parent=win):
                return
            was = checked_names() - {name}
            self._bg_job = True
            self.progress_var.set(f"Deleting class '{name}'...")
            win.config(cursor="watch")

            def worker():
                try:
                    r = self.project.delete_class(
                        idx, move_to=move_to,
                        progress=lambda d, t: self.root.after(
                            0, lambda: self.progress_var.set(f"Deleting class '{name}': {d * 100 // max(1, t)}%")))
                    err = None
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    r, err = None, f"{type(e).__name__}: {e}"
                self.root.after(0, lambda: finished(r, err))
            threading.Thread(target=worker, daemon=True).start()

            def finished(r, err):
                self._bg_job = False
                try:
                    win.config(cursor="")
                except Exception:
                    pass
                if err:
                    self.progress_var.set("")
                    messagebox.showerror("Couldn't delete the class", err + "\n\nYour label files were backed up in "
                                         "_class_backups first.", parent=win)
                    return
                msg = f"Deleted class '{name}'."
                if r['removed']:
                    msg += f" {r['removed']} box(es) removed."
                if r['moved']:
                    msg += f" {r['moved']} box(es) moved to '{self.project.classes[move_to - (1 if move_to > idx else 0)]}'."
                if r['unannotated']:
                    msg += f" {r['unannotated']} image(s) have no boxes left and are back in the to-do list."
                msg += " (Label backup: _class_backups.)"
                self._after_class_change(msg, boxes_changed=True, was_checked=was)
                self._refresh_image_list()

        ttk.Button(btns, text="Add...", command=do_add).pack(side=LEFT, padx=(0, 6))
        ttk.Button(btns, text="Rename...", command=do_rename).pack(side=LEFT, padx=6)
        ttk.Button(btns, text="Delete...", command=do_delete).pack(side=LEFT, padx=6)
        ttk.Button(btns, text="Close", command=lambda: win.destroy()).pack(side=RIGHT)
        lb.bind("<Double-Button-1>", lambda e: do_rename())

        def on_close(event=None):
            if event is not None and event.widget is not win:
                return
            self._classes_win = None
            self._classes_refresh = None
        win.bind("<Destroy>", on_close)
        self._classes_actions = dict(add=do_add, rename=do_rename, delete=do_delete)

    def _ask_delete_class(self, parent, idx, n_boxes):
        """Modal: what happens to the boxes of a class being deleted. Returns (move_to,) with move_to an
        index or None (remove the boxes), or None if cancelled."""
        name = self.project.classes[idx]
        dlg = Toplevel(parent)
        dlg.title("Delete class")
        dlg.transient(parent)
        dlg.grab_set()
        dlg.resizable(False, False)
        ttk.Label(dlg, justify=LEFT, wraplength=380,
                  text=f"'{name}' has {n_boxes} box(es) in the project. What should happen to them?"
                  ).pack(padx=14, pady=(14, 8), anchor="w")
        mode = StringVar(value="remove")
        ttk.Radiobutton(dlg, text="Delete those boxes", variable=mode, value="remove").pack(anchor="w", padx=20)
        row = ttk.Frame(dlg)
        row.pack(anchor="w", padx=20, pady=(2, 0))
        ttk.Radiobutton(row, text="Move them to:", variable=mode, value="move").pack(side=LEFT)
        others = [c for i, c in enumerate(self.project.classes) if i != idx]
        target = StringVar(value=others[0])
        ttk.Combobox(row, textvariable=target, values=others, state="readonly", width=22).pack(side=LEFT, padx=6)
        ttk.Label(dlg, justify=LEFT, wraplength=380,
                  text="A copy of your label files is saved to _class_backups first. Images left with no boxes "
                       "go back to the to-do list. This renumbers the classes after it.").pack(
            padx=14, pady=(10, 4), anchor="w")
        out = {'v': None}

        def ok():
            if mode.get() == "move":
                out['v'] = (self.project.classes.index(target.get()),)
            else:
                out['v'] = (None,)
            dlg.destroy()
        bf = ttk.Frame(dlg)
        bf.pack(pady=(6, 14))
        ttk.Button(bf, text="Delete class", command=ok).pack(side=LEFT, padx=6)
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=LEFT, padx=6)
        dlg.update_idletasks()
        dlg.geometry(f"+{parent.winfo_rootx() + 50}+{parent.winfo_rooty() + 50}")
        self.root.wait_window(dlg)
        return out['v']

    # ------------------------------------------------------------------
    # Image list / navigation
    # ------------------------------------------------------------------

    def _visible_image_ids(self):
        """The sidebar is a work queue, not a full gallery: once an image
        is annotated it drops off, so what's left is exactly what still
        needs doing. Two exceptions: the "Show annotated too" toggle turns
        this back into the full unfiltered list for review, and whatever
        image is currently open always stays visible even if it just
        became annotated - so it doesn't vanish out from under you
        mid-edit. It only disappears once you navigate away from it.
        Ids sort chronologically within a Gimble session (zero-padded
        sample number), so Prev/Next steps through the recording in order."""
        ids = self.project.list_images()
        imgs = self.project.manifest['images']
        if self.me:
            ids = [i for i in ids if not self._is_hidden_info(imgs[i])]    # other people's data stays shut until Access
        if self.active_session is not None:
            ids = [i for i in ids if imgs[i].get('session_id') == self.active_session]
        if self.show_annotated_var.get():
            return ids
        return [img_id for img_id in ids
                if img_id == self.current_image_id or not self.project.is_annotated(img_id)]

    def _refresh_session_combo(self):
        """Rebuilds the working-set picker: 'All sets' plus one entry per imported set, marked
        [A] once every frame in it is annotated, with (annotated/total)."""
        prog = self.project.session_progress()
        hidden = self._hidden_sessions()
        prog = {sid: v for sid, v in prog.items() if sid not in hidden}
        if self.active_session is not None and self.active_session not in prog:
            self.active_session = None
        keys, labels = [None], []
        tot = sum(t for t, _ in prog.values())
        ann = sum(a for _, a in prog.values())
        labels.append(f"All sets  ({ann}/{tot})")
        for sid in sorted(prog):
            t, a = prog[sid]
            keys.append(sid)
            labels.append(("[A] " if t and a == t else "") + f"{sid}"
                          + f"  ({a}/{t})")
        self._session_keys = keys
        self.session_combo.configure(values=labels)
        self.session_combo.current(keys.index(self.active_session))

    def _on_session_pick(self, event=None):
        i = self.session_combo.current()
        if i < 0 or i >= len(self._session_keys):
            return
        self.set_active_session(self._session_keys[i])

    def set_active_session(self, sid):
        """Show only this set's frames (None = every set). Lands on the set's first frame that still
        needs work; a fully annotated set opens in review mode (annotated frames shown)."""
        self.active_session = sid
        self.current_image_id = None
        self.current_pil_image = None
        self._set_boxes([])
        self.selected_box_index = None
        self._depth_cache = (None, None)
        if not self._visible_image_ids() and self.project.list_images():
            self.show_annotated_var.set(True)       # nothing left to do here - show everything to review
        self._refresh_image_list()
        self._redraw()
        self._update_status()

    def _refresh_image_list(self):
        self._refresh_session_combo()
        self.image_listbox.delete(0, END)
        ids = self._visible_image_ids()
        for img_id in ids:
            if self.project.is_annotated(img_id):
                mark = "✓"
            elif self.project.is_ai_pending(img_id):
                mark = "?"
            else:
                mark = " "
            self.image_listbox.insert(END, f"[{mark}] {img_id}")
        if self.current_image_id is None:
            if ids:
                self.load_image(ids[0])
        elif self.current_image_id in ids:
            self.image_listbox.selection_clear(0, END)
            self.image_listbox.selection_set(ids.index(self.current_image_id))

    def _on_image_select(self, event):
        sel = self.image_listbox.curselection()
        if not sel:
            return
        ids = self._visible_image_ids()
        self.load_image(ids[sel[0]])

    def prev_image(self):
        ids = self._visible_image_ids()
        if not ids or self.current_image_id not in ids:
            return
        i = ids.index(self.current_image_id)
        if i > 0:
            self.load_image(ids[i - 1])

    def next_image(self):
        ids = self._visible_image_ids()
        if not ids or self.current_image_id not in ids:
            return
        i = ids.index(self.current_image_id)
        if i < len(ids) - 1:
            self.load_image(ids[i + 1])

    def rotate_current_frame(self):
        """Rotates the currently open image 90 degrees clockwise - for a
        frame that came out sideways or upside-down (two clicks fixes
        upside-down; direction doesn't matter there since 180 is 180
        either way). Project.rotate_image does the real work: rewrites
        the image file rotated, and remaps any existing box coordinates
        to match so they stay pinned to the same real-world objects
        rather than ending up scattered across the rotated frame.

        No confirmation dialog - unlike Delete Frame this is easily
        undone (three more clicks gets back to the original orientation)
        and doesn't discard anything. Reloads the image and boxes fresh
        from disk afterward rather than trying to rotate the in-memory
        PIL image/box list in place, so what's on screen is guaranteed
        to match exactly what Project.rotate_image actually wrote."""
        img_id = self.current_image_id
        if img_id is None:
            return
        self.project.rotate_image(img_id, clockwise=True)
        self._depth_cache = (None, None)      # depth was rotated along with the image
        self.zoom, self.view_center = 1.0, None
        self.current_pil_image = Image.open(self.project.image_path(img_id)).convert("RGB")
        self._set_boxes(self.project.get_boxes(img_id), self.project.get_distances(img_id))
        self.selected_box_index = None
        self._redraw()
        self._update_status()

    def delete_current_frame(self):
        """Permanently deletes the currently open image from the project
        - not just its boxes (see clear_and_retry_ai for that), the
        image file itself, its label file, and any distorted variants
        (Project.delete_image handles all of that). For genuinely
        unwanted footage - a blurry frame, a near-duplicate, one where
        nothing of interest is in view - that's just clutter to keep
        annotating or reviewing, rather than an annotation worth
        redoing.

        Always confirms first, naming what's about to be lost, since
        there's no undo once the files are gone. Afterward, lands on
        whatever took this frame's place at the same position in the
        current filtered view (or the previous frame if this was the
        last one) - so deleting several bad frames in a row while
        stepping through a recording doesn't keep bouncing you back to the
        start of the list."""
        img_id = self.current_image_id
        if img_id is None:
            return

        detail = f"'{img_id}'"
        if self.boxes:
            status = "confirmed" if self.project.is_annotated(img_id) else "an unconfirmed AI draft"
            detail += f" ({len(self.boxes)} box(es), {status})"
        if not messagebox.askyesno(
            "Delete this frame?",
            f"Permanently delete {detail} and its image/label file(s) from disk?\n\n"
            "This can't be undone."
        ):
            return

        ids = self._visible_image_ids()
        pos = ids.index(img_id) if img_id in ids else None

        self.project.delete_image(img_id)

        self.current_image_id = None
        self.current_pil_image = None
        self._set_boxes([])
        self.selected_box_index = None

        remaining = self._visible_image_ids()
        if remaining:
            # same position if something slid into it, otherwise the new
            # last frame (this was the last one) - never back to the top
            next_id = remaining[pos] if pos is not None and pos < len(remaining) else remaining[-1]
            self.load_image(next_id)
        else:
            self._refresh_image_list()
            self._redraw()
            self._update_status()

    def load_image(self, img_id):
        self.current_image_id = img_id
        self.zoom, self.view_center = 1.0, None
        self.current_pil_image = Image.open(self.project.image_path(img_id)).convert("RGB")
        self._set_boxes(self.project.get_boxes(img_id), self.project.get_distances(img_id))
        self.selected_box_index = None

        # the newly-current image is always included by _visible_image_ids,
        # so a full refresh here keeps the sidebar list and its selection
        # correct even if img_id just became visible (or, if we navigated
        # away from a now-annotated image, just dropped out of the list)
        self._refresh_image_list()

        self._redraw()
        self._update_status()

        # If a model's loaded and you've got classes checked, this new
        # image gets predicted immediately (and only this one) - see
        # _trigger_live_predict. No-op if there's no model, no classes
        # checked, or the image is already human-confirmed.
        self._trigger_live_predict(img_id)

    # ------------------------------------------------------------------
    # Canvas drawing + coordinate conversion
    # ------------------------------------------------------------------

    def _fit_geometry(self):
        """(scale, (off_x, off_y), (disp_w, disp_h)) of the image on the canvas. At zoom 1 the whole image is
        fitted and centred; zoomed in, the point self.view_center sits in the middle of the window, kept so the
        image always covers the window (no empty borders) where it is larger than the window."""
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self.current_pil_image.size
        fit = min(cw / iw, ch / ih)
        scale = fit * self.zoom
        disp_w, disp_h = iw * scale, ih * scale
        vcx, vcy = self.view_center if (self.zoom > 1 and self.view_center) else (iw / 2, ih / 2)
        off_x = cw / 2 - vcx * scale
        off_y = ch / 2 - vcy * scale
        off_x = (cw - disp_w) / 2 if disp_w <= cw else min(0, max(cw - disp_w, off_x))
        off_y = (ch - disp_h) / 2 if disp_h <= ch else min(0, max(ch - disp_h, off_y))
        return scale, (off_x, off_y), (disp_w, disp_h)

    def _set_view(self, zoom, center):
        """Applies a zoom + view centre (image px) and stores the clamped result."""
        self.zoom = max(1.0, min(ZOOM_MAX, zoom))
        self.view_center = center if self.zoom > 1 else None
        if self.current_pil_image is not None and self.zoom > 1:
            scale, (off_x, off_y), _ = self._fit_geometry()          # re-read: this applies the clamping
            cw = max(self.canvas.winfo_width(), 1)
            ch = max(self.canvas.winfo_height(), 1)
            self.view_center = ((cw / 2 - off_x) / scale, (ch / 2 - off_y) / scale)
        self._redraw()

    def fit_view(self):
        """Back to the whole image in the window."""
        self.zoom, self.view_center = 1.0, None
        if self.current_pil_image is not None:
            self._redraw()

    def _zoom_at(self, cx, cy, steps):
        """Zoom in (steps > 0) or out around canvas point (cx, cy): the spot under the pointer stays put."""
        if self.current_pil_image is None:
            return
        new_zoom = max(1.0, min(ZOOM_MAX, self.zoom * (ZOOM_STEP ** steps)))
        if new_zoom == self.zoom:
            return
        ix, iy = self._canvas_to_image_px(cx, cy)
        old_zoom, old_center = self.zoom, self.view_center
        self.zoom = new_zoom
        if new_zoom <= 1:
            self.fit_view()
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self.current_pil_image.size
        scale = min(cw / iw, ch / ih) * new_zoom
        self._set_view(new_zoom, (ix - (cx - cw / 2) / scale, iy - (cy - ch / 2) / scale))

    def _on_canvas_wheel(self, event):
        steps = 1 if event.delta > 0 else -1
        self._zoom_at(event.x, event.y, steps)

    def _on_pan_start(self, event):
        self._pan_last = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self._pan_last is None or self.current_pil_image is None or self.zoom <= 1:
            return
        dx, dy = event.x - self._pan_last[0], event.y - self._pan_last[1]
        self._pan_last = (event.x, event.y)
        scale = self._fit_geometry()[0]
        vcx, vcy = self.view_center
        self._set_view(self.zoom, (vcx - dx / scale, vcy - dy / scale))

    def _canvas_to_image_px(self, cx, cy):
        scale, (off_x, off_y), _ = self._fit_geometry()
        return (cx - off_x) / scale, (cy - off_y) / scale

    def _image_px_to_canvas(self, ix, iy):
        scale, (off_x, off_y), _ = self._fit_geometry()
        return ix * scale + off_x, iy * scale + off_y

    def _toggle_depth(self):
        self._toggle_pick()

    def _current_depth_points(self):
        """(u, v, z_m) arrays for the open image's LiDAR returns, or None
        if it has no depth. Cached per image id (reading the PNG on every
        redraw - resizes fire a lot of them - would be wasteful);
        rotate_current_frame clears the cache since the depth turns with
        the image."""
        img_id = self.current_image_id
        if img_id is None:
            return None
        if self._depth_cache[0] != img_id:
            self._depth_cache = (img_id, self.project.depth_points(img_id))
        return self._depth_cache[1]

    def _overlay_depth(self, resized, scale, origin=(0, 0)):
        """Paints the LiDAR returns onto the already-resized display image
        (colour = distance: red near, blue far), far points first so near
        ones end up on top. Done in numpy on the PIL image rather than as
        thousands of canvas items - a single sample has ~7000 returns."""
        pts = self._current_depth_points()
        if pts is None or len(pts[0]) == 0:
            return resized
        u, v, z = pts
        arr = np.array(resized)
        h, w = arr.shape[:2]
        order = np.argsort(-z)
        cols = ac.depth_colors(z[order], DEPTH_NEAR_M, DEPTH_FAR_M)
        x = np.round((u[order] - origin[0]) * scale).astype(int)
        y = np.round((v[order] - origin[1]) * scale).astype(int)
        r = 1 if scale < 1.6 else (2 if scale < 4 else 3)
        for dy in range(-r + 1, r + 1):
            for dx in range(-r + 1, r + 1):
                arr[np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)] = cols
        return Image.fromarray(arr)

    def _redraw(self):
        self.canvas.delete("all")
        if self.current_pil_image is None:
            return

        scale, (off_x, off_y), (disp_w, disp_h) = self._fit_geometry()
        self.scale, self.offset = scale, (off_x, off_y)

        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self.current_pil_image.size
        # only the part of the image that is on screen is resized (zoomed in, the whole image would be enormous)
        x0 = int(max(0, math.floor(-off_x / scale)))
        y0 = int(max(0, math.floor(-off_y / scale)))
        x1 = int(min(iw, math.ceil((cw - off_x) / scale)))
        y1 = int(min(ih, math.ceil((ch - off_y) / scale)))
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        if self.zoom <= 1:
            crop = self.current_pil_image
            resized = crop.resize((max(int(disp_w), 1), max(int(disp_h), 1)))
            x0 = y0 = 0
        else:
            crop = self.current_pil_image.crop((x0, y0, x1, y1))
            resized = crop.resize((max(int(round((x1 - x0) * scale)), 1), max(int(round((y1 - y0) * scale)), 1)))
        if self.show_depth_var.get():
            resized = self._overlay_depth(resized, scale, (x0, y0))
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(off_x + x0 * scale, off_y + y0 * scale, anchor=NW, image=self.tk_image)
        if self.show_depth_var.get():
            pts = self._current_depth_points()
            note = (f"LiDAR depth: {len(pts[0])} points, {DEPTH_NEAR_M:g} m (red) to {DEPTH_FAR_M:g} m (blue)"
                    if pts is not None else "No LiDAR depth for this image")
            self.canvas.create_text(off_x + 8, off_y + 8, text=note, fill="#ffffff", anchor=NW,
                                     font=("Segoe UI", 9, "bold"))

        # Boxes the model predicted but no human has confirmed yet get a
        # dashed outline and an "(AI)" tag instead of the normal solid
        # rectangle - a deliberate visual reminder that these are drafts,
        # not confirmed annotations, so review doesn't turn into a rubber
        # stamp. Goes solid the instant you edit or Confirm Image them.
        pending = self.project.is_ai_pending(self.current_image_id)

        img_w, img_h = self.current_pil_image.size
        for i, (cls, xc, yc, w, h) in enumerate(self.boxes):
            x1, y1, x2, y2 = ac.yolo_box_to_pixel(xc, yc, w, h, img_w, img_h)
            cx1, cy1 = self._image_px_to_canvas(x1, y1)
            cx2, cy2 = self._image_px_to_canvas(x2, y2)
            color = color_for_class(cls)
            width = 3 if i == self.selected_box_index else 2
            rect_kwargs = {"outline": color, "width": width}
            if pending:
                rect_kwargs["dash"] = (6, 3)
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, **rect_kwargs)
            label = self.project.classes[cls] if cls < len(self.project.classes) else f"class {cls}"
            if pending:
                label += " (AI)"
            pk = self.box_dist[i] if i < len(self.box_dist) else None
            if pk:
                label += f"  {pk['distance_m']:.2f} m"
            self.canvas.create_text(cx1 + 4, cy1 - 10, text=label, fill=color, anchor=NW,
                                     font=("Segoe UI", 9, "bold"))
            if pk:
                mx, my = self._image_px_to_canvas(pk['u'], pk['v'])
                self.canvas.create_oval(mx - 6, my - 6, mx + 6, my + 6, outline="#ffffff", width=2)
                self.canvas.create_oval(mx - 3, my - 3, mx + 3, my + 3, fill=color, outline=color)

    # ------------------------------------------------------------------
    # Canvas mouse interaction
    # ------------------------------------------------------------------

    def _box_at_canvas_point(self, cx, cy):
        """Returns the index of the smallest box (by area) containing this
        canvas point, or None. Smallest-first so clicking inside a small
        box nested in a bigger one selects the small one, as expected."""
        ix, iy = self._canvas_to_image_px(cx, cy)
        img_w, img_h = self.current_pil_image.size
        best_idx, best_area = None, None
        for i, (cls, xc, yc, w, h) in enumerate(self.boxes):
            x1, y1, x2, y2 = ac.yolo_box_to_pixel(xc, yc, w, h, img_w, img_h)
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:
                    best_area, best_idx = area, i
        return best_idx

    def _on_canvas_press(self, event):
        """Always just records a potential drag start - there's no mode
        to branch on. _on_canvas_release decides afterward whether the
        gesture that just happened was a real drag (draw a new box, even
        on top of existing ones - crowded scenes never block you) or too
        small a movement to be one (a plain click - select whatever's
        underneath instead, see _on_canvas_release). Deleting a box never
        goes through press/release at all - see _on_canvas_double_click
        and _on_canvas_right_click."""
        if self.current_pil_image is None:
            return
        self.draw_start = (event.x, event.y)

    def _on_canvas_drag(self, event):
        if self.draw_start is None:
            return
        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)
        x0, y0 = self.draw_start
        color = color_for_class(self.current_class_id)
        self.temp_rect_id = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline=color, width=2, dash=(4, 2))

    def _on_canvas_release(self, event):
        if self.draw_start is None:
            return
        x0, y0 = self.draw_start
        self.draw_start = None
        if self.temp_rect_id is not None:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None

        if abs(event.x - x0) < 4 or abs(event.y - y0) < 4:
            if self.pick_var.get():
                self._assign_pick(event.x, event.y)
                return
            # too small a movement to be an intentional box - treat this
            # as a plain click instead: select whatever's under it (for
            # relabeling with a number key, Delete Selected Box, or a
            # follow-up double-click), or clear the selection if nothing
            # is there.
            self.selected_box_index = self._box_at_canvas_point(event.x, event.y)
            self._redraw()
            return

        if not self.project.classes:
            messagebox.showinfo("Add a class first", "Add at least one class before drawing boxes.")
            return

        ix0, iy0 = self._canvas_to_image_px(x0, y0)
        ix1, iy1 = self._canvas_to_image_px(event.x, event.y)
        img_w, img_h = self.current_pil_image.size
        xc, yc, w, h = ac.pixel_box_to_yolo(ix0, iy0, ix1, iy1, img_w, img_h)
        self.boxes.append((self.current_class_id, xc, yc, w, h))
        self.box_dist.append(None)
        self.selected_box_index = None
        self._save_current_boxes()
        self._redraw()

    def _on_canvas_double_click(self, event):
        """Double-click a box to delete it instantly - whatever's right
        under the cursor, not necessarily whatever's currently selected
        (the first click of the pair already ran through
        _on_canvas_press/_on_canvas_release like any other click, which
        selects it anyway, but this deletes by position so it works even
        if the two clicks land on slightly different boxes)."""
        if self.pick_var.get():
            return      # in pick mode clicks assign distances; deleting needs the button or right-click
        self._delete_box_index(self._box_at_canvas_point(event.x, event.y))

    def _on_canvas_right_click(self, event):
        """Right-click a box to delete it instantly - no selection step
        needed first, same as double-click (see _on_canvas_double_click)
        but a single click instead of two. In pick mode a right-click
        clears that box's distance instead (never deletes the box)."""
        if self.pick_var.get():
            self._clear_distance_index(self._box_at_canvas_point(event.x, event.y))
            return
        self._delete_box_index(self._box_at_canvas_point(event.x, event.y))

    def _delete_box_index(self, index):
        """Deletes boxes[index] and keeps selected_box_index consistent -
        shared by the sidebar's Delete Selected Box button, double-click,
        and right-click, so there's exactly one place that knows how to
        remove a box and fix up the selection afterward (removing index i
        shifts every later box's index down by one, so a stale
        selected_box_index would otherwise silently start pointing at the
        wrong box)."""
        if index is None or not (0 <= index < len(self.boxes)):
            return
        del self.boxes[index]
        del self.box_dist[index]
        if self.selected_box_index == index:
            self.selected_box_index = None
        elif self.selected_box_index is not None and self.selected_box_index > index:
            self.selected_box_index -= 1
        self._save_current_boxes()
        self._redraw()

    def delete_selected_box(self):
        self._delete_box_index(self.selected_box_index)

    def _save_current_boxes(self):
        if self.current_image_id is None:
            return
        img_id = self.current_image_id
        self._fit_dist()
        self.project.set_boxes(img_id, self.boxes, distances=self.box_dist)
        # rebuilds the visible list (checkmark now correct) and re-selects
        # img_id, which _visible_image_ids always keeps pinned while it's
        # the open image - see load_image/_visible_image_ids
        self._refresh_image_list()
        self._update_status()

    # ------------------------------------------------------------------
    # Per-box LiDAR distance picking
    # ------------------------------------------------------------------

    def _set_boxes(self, boxes, dists=None):
        """The one place the box list and its parallel distance-pick list are replaced together,
        so they can never drift apart."""
        self.boxes = list(boxes)
        self.box_dist = list(dists) if dists is not None else []
        self._fit_dist()

    def _fit_dist(self):
        n = len(self.boxes)
        self.box_dist = (self.box_dist + [None] * n)[:n]

    def _toggle_pick(self):
        self.pick_var.set(not self.pick_var.get())
        self._on_pick_toggle()

    def _on_pick_toggle(self):
        on = self.pick_var.get()
        self.show_depth_var.set(on)             # dots and picking are one switch
        self.canvas.configure(cursor="tcross" if on else "crosshair")
        self.canvas.delete("hover")
        self._redraw()

    def _pick_candidate(self, cx, cy):
        """(box_index, u, v, z_m) for the LiDAR point that a click at canvas (cx, cy) would attach
        to the box under the cursor: the nearest return INSIDE that box, within PICK_RADIUS_PX
        of the cursor. None if there's no box / depth / point close enough."""
        if self.current_pil_image is None:
            return None
        idx = self._box_at_canvas_point(cx, cy)
        pts = self._current_depth_points()
        if idx is None or pts is None or len(pts[0]) == 0:
            return None
        u, v, z = pts
        img_w, img_h = self.current_pil_image.size
        _, xc, yc, w, h = self.boxes[idx]
        x1, y1, x2, y2 = ac.yolo_box_to_pixel(xc, yc, w, h, img_w, img_h)
        inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        if not inside.any():
            return None
        ix, iy = self._canvas_to_image_px(cx, cy)
        ui, vi, zi = u[inside], v[inside], z[inside]
        d2 = (ui - ix) ** 2 + (vi - iy) ** 2
        k = int(np.argmin(d2))
        if (d2[k] ** 0.5) * self.scale > PICK_RADIUS_PX:
            return None
        return idx, int(ui[k]), int(vi[k]), float(zi[k])

    def _on_canvas_motion(self, event):
        self.canvas.delete("hover")
        if not self.pick_var.get() or self.current_pil_image is None:
            return
        cand = self._pick_candidate(event.x, event.y)
        if cand is None:
            idx = self._box_at_canvas_point(event.x, event.y)
            if self._current_depth_points() is None:
                msg = "no LiDAR depth for this image"
            elif idx is None:
                msg = "hover inside a box"
            else:
                msg = "no LiDAR point near here"
            self._hover_label(event.x + 14, event.y + 14, msg, "#bbbbbb")
            return
        idx, u, v, z = cand
        px, py = self._image_px_to_canvas(u, v)
        self.canvas.create_oval(px - 8, py - 8, px + 8, py + 8, outline="#ffffff", width=2, tags="hover")
        self.canvas.create_line(px - 12, py, px + 12, py, fill="#ffffff", tags="hover")
        self.canvas.create_line(px, py - 12, px, py + 12, fill="#ffffff", tags="hover")
        self._hover_label(px + 14, py - 26, f"{z:.2f} m  (click to attach)", "#ffffff")

    def _hover_label(self, x, y, text, color):
        t = self.canvas.create_text(x, y, text=text, fill=color, anchor=NW,
                                     font=("Segoe UI", 10, "bold"), tags="hover")
        bb = self.canvas.bbox(t)
        if bb:
            r = self.canvas.create_rectangle(bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2,
                                              fill="#000000", outline="", tags="hover")
            self.canvas.tag_lower(r, t)

    def _assign_pick(self, cx, cy):
        cand = self._pick_candidate(cx, cy)
        if cand is None:
            idx = self._box_at_canvas_point(cx, cy)
            self.progress_var.set("Hover inside a box, near a LiDAR point, then click." if idx is None
                                   else "No LiDAR point close enough inside that box.")
            self.root.after(2500, lambda: self.progress_var.set("") if not self._bg_job else None)
            return
        idx, u, v, z = cand
        self._fit_dist()
        self.box_dist[idx] = {"distance_m": round(z, 3), "u": u, "v": v}
        self.selected_box_index = idx
        self._save_current_boxes()
        self._redraw()

    def _clear_distance_index(self, index):
        if index is None or not (0 <= index < len(self.boxes)):
            return
        self._fit_dist()
        if self.box_dist[index] is None:
            return
        self.box_dist[index] = None
        self._save_current_boxes()
        self._redraw()

    def clear_selected_distance(self):
        self._clear_distance_index(self.selected_box_index)

    # ------------------------------------------------------------------
    # Session management ("Manage Sessions..." mini menu)
    # ------------------------------------------------------------------

    def _bg_blocked(self):
        """Renaming/deleting moves files the background jobs are using - refuse until they finish."""
        busy = self._bg_job or self._predict_busy
        if busy:
            messagebox.showinfo(
                "Busy", "Background work (import, publish or AI prediction) is still "
                "running. Try again in a moment.")
        return bool(busy)

    def manage_sessions(self):
        if getattr(self, '_sessions_win', None) is not None:
            try:
                self._sessions_win.lift()
                return
            except Exception:
                self._sessions_win = None
        win = self._sessions_win = Toplevel(self.root)
        win.title("Sessions")
        win.geometry("1000x400")
        win.minsize(900, 300)
        win.transient(self.root)

        cols = ("session", "samples", "annotated", "depth", "source")
        tree = Treeview(win, columns=cols, show="headings", selectmode="browse", height=9)
        for c, text, wd, anchor in (("session", "Session", 240, "w"), ("samples", "Samples", 70, "e"),
                                     ("annotated", "Annotated", 80, "e"), ("depth", "With depth", 80, "e"),
                                     ("source", "Recorded from", 300, "w")):
            tree.heading(c, text=text)
            tree.column(c, width=wd, anchor=anchor, stretch=(c == "source"))
        btns = ttk.Frame(win)
        btns.pack(side=BOTTOM, fill=X, padx=10, pady=10)
        info = StringVar(value="")
        ttk.Label(win, textvariable=info, wraplength=860, justify=LEFT).pack(side=BOTTOM, anchor="w", padx=10)
        tree.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=(10, 6))

        def refresh():
            sel = tree.selection()
            keep = sel[0] if sel else None
            tree.delete(*tree.get_children())
            rows = self.project.session_summary()
            owner_of = {r['session_id']: sz.session_owner(self.project, r['session_id'], self.me) for r in rows}
            mine = [r for r in rows if not self._is_foreign_owner(owner_of[r['session_id']])]
            people = {}
            for r in rows:
                if self._is_foreign_owner(owner_of[r['session_id']]):
                    people.setdefault(owner_of[r['session_id']], []).append(r)

            def add(r, indent=""):
                tree.insert("", END, iid=r['session_id'],
                            values=(indent + r['session_id'] + ("  (current set)" if r['session_id'] == self.active_session else ""),
                                    r['samples'], r['annotated'], r['with_depth'], r['source_path']))
            for r in mine:
                add(r)
            for pid in sorted(people, key=lambda x: sz.people_name(self.project, x).lower()):
                grp = people[pid]
                opened = pid in self.access_people
                tree.insert("", END, iid=f"person:{pid}", tags=("person",),
                            values=(f"[{sz.people_name(self.project, pid)}] data - {len(grp)} set(s), {sum(g['samples'] for g in grp)} image(s)"
                                    + ("   (open - click Access to hide)" if opened else "   (click Access to open)"), "", "", "", ""))
                if opened:
                    for r in grp:
                        add(r, indent="     ")
            if keep and tree.exists(keep):
                tree.selection_set(keep)
            shown = mine + [r for pid, g in people.items() if pid in self.access_people for r in g]
            tot = sum(r['samples'] for r in shown)
            info.set(f"{len(shown)} session(s), {tot} sample(s)." + (
                         "  Other people's data is closed until you click Access." if people else "")
                     if shown or people else "No sessions imported yet - use File > Import Gimble Session...")

        self._sessions_refresh = refresh

        def chosen():
            sel = tree.selection()
            if not sel:
                messagebox.showinfo("Pick a session", "Click a session in the list first.", parent=win)
                return None
            if sel[0].startswith("person:"):
                messagebox.showinfo("Open their data first", "That row is a person's data - click Access to open it, "
                                    "then pick one of their sets.", parent=win)
                return None
            return sel[0]

        def do_access():
            sel = tree.selection()
            if not sel:
                messagebox.showinfo("Pick a person", "Click a person's row first (the one starting with their name in [ ]).", parent=win)
                return
            if sel[0].startswith("person:"):
                pid = sel[0][len("person:"):]
            else:
                pid = sz.session_owner(self.project, sel[0], self.me)
                if not self._is_foreign_owner(pid):
                    messagebox.showinfo("Your own data", "That set is yours - Access is only for other people's data.", parent=win)
                    return
            if pid in self.access_people:
                self.access_people.discard(pid)
                if self.active_session and sz.session_owner(self.project, self.active_session, self.me) == pid:
                    self.active_session = None
            else:
                self.access_people.add(pid)
            self._refresh_session_combo()
            self._refresh_image_list()
            self._update_status()
            refresh()

        def do_rename():
            sid = chosen()
            if sid is None or self._bg_blocked():
                return
            new = sid
            while True:
                new = simpledialog.askstring(
                    "Rename session", f"New name for '{sid}':\n(no spaces; every image, label, depth file and\n"
                    "export entry is renamed to match)", initialvalue=new, parent=win)
                if new is None:
                    return
                new = new.strip()
                if new == sid:
                    return
                err = ac.Project.validate_session_name(new)
                if err is None and new in {r['session_id'] for r in self.project.session_summary()}:
                    err = f"A session named '{new}' already exists."
                if err is None:
                    break
                messagebox.showerror("Can't use that name", err, parent=win)
            try:
                res = self.project.rename_session(sid, new)
            except Exception as e:
                messagebox.showerror("Rename failed", str(e), parent=win)
                return
            cur = self.current_image_id
            if cur in res['id_map']:
                self.current_image_id = res['id_map'][cur]
                self._depth_cache = (None, None)
            if self.active_session == sid:
                self.active_session = new
            self._refresh_image_list()
            self._update_status()
            refresh()
            self._announce(f"Renamed {res['renamed']} sample(s) to '{new}'.")

        def do_delete():
            sid = chosen()
            if sid is None or self._bg_blocked():
                return
            r = next(x for x in self.project.session_summary() if x['session_id'] == sid)
            if not messagebox.askyesno(
                    "Delete session?",
                    f"Permanently delete session '{sid}' from the project?\n\n"
                    f"{r['samples']} sample(s), {r['annotated']} annotated - their images, LiDAR depth, "
                    "boxes and distance picks are all removed, and they drop out of "
                    "the shared data.\n\nThe original recording on the drive is not touched.\n"
                    "This can't be undone.", icon="warning", default="no", parent=win):
                return
            cur_in = self.current_image_id in set(self.project.session_image_ids(sid))
            n = self.project.delete_session(sid)
            if cur_in:
                self.current_image_id = None
                self.current_pil_image = None
                self._set_boxes([])
                self.selected_box_index = None
                self._depth_cache = (None, None)
                self._redraw()
            self._refresh_image_list()
            self._update_status()
            refresh()
            self._announce(f"Deleted {n} sample(s) of '{sid}'.")

        ttk.Button(btns, text="Access", command=do_access).pack(side=LEFT, padx=(0, 6))
        ttk.Button(btns, text="Rename...", command=do_rename).pack(side=LEFT, padx=6)
        ttk.Button(btns, text="Delete...", command=do_delete).pack(side=LEFT, padx=6)

        def do_thin():
            sid = chosen()
            if sid is None or self._bg_blocked():
                return
            self.thin_set(sid, parent=win)
        ttk.Button(btns, text="Thin Out...", command=do_thin).pack(side=LEFT, padx=6)

        def do_open():
            sid = chosen()
            if sid is None:
                return
            self.set_active_session(sid)
            win.destroy()
        ttk.Button(btns, text="Current Set", command=do_open).pack(side=LEFT, padx=6)
        ttk.Button(btns, text="Close", command=lambda: win.destroy()).pack(side=RIGHT)
        tree.bind("<Double-Button-1>", lambda e: do_access() if tree.selection() and tree.selection()[0].startswith("person:") else do_rename())
        self._sessions_access = do_access

        def on_close(event=None):
            if event is not None and event.widget is not win:
                return
            self._sessions_refresh = None
            self._sessions_win = None
        win.bind("<Destroy>", on_close)
        refresh()

    # ------------------------------------------------------------------
    # Sharing annotated data (publish to your folder in the shared folder) - see share_zip.py
    # ------------------------------------------------------------------

    def _is_foreign_owner(self, pid):
        return bool(pid) and bool(self.me) and pid != self.me['id']

    def _is_hidden_info(self, info):
        """True for an image of somebody else whose data you have not opened with Access."""
        pid = info.get('owner_id')
        return self._is_foreign_owner(pid) and pid not in self.access_people

    def _hidden_sessions(self):
        if not self.me:
            return set()
        return {info['session_id'] for info in self.project.manifest['images'].values()
                if info.get('session_id') and self._is_hidden_info(info)}

    def _save_cfg(self):
        try:
            sz.save_config(self.cfg)
        except OSError:
            pass

    def _ensure_me(self, parent=None):
        """Your name, asked once and remembered (stays on this computer). None if you cancel."""
        if self.me:
            return self.me
        msg = ("What's your name?\\n\\nAsked once. It is stamped on the data you share, so everyone can tell "
               "whose annotations are whose.\\n(no spaces or dots - use _ or -)")
        while True:
            name = simpledialog.askstring("Your name", msg, parent=parent or self.root)
            if name is None:
                return None
            err = sz.person_name_error(name)
            if err is None:
                break
            messagebox.showerror("Can't use that name", err, parent=parent or self.root)
        self.me = sz.set_identity(self.cfg, name)
        self._save_cfg()
        self.project.editor_id = self.me['id']
        self.project.manifest.setdefault('people', {})[self.me['id']] = self.me['name']
        self.project.save_manifest()
        return self.me

    def change_my_name(self):
        if not self.me:
            self._ensure_me()
            return
        while True:
            name = simpledialog.askstring("Your name", "Your name (your data keeps its owner id, so changing this "
                                          "only changes how it is shown):", initialvalue=self.me['name'], parent=self.root)
            if name is None:
                return
            err = sz.person_name_error(name)
            if err is None:
                break
            messagebox.showerror("Can't use that name", err)
        self.me = sz.set_identity(self.cfg, name)
        self._save_cfg()
        self.project.manifest.setdefault('people', {})[self.me['id']] = self.me['name']
        self.project.save_manifest()

    def choose_shared_dir(self):
        """Where the shared folder is (e.g. inside your synced Google Drive). Remembered on this computer."""
        cur = self.cfg.get('shared_dir') or ''
        d = filedialog.askdirectory(title="Pick the shared folder (everyone publishes into it, one folder per person)",
                                    initialdir=cur or None)
        if not d:
            return None
        self.cfg['shared_dir'] = os.path.normpath(d)
        self._save_cfg()
        return self.cfg['shared_dir']

    def _shared_dir(self):
        d = self.cfg.get('shared_dir')
        if d and os.path.isdir(d):
            return d
        if d:
            messagebox.showinfo("Shared folder not found", f"The shared folder isn't at\n{d}\nany more (drive not synced?). Pick it again.")
        return self.choose_shared_dir()

    def _share_fail(self, title, err):
        self._bg_job = False
        self.progress_var.set("")
        messagebox.showerror(title, str(err))

    def publish_to_shared(self):
        """Brings your own folder in the shared folder up to date: it looks at what is really there and copies only
        the annotated images that are missing or changed (an empty folder gets everything)."""
        if self._bg_blocked():
            return
        me = self._ensure_me()
        if not me:
            return
        shared = self._shared_dir()
        if not shared:
            return
        self._bg_job = True
        self.progress_var.set("Publishing...")

        def progress(done, total):
            self.root.after(0, lambda: self.progress_var.set(f"Publishing: {done}/{total}"))

        def worker():
            try:
                res = sz.publish(self.project, shared, me, progress_cb=progress)
            except Exception as e:  # noqa: BLE001
                err = e
                self.root.after(0, lambda: self._share_fail("Publish failed", err))
                return
            self.root.after(0, lambda: self._publish_done(res))
        threading.Thread(target=worker, daemon=True).start()

    def _publish_done(self, res):
        self._bg_job = False
        self.progress_var.set("")
        if not res['count']:
            messagebox.showinfo("Nothing to publish", f"Your folder in the shared folder already has all {res['checked']} "
                                "annotated image(s) you have made or edited." if res['checked'] else
                                "You have no annotated images to publish yet (only annotated images are shared).")
            return
        parts = []
        if res['new']:
            parts.append(f"{res['new']} new image(s)")
        if res['edits']:
            parts.append(f"{res['edits']} edited image(s)")
        messagebox.showinfo("Published", f"Published {' and '.join(parts)} to {res['path']}.")

    def import_to_edit(self):
        """A window to pick whose data (or everything) to bring in so you can fix it. The fix goes out in your own
        folder on the next Publish; the image keeps its original owner's name."""
        if self._bg_blocked():
            return
        me = self._ensure_me()
        if not me:
            return
        shared = self._shared_dir()
        if not shared:
            return
        packs, bad = sz.list_folders(shared)
        if not packs:
            messagebox.showinfo("Nothing there", "There is no published data in the shared folder yet."
                                + (f"\n({len(bad)} file(s) could not be read - still syncing?)" if bad else ""))
            return
        merged = sz.merge_folders(packs)
        sets = sz.available_sets(merged)
        win = Toplevel(self.root)
        win.title("Import data to edit")
        win.geometry("700x420")
        win.transient(self.root)
        ttk.Label(win, wraplength=660, justify=LEFT, text=(
            "Pick the sets you want to look at or fix. They are copied into your project; whatever you change goes into "
            "your own folder at the next Publish, and the images keep the original annotator's name. "
            "Images you do not touch are not published again.")).pack(anchor="w", padx=12, pady=10)
        tree = Treeview(win, columns=("person", "set", "images", "depth"), show="headings", selectmode="extended", height=10)
        for c, text, wd, anchor in (("person", "Annotated by", 160, "w"), ("set", "Set", 260, "w"),
                                     ("images", "Images", 80, "e"), ("depth", "With depth", 90, "e")):
            tree.heading(c, text=text)
            tree.column(c, width=wd, anchor=anchor)
        for i, g in enumerate(sets):
            tree.insert("", END, iid=str(i), values=(g['name'], g['set'], g['images'], g['with_depth']))
        tree.pack(fill=BOTH, expand=True, padx=12, pady=4)
        btns = ttk.Frame(win)
        btns.pack(fill=X, padx=12, pady=10)

        def go(everything):
            if everything:
                picks = None
            else:
                sel = tree.selection()
                if not sel:
                    messagebox.showinfo("Pick a set", "Click one or more sets in the list first.", parent=win)
                    return
                picks = [(sets[int(i)]['owner_id'], sets[int(i)]['set']) for i in sel]
            entries = sz.select_entries(merged, picks)
            win.destroy()
            self._run_import(entries, merged['people'], me)
        ttk.Button(btns, text="Import selected", command=lambda: go(False)).pack(side=LEFT)
        ttk.Button(btns, text="Import everything", command=lambda: go(True)).pack(side=LEFT, padx=8)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=RIGHT)

    def _run_import(self, entries, people, me):
        self._bg_job = True
        self.progress_var.set("Importing...")

        def progress(done, total):
            self.root.after(0, lambda: self.progress_var.set(f"Importing: {done}/{total}"))

        def worker():
            try:
                plan = sz.analyze_entries(self.project, entries, people, me)
                if plan['bad_classes']:
                    raise ValueError("The shared folders use class names this project can't hold: " + ", ".join(plan['bad_classes']))
                report = sz.apply_import(self.project, plan, None, me, progress_cb=progress)
            except Exception as e:  # noqa: BLE001
                err = e
                self.root.after(0, lambda: self._share_fail("Import failed", err))
                return
            self.root.after(0, lambda: self._import_finished(plan, report))
        threading.Thread(target=worker, daemon=True).start()

    def _import_finished(self, plan, report):
        self._bg_job = False
        self.progress_var.set("")
        for pid in {e['owner_id'] for e in plan['new']} | {e['owner_id'] for e, _ in plan['update']}:
            if self._is_foreign_owner(pid):
                self.access_people.add(pid)           # you asked for their data: it is open
        self._depth_cache = (None, None)
        if report['added_classes']:
            self._after_class_change("")
        self._refresh_session_combo()
        self._refresh_image_list()
        if self.current_image_id in report['changed_ids']:
            self.load_image(self.current_image_id)
        if self._sessions_refresh:
            self._sessions_refresh()
        self._update_status()
        parts = []
        if report['new']:
            parts.append(f"{report['new']} new image(s)")
        if report['updated']:
            parts.append(f"{report['updated']} image(s) updated")
        if report['kept_mine'] or report['kept_local']:
            parts.append(f"{report['kept_mine'] + report['kept_local']} of your newer edit(s) kept")
        if report['took_theirs']:
            parts.append(f"{report['took_theirs']} replaced by a newer published edit")
        if report['duplicates']:
            parts.append(f"{report['duplicates']} duplicate picture(s) skipped")
        if report['added_classes']:
            parts.append("new class(es): " + ", ".join(report['added_classes']))
        messagebox.showinfo("Import complete", ("Imported: " + "; ".join(parts) if parts else "Nothing new to import")
                            + f". ({report['same']} already identical.) Pick the set in the Working set box; "
                            "edit, then File > Publish to shared folder.")

    def thin_set(self, sid, parent=None):
        """Dialog: remove redundant frames from a set (e.g. a 60 fps video that is mostly repetition).
        Two ways - keep every Nth frame, or drop frames that look almost the same as the last kept one.
        Preview shows how many would go before anything is deleted. Annotated frames are never removed."""
        total, annotated = self.project.session_progress().get(sid, (0, 0))
        if total == 0:
            return
        dlg = Toplevel(parent or self.root)
        dlg.title(f"Thin out '{sid}'")
        dlg.transient(parent or self.root)
        dlg.grab_set()
        dlg.resizable(False, False)
        ttk.Label(dlg, justify=LEFT, wraplength=430,
                  text=f"'{sid}' has {total} frame(s), {annotated} annotated. Annotated frames are never removed. "
                       "Removing frames deletes them from the project for good (importing the same video / "
                       "session again would bring them back)."
                  ).pack(padx=14, pady=(14, 8), anchor="w")
        mode = StringVar(value="similar")
        nvar = StringVar(value="2")
        dvar = StringVar(value="3.0")
        r1 = ttk.Frame(dlg); r1.pack(anchor="w", padx=20, pady=2)
        ttk.Radiobutton(r1, text="Remove near-duplicates (frames that differ from the last kept one by less than)",
                        variable=mode, value="similar").pack(side=LEFT)
        r1b = ttk.Frame(dlg); r1b.pack(anchor="w", padx=44, pady=(0, 6))
        ttk.Entry(r1b, textvariable=dvar, width=6).pack(side=LEFT)
        ttk.Label(r1b, text="%   (bigger = removes more; try 2-6, check the preview)").pack(side=LEFT, padx=6)
        r2 = ttk.Frame(dlg); r2.pack(anchor="w", padx=20, pady=2)
        ttk.Radiobutton(r2, text="Keep every", variable=mode, value="nth").pack(side=LEFT)
        ttk.Entry(r2, textvariable=nvar, width=5).pack(side=LEFT, padx=4)
        ttk.Label(r2, text="th frame (2 = half of them, 3 = a third ...)").pack(side=LEFT)
        status = StringVar(value="")
        ttk.Label(dlg, textvariable=status, justify=LEFT, wraplength=430).pack(padx=14, pady=(10, 4), anchor="w")
        state = {'thumbs': None, 'busy': False, 'plan': None}

        def settings():
            try:
                if mode.get() == "nth":
                    n = int(nvar.get())
                    if n < 1:
                        raise ValueError
                    return dict(every_n=n)
                d = float(dvar.get())
                if not (0 <= d <= 100):
                    raise ValueError
                return dict(difference_pct=d)
            except ValueError:
                status.set("Type a whole number (every Nth) or a percentage between 0 and 100.")
                return None

        def make_plan():
            kw = settings()
            if kw is None:
                return None
            plan = self.project.plan_thin(sid, thumbs=state['thumbs'], **kw)
            state['plan'] = plan
            return plan

        def show(plan):
            r = len(plan['remove'])
            status.set(f"Would remove {r} of {plan['total']} frame(s) and keep {len(plan['keep'])}"
                       + (f" ({plan['annotated']} annotated kept)." if plan['annotated'] else "."))

        def preview():
            if state['busy'] or settings() is None:
                return
            if mode.get() == "similar" and state['thumbs'] is None:
                state['busy'] = True
                status.set("Reading frames: 0")

                def prog(done, tot):
                    self.root.after(0, lambda: status.set(f"Reading frames: {done}/{tot}") if dlg.winfo_exists() else None)

                def worker():
                    try:
                        th = self.project.frame_thumbnails(sid, progress_cb=prog)
                    except Exception as e:  # noqa: BLE001
                        th, err = None, str(e)
                    else:
                        err = None

                    def done():
                        state['busy'] = False
                        if not dlg.winfo_exists():
                            return
                        if err:
                            status.set("Couldn't read the frames: " + err)
                            return
                        state['thumbs'] = th
                        plan = make_plan()
                        if plan:
                            show(plan)
                    self.root.after(0, done)
                threading.Thread(target=worker, daemon=True).start()
                return
            plan = make_plan()
            if plan:
                show(plan)

        def apply():
            if state['busy']:
                return
            if mode.get() == "similar" and state['thumbs'] is None:
                status.set("Click Preview first to see what would be removed.")
                return
            plan = make_plan()
            if not plan:
                return
            if not plan['remove']:
                status.set("Nothing would be removed with these settings.")
                return
            if not messagebox.askyesno(
                    "Remove frames?",
                    f"Permanently remove {len(plan['remove'])} of {plan['total']} frame(s) from '{sid}'?\n"
                    "This can't be undone.", icon="warning", default="no", parent=dlg):
                return
            cur_gone = self.current_image_id in set(plan['remove'])
            n = self.project.thin_session(sid, plan['remove'])
            dlg.destroy()
            if cur_gone:
                self.current_image_id = None
                self.current_pil_image = None
                self._set_boxes([])
                self.selected_box_index = None
                self._depth_cache = (None, None)
                self._redraw()
            self._refresh_image_list()
            self._update_status()
            if self._sessions_refresh:
                self._sessions_refresh()
            self._announce(f"Removed {n} frame(s) from '{sid}'.")

        bf = ttk.Frame(dlg)
        bf.pack(pady=(6, 14))
        ttk.Button(bf, text="Preview", command=preview).pack(side=LEFT, padx=6)
        ttk.Button(bf, text="Remove frames", command=apply).pack(side=LEFT, padx=6)
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=LEFT, padx=6)
        self._thin_ui = dict(mode=mode, nvar=nvar, dvar=dvar, status=status, preview=preview, apply=apply, dlg=dlg, state=state)
        dlg.update_idletasks()
        p = parent or self.root
        dlg.geometry(f"+{p.winfo_rootx() + 40}+{p.winfo_rooty() + 40}")
        if getattr(self, '_thin_modal', True):
            self.root.wait_window(dlg)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_gimble_session(self):
        """Pick a session folder recorded by gimble_pi.py (the one holding
        index.csv + images/ + depth/), or the dataset folder that contains
        several sessions to import them all. Every sample becomes an image
        with its LiDAR depth attached (see Project.import_gimble_session).
        Re-importing a session only adds samples that aren't already in the
        project, so annotations are never disturbed."""
        path = filedialog.askdirectory(title="Select a Gimble session folder (or the dataset folder)")
        if not path:
            return
        path = os.path.normpath(path)
        if ac.is_gimble_session(path):
            sessions = [path]
        else:
            sessions = sorted(os.path.join(path, d) for d in os.listdir(path)
                              if ac.is_gimble_session(os.path.join(path, d)))
            if not sessions:
                messagebox.showerror(
                    "Not a Gimble session",
                    "That folder isn't a session (no index.csv + images/), and it doesn't "
                    "contain any sessions either.\n\nPick a folder like\n"
                    "dataset/20260930_095437")
                return
            if not messagebox.askyesno(
                    "Import all sessions?",
                    f"Found {len(sessions)} session(s) in this folder. Import all of them?"):
                return
        stride = simpledialog.askinteger(
            "Sample interval",
            "Import every Nth sample (recordings are ~3 per second, so\n"
            "neighbouring samples are often near-duplicates; 1 = all):",
            initialvalue=1, minvalue=1,
        )
        if not stride:
            return

        if self._bg_job:
            messagebox.showinfo("Busy", "Wait for the current import / publish to finish first.")
            return
        self._bg_job = True
        self.progress_var.set("Importing samples: 0")

        def progress(done, total):
            self.root.after(0, lambda: self.progress_var.set(f"Importing samples: {done}/{total}"))

        def worker():
            added = skipped = 0
            missing, no_depth = [], []
            try:
                for sess in sessions:
                    r = self.project.import_gimble_session(
                        sess, stride=stride, copy_lidar=COPY_LIDAR_ON_IMPORT, progress_cb=progress)
                    added += len(r['new_ids'])
                    skipped += r['skipped_existing']
                    missing += r['missing']
                    no_depth += r['no_depth']
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: messagebox.showerror("Import failed", err))
                self.root.after(0, lambda: self.progress_var.set(""))
                self.root.after(0, lambda: setattr(self, '_bg_job', False))
                return

            def done():
                self.progress_var.set("")
                self._bg_job = False
                self._refresh_image_list()
                if self._sessions_refresh:
                    self._sessions_refresh()
                self._update_status()
                lines = [f"Added {added} new sample(s) from {len(sessions)} session(s)."]
                if skipped:
                    lines.append(f"{skipped} were already in the project and were left alone.")
                if no_depth:
                    lines.append(f"{len(no_depth)} sample(s) came in without usable depth data.")
                if missing:
                    lines.append(f"{len(missing)} sample(s) listed in index.csv had no image file.")
                messagebox.showinfo("Import complete", "\n".join(lines))
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def import_video(self):
        """Pick one or more video files. Each becomes its own set: every Nth frame is saved as an image
        (see Project.import_video). Videos have no LiDAR depth, so these frames can be boxed and
        labeled but not given a distance. Re-importing only adds frames that aren't there yet."""
        paths = filedialog.askopenfilenames(
            title="Select video file(s)",
            filetypes=[("Video files", " ".join("*" + e for e in ac.Project.VIDEO_EXTS)), ("All files", "*.*")])
        if not paths:
            return
        paths = [os.path.normpath(p) for p in paths]
        try:
            n, fps, w, h = ac.Project.video_info(paths[0])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Can't read video", str(e))
            return
        secs = (n / fps) if fps > 0 and n > 0 else 0
        stride = simpledialog.askinteger(
            "Frame interval",
            f"{os.path.basename(paths[0])}: {w}x{h}, {fps:.1f} fps, {n} frames"
            + (f" ({secs:.0f} s)" if secs else "") + ".\n\n"
            "Keep one frame every N frames. Neighbouring frames are near-duplicates, so\n"
            f"{max(1, round(fps / 3)) if fps > 0 else 10} (about 3 frames per second) is a good start; 1 keeps every frame:",
            initialvalue=max(1, round(fps / 3)) if fps > 0 else 10, minvalue=1)
        if not stride:
            return
        names = {}
        for p in paths:
            default = self.project.find_video_session(p) or ac.Project.default_video_session_name(p)
            if len(paths) == 1:
                nm = simpledialog.askstring("Set name", "Name for this set of frames:", initialvalue=default)
                if not nm:
                    return
                default = nm.strip()
            names[p] = default
        if self._bg_job:
            messagebox.showinfo("Busy", "Wait for the current import / publish to finish first.")
            return
        self._bg_job = True
        self.progress_var.set("Importing video frames: 0")

        def progress(done, total):
            self.root.after(0, lambda: self.progress_var.set(f"Importing video frames: {done}/{total}"))

        def worker():
            added = skipped = 0
            made, errors = [], []
            for p in paths:
                try:
                    r = self.project.import_video(p, stride=stride, session_name=names[p], progress_cb=progress)
                    added += len(r['new_ids'])
                    skipped += r['skipped_existing']
                    made.append(r['session_id'])
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{os.path.basename(p)}: {e}")

            def done():
                self.progress_var.set("")
                self._bg_job = False
                self._refresh_image_list()
                if self._sessions_refresh:
                    self._sessions_refresh()
                self._update_status()
                lines = [f"Added {added} frame(s) in {len(made)} set(s)" + (f": {', '.join(made)}" if made else "") + "."]
                if skipped:
                    lines.append(f"{skipped} frame(s) were already in the project and were left alone.")
                if made:
                    lines.append("Pick the set from the Working set box on the right.")
                if errors:
                    lines.append("\nProblems:\n" + "\n".join(errors))
                (messagebox.showwarning if errors and not made else messagebox.showinfo)("Video import", "\n".join(lines))
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Sync (catch up on files deleted outside the app)
    # ------------------------------------------------------------------

    def sync_deleted_files(self):
        """This also runs automatically every time a project is opened
        (see AnnotatorApp.__init__), so restarting the app after cleaning
        up in File Explorer is enough on its own - this button is only
        for catching up without restarting, e.g. if you delete some
        unannotated images while the app is still open."""
        removed = self.project.reconcile_deleted_images()
        if removed:
            if self.current_image_id in removed:
                self.current_image_id = None
                self.current_pil_image = None
                self._set_boxes([])
                self.selected_box_index = None
            self._refresh_image_list()
            self._update_status()
            messagebox.showinfo("Sync complete", f"Cleaned up tracking for {len(removed)} deleted image(s).")
        else:
            messagebox.showinfo("Sync complete", "Nothing to clean up - every tracked image still exists on disk.")

    # ------------------------------------------------------------------
    # Model-assisted labeling
    #
    # Load Model once (checkpoint + confidence threshold), then check
    # whichever classes you actually want AI help with in the "AI Assist"
    # panel - e.g. just "person" while other classes don't have enough
    # training data yet to be worth reviewing. From then on, every time
    # you land on an unannotated image (Next/Prev/click) or toggle a
    # checkbox, ONLY the current photo gets (re-)predicted, filtered to
    # whatever's checked at that moment - nothing runs over the whole
    # list, and nothing waits on a batch job. Predictions are written via
    # Project.set_ai_predicted_boxes(), which deliberately never marks an
    # image "annotated" - you still have to fix it (any edit confirms it
    # the normal way) or click "Confirm Image". An already-confirmed image
    # is never touched by any of this again.
    # ------------------------------------------------------------------

    def _refresh_ai_choices(self, entries=None):
        """Fills the AI model box with the trained model slots that exist right now."""
        try:
            entries = entries if entries is not None else mr.discover(RUNS_DIR)
        except Exception:  # noqa: BLE001
            entries = []
        self._ai_entries = {e['label']: e for e in entries}
        for label, e in self._inserted.items():
            self._ai_entries.setdefault(label, e)
        self.ai_choice_box.configure(values=AI_CHOICES + [e['label'] for e in entries]
                                     + [l for l in sorted(self._inserted) if l not in {e['label'] for e in entries}])

    def _identify_old_models_in_background(self):
        """An older single-folder run (helper / gimble_model) has to be opened once to learn its size; do that off
        the GUI thread, then show it in the box."""
        def worker():
            try:
                entries = mr.discover(RUNS_DIR, detect_legacy=True)
            except Exception:  # noqa: BLE001
                return
            self.root.after(0, lambda: self._refresh_ai_choices(entries))
        threading.Thread(target=worker, daemon=True).start()

    def load_ai_model(self, kind=None):
        """Loads the model picked in the toolbar box: the Original helper (never modified), or any trained slot
        (Nano/Small/Medium/Large helper or scratch). Switching re-predicts the current photo, so you can flip
        between models on the same frame and compare. A slot is re-read automatically whenever training has
        replaced its best.pt."""
        choice = kind or self.ai_choice_var.get()
        self.ai_choice_var.set(choice)
        self.model_assist = None
        self._ai_kind = None
        if choice == AI_CHOICES[0]:
            self.ai_model_status_var.set("No model loaded")
            return
        entry = None
        if choice != AI_CHOICES[1]:
            self._refresh_ai_choices()
            entry = self._ai_entries.get(choice)
            if entry is None:
                self.ai_choice_var.set(AI_CHOICES[0])
                self.ai_model_status_var.set("No model loaded")
                messagebox.showerror("Model not found", f"'{choice}' doesn't exist (any more) under:\n\n{RUNS_DIR}")
                return
        path = entry['path'] if entry else BACKUP_PATH
        if not os.path.exists(path):
            self.ai_choice_var.set(AI_CHOICES[0])
            self.ai_model_status_var.set("No model loaded")
            messagebox.showerror(
                "No backup checkpoint found",
                f"Couldn't find the original helper model at:\n\n{path}\n\n"
                "Run update_model_backup.py first (after a training run has finished) to create one.")
            return
        try:
            if entry and entry['kind'] == 'scratch':
                import gimble_assist
            else:
                import model_assist
                if entry:
                    import gimble_assist
        except ImportError as e:
            self.ai_choice_var.set(AI_CHOICES[0])
            self.ai_model_status_var.set("No model loaded")
            if au.is_frozen():
                messagebox.showerror(
                    "AI models not included",
                    "This is the Lite version of the annotator, which doesn't include the AI parts.\n\n"
                    "To pre-label with an AI model, download GeneralClassificationAnnotator-AI.zip from the "
                    f"releases page and use that one instead:\n{au.RELEASES_URL}")
            else:
                messagebox.showerror(
                    "Missing package",
                    f"{e}\n\nModel-assisted labeling needs ultralytics and torch:\n\n    pip install ultralytics")
            return

        self.ai_model_status_var.set("Loading model...")
        kind_now = choice

        def worker():
            try:
                if entry and entry['kind'] == 'scratch':
                    assist = gimble_assist.GimbleAssist(path)
                elif entry:
                    assist = gimble_assist.HelperAssist(path)
                else:
                    assist = model_assist.ModelAssist(path)
                error = None
            except Exception as e:  # noqa: BLE001
                assist, error = None, str(e)

            def done():
                if self.ai_choice_var.get() != kind_now:
                    return                      # the user already picked something else
                if error:
                    self.ai_choice_var.set(AI_CHOICES[0])
                    self.ai_model_status_var.set("No model loaded")
                    messagebox.showerror("Couldn't load checkpoint", error)
                    return
                self.model_assist = assist
                self._ai_kind = kind_now
                self._refresh_ai_status()
                if self.current_image_id is not None:
                    self._trigger_live_predict(self.current_image_id)
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_ai_status(self):
        if self.model_assist is None:
            self.ai_model_status_var.set("No model loaded")
            return
        kind = getattr(self, '_ai_kind', None)
        if kind == AI_CHOICES[1]:
            desc = "original: " + _describe_backup()
        else:
            desc = f"{(kind or 'model').lower()}: " + self.model_assist.describe()
        self.ai_model_status_var.set(f"{desc} (conf {self.model_conf:.2f})")

    def set_ai_confidence(self):
        conf = simpledialog.askfloat(
            "Confidence threshold",
            "Only keep predictions at or above this confidence (0-1).\n"
            "Lower catches more but with more false positives:",
            initialvalue=self.model_conf, minvalue=0.0, maxvalue=1.0)
        if conf is None:
            return
        self.model_conf = conf
        self._refresh_ai_status()
        if self.model_assist is not None and self.current_image_id is not None:
            self._trigger_live_predict(self.current_image_id)

    def _map_model_classes(self, boxes):
        """The helper model numbers its classes its own way; the project's list can change. Match by
        NAME and drop predictions of classes the project doesn't have (any more)."""
        names = getattr(self.model_assist, 'class_names', None)
        if not names:
            return boxes
        out = []
        for (c, xc, yc, w, h) in boxes:
            if 0 <= c < len(names) and names[c] in self.project.classes:
                out.append((self.project.classes.index(names[c]), xc, yc, w, h))
        return out

    def _selected_ai_classes(self):
        return {i for i, var in self.ai_class_vars.items() if var.get()}

    def _on_ai_assist_toggle(self):
        if self.current_image_id is not None:
            self._trigger_live_predict(self.current_image_id)

    def _trigger_live_predict(self, img_id):
        """Entry point for (re-)predicting exactly one image - called on
        every navigation and every AI-Assist checkbox change. Only one
        prediction ever runs at a time (the model isn't safe to hit
        concurrently, and it avoids piling up GPU work); if a request
        comes in while one's already running, it replaces any previously
        queued request rather than stacking up, so you always end up with
        a fresh result for wherever you actually are, not a backlog."""
        if self.model_assist is None:
            return
        if self.project.is_annotated(img_id):
            return  # confirmed - human ground truth, never touched again

        if self._predict_busy:
            self._predict_queued_img_id = img_id
            return
        self._run_predict_now(img_id)

    def _run_predict_now(self, img_id):
        selected = self._selected_ai_classes()

        if not selected:
            # nothing checked right now - if this image is showing an
            # earlier AI draft, clear it back to empty so what's on
            # screen always matches what's actually checked. Never
            # touches boxes you drew yourself (those confirmed the image
            # already, so _trigger_live_predict returned above).
            if self.project.is_ai_pending(img_id):
                self.project.set_ai_predicted_boxes(img_id, [])
                if img_id == self.current_image_id:
                    self._set_boxes([])
                    self.selected_box_index = None
                    self._redraw()
                    self._update_status()
                self._refresh_image_list()
            self._advance_predict_queue()
            return

        self._predict_busy = True
        self.progress_var.set(f"AI predicting: {img_id}")

        def worker():
            try:
                reloaded = bool(getattr(self.model_assist, 'reload_if_changed', lambda: False)())
                boxes = self.model_assist.predict_boxes(
                    self.project.image_path(img_id), conf_threshold=self.model_conf)
                boxes = self._map_model_classes(boxes)
                boxes = [b for b in boxes if b[0] in selected]
                error = None
            except Exception as e:
                boxes = None
                error = str(e)
                reloaded = False

            def done():
                self._predict_busy = False
                self.progress_var.set("")
                if reloaded:
                    self._refresh_ai_status()
                if error:
                    messagebox.showerror("Prediction failed", error)
                else:
                    self.project.set_ai_predicted_boxes(img_id, boxes)
                    if img_id == self.current_image_id:
                        self._set_boxes(boxes)
                        self.selected_box_index = None
                        self._redraw()
                        self._update_status()
                    self._refresh_image_list()
                self._advance_predict_queue()
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _advance_predict_queue(self):
        nxt = self._predict_queued_img_id
        self._predict_queued_img_id = None
        if nxt is not None:
            self._trigger_live_predict(nxt)

    def clear_and_retry_ai(self):
        """Wipes the current image's boxes and sends it straight back
        through the live AI-assist path for a fresh prediction - the
        point being that a newer checkpoint might catch things an
        earlier, weaker one missed, so a confirmed or drafted annotation
        is never permanently stuck with whatever an old model happened
        to produce.

        For a CONFIRMED image, reuses Project.unannotate_image() (moves
        the file back to the pending queue, wipes its label file, cleans
        up its now-stale distorted variants - see that method's own
        docstring) so there's exactly one way an image resets to
        "unreviewed," whether you're backing out a mistake or upgrading
        to a better model. unannotate_image() no-ops on an image that
        was never confirmed in the first place, so an unconfirmed AI
        draft is cleared directly instead - the same blank-out
        _run_predict_now already does when nothing's checked anymore.

        Confirms first only when this would discard a CONFIRMED
        annotation (real ground truth) - never needed for an
        unconfirmed draft, since nothing was ever committed to for it."""
        img_id = self.current_image_id
        if img_id is None:
            return

        if self.project.is_annotated(img_id):
            n = len(self.boxes)
            if not messagebox.askyesno(
                "Clear confirmed annotation?",
                f"'{img_id}' is confirmed annotated with {n} box(es). Clear it and "
                "send it back for a fresh AI pass?\n\nThis can't be undone."
            ):
                return
            self.project.unannotate_image(img_id)
        elif self.boxes or self.project.is_ai_pending(img_id):
            self.project.set_ai_predicted_boxes(img_id, [])

        self._set_boxes([])
        self.selected_box_index = None
        self._redraw()
        self._refresh_image_list()
        self._update_status()

        # only actually predicts anything if a model's loaded AND at
        # least one AI Assist class is currently checked - same gate
        # every other navigation/toggle trigger goes through, see
        # _trigger_live_predict
        self._trigger_live_predict(img_id)

    def confirm_current_as_is(self):
        """Accepts the current image's boxes exactly as they stand - the
        normal path when the AI's draft already looks right and there's
        nothing to fix. Just reuses _save_current_boxes, so it's
        indistinguishable from a human having drawn/edited these boxes:
        ai_pending clears, the image counts as annotated, and it moves to
        images/annotated/ like anything else."""
        if self.current_image_id is None:
            return
        self._save_current_boxes()
        self._redraw()

    def _update_status(self):
        if self.current_image_id is None:
            n = len(self.project.list_images())
            hint = "" if n else "  |  empty - File > Import Gimble Session... or Import Video..."
            if n and self.active_session is not None:
                hint = f"  |  set {self.active_session}: nothing to show"
            self.status_var.set(f"Project: {self.project.root}  |  {n} image(s){hint}")
            return
        if self.project.is_annotated(self.current_image_id):
            status = "annotated"
        elif self.project.is_ai_pending(self.current_image_id):
            status = "AI pre-labeled - needs review"
        else:
            status = "not yet annotated"
        info = self.project.manifest['images'].get(self.current_image_id, {})
        depth_txt = (f"{info.get('depth_pixels', 0)} LiDAR pts" if self.project.has_depth(self.current_image_id)
                     else "no depth")
        self.status_var.set(
            f"{self.current_image_id}  |  {len(self.boxes)} box(es) ({sum(1 for d in self.box_dist if d)} with distance)  |  {status}  |  {depth_txt}  |  "
            f"{len(self.project.list_images())} image(s) in project"
            + (f"  |  set: {self.active_session}" if self.active_session else "")
        )


def _same(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def pick_project(argv, cfg, here):
    """Which project folder to open -> (path, note). Order: a folder given on the command line; the one remembered
    FOR THIS COPY of the program (so a second copy in another folder starts fresh instead of borrowing the first
    one's data); otherwise ./Dataset next to the program, made on first use. An older ./GimbleDataset is renamed
    to Dataset (everything inside stays)."""
    new, old = os.path.join(here, "Dataset"), os.path.join(here, "GimbleDataset")
    if len(argv) > 1:
        return argv[1], None
    mem = (cfg.get('projects') or {}).get(os.path.normcase(os.path.abspath(here)))
    if mem and os.path.isdir(mem):
        return mem, None
    if os.path.isdir(new):
        return new, None
    if os.path.isdir(old):
        try:
            os.rename(old, new)
            return new, ("Your project folder 'GimbleDataset' was renamed to 'Dataset'. Nothing inside it changed.")
        except OSError:
            return old, None                                  # in use / not allowed: keep working in the old name
    return new, None                                          # first time: a new empty project is made here


def selftest(argv):
    """python annotator_gui.py --selftest out.txt [--expect-ai]
    Builds the window on a throw-away project, tries the AI imports, writes what happened to out.txt (a packaged
    program has no console) and returns 0 on success, 1 on failure. Never touches your real settings or project."""
    import tempfile
    i = argv.index("--selftest")
    out = argv[i + 1] if len(argv) > i + 1 and not argv[i + 1].startswith("--") else None
    expect_ai = "--expect-ai" in argv
    lines, ok = [f"version {au.APP_VERSION}", f"packaged {au.is_frozen()}", f"app folder {au.app_dir(__file__)}"], True
    tmp = tempfile.mkdtemp(prefix="gca_selftest_")
    try:
        sz.CONFIG_PATH = os.path.join(tmp, "cfg.json")                 # never the real ~/.trash_annotator.json
        root = Tk()
        root.withdraw()
        app = AnnotatorApp(root, os.path.join(tmp, "proj"))
        root.update()
        labels = [app.file_menu.entryconfigure(k, "label")[-1] for k in range(app.file_menu.index("end") + 1)
                  if app.file_menu.type(k) == "command"]
        lines.append("window built; File menu: " + ", ".join(labels))
        if "Check for updates..." not in labels:
            ok = False
            lines.append("FAIL: no 'Check for updates...' entry")
        root.destroy()
    except Exception as e:  # noqa: BLE001
        ok = False
        lines.append(f"FAIL: window: {type(e).__name__}: {e}")
    ai_ok = True
    for mod in ("torch", "ultralytics", "gimble_assist", "model_assist"):
        try:
            __import__(mod)
            lines.append(f"import {mod}: ok")
        except Exception as e:  # noqa: BLE001
            ai_ok = False
            lines.append(f"import {mod}: not available ({type(e).__name__}: {e})")
    if expect_ai and not ai_ok:
        ok = False
        lines.append("FAIL: --expect-ai was given but the AI packages did not import")
    elif not expect_ai and not ai_ok:
        lines.append("(AI parts not included - fine for the Lite version)")
    lines.append("SELFTEST " + ("OK" if ok else "FAILED"))
    text = "\n".join(lines) + "\n"
    if out:
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass
    try:
        print(text)
    except Exception:  # noqa: BLE001  (no console in a packaged window program)
        pass
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest(sys.argv))
    root = Tk()
    if HAVE_SV_TTK:
        sv_ttk.set_theme("dark")
    cfg = sz.load_config()
    here = au.app_dir(__file__)
    project_root, note = pick_project(sys.argv, cfg, here)
    os.makedirs(project_root, exist_ok=True)
    cfg.pop('project_dir', None)                              # (older versions remembered one folder for every copy)
    cfg.setdefault('projects', {})[os.path.normcase(os.path.abspath(here))] = os.path.normpath(project_root)
    try:
        sz.save_config(cfg)
    except OSError:
        pass
    if note:
        messagebox.showinfo("Project folder", note)
    app = AnnotatorApp(root, project_root)
    root.mainloop()


if __name__ == "__main__":
    main()
