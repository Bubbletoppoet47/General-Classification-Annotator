# -*- mode: python ; coding: utf-8 -*-
# The program: a launcher .exe plus a FOLDER of libraries (CPU torch + ultralytics, so pre-labeling works).
# The program's own code is NOT packed into the .exe: packaging/make_release.py puts it as plain .py files in an
# `app` folder next to the .exe, so an update only has to swap that small folder (see src/app_update.py).
# The ready-made models are not inside either; the program downloads them from GitHub when you ask.
#     pyinstaller packaging/annotator.spec --noconfirm
#     python packaging/make_release.py dist/GeneralClassificationAnnotator release-files
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")
ICON = os.path.join(ROOT, "assets", "icon.ico")

# our own modules: analysed (so the libraries they use get packed) but kept OUT of the .exe
OURS = ["annotator_gui", "annotator_core", "app_update", "hotkeys", "share_zip", "model_library", "model_registry",
        "model_download", "model_assist", "gimble_assist", "gimble_data", "gimble_model", "web_mode", "web_page"]

try:
    ul_datas, ul_bins, ul_hidden = collect_all("ultralytics")
except Exception:                      # building the launcher without the AI packages (tests only)
    ul_datas, ul_bins, ul_hidden = [], [], []
try:                                   # torchvision's compiled part (the NMS operator) must come along
    tv_datas, tv_bins, tv_hidden = collect_all("torchvision")
except Exception:
    tv_datas, tv_bins, tv_hidden = [], [], []
a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[SRC],
    binaries=ul_bins + tv_bins,
    datas=collect_data_files("sv_ttk") + ul_datas + tv_datas,
    hiddenimports=ul_hidden + tv_hidden + OURS + ["qrcode", "PIL.ImageTk", "tkinter", "tkinter.ttk", "tkinter.filedialog",
                                      "tkinter.messagebox", "tkinter.simpledialog"],
    excludes=["IPython", "pytest", "tensorboard"],
)
a.pure[:] = [t for t in a.pure if t[0] not in OURS]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="GeneralClassificationAnnotator",
    console=False,
    icon=ICON if os.path.exists(ICON) else None,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="GeneralClassificationAnnotator", upx=False)
