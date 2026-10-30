# -*- mode: python ; coding: utf-8 -*-
# AI build: a FOLDER (too big for one file) with CPU torch + ultralytics, for pre-labeling with the models.
#     pyinstaller packaging/annotator-ai.spec --noconfirm
# Result: dist/GeneralClassificationAnnotator-AI/  (zipped for the release)
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")
ICON = os.path.join(ROOT, "assets", "icon.ico")

ul_datas, ul_bins, ul_hidden = collect_all("ultralytics")
a = Analysis(
    [os.path.join(SRC, "annotator_gui.py")],
    pathex=[SRC],
    binaries=ul_bins,
    datas=collect_data_files("sv_ttk") + ul_datas,
    hiddenimports=ul_hidden + ["annotator_core", "app_update", "hotkeys", "share_zip", "model_library",
                               "model_registry", "model_assist", "gimble_assist", "gimble_data", "gimble_model"],
    excludes=["IPython", "pytest", "tensorboard"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="GeneralClassificationAnnotator-AI",
    console=False,
    icon=ICON if os.path.exists(ICON) else None,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="GeneralClassificationAnnotator-AI", upx=False)
