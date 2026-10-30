# -*- mode: python ; coding: utf-8 -*-
# Lite build: ONE .exe, no AI parts (no torch / ultralytics). Run from the repo root:
#     pyinstaller packaging/annotator-lite.spec --noconfirm
# Result: dist/GeneralClassificationAnnotator-Lite(.exe)
import os
from PyInstaller.utils.hooks import collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")
ICON = os.path.join(ROOT, "assets", "icon.ico")          # used automatically when this file exists

a = Analysis(
    [os.path.join(SRC, "annotator_gui.py")],
    pathex=[SRC],
    datas=collect_data_files("sv_ttk"),                   # the dark/light theme files
    hiddenimports=["annotator_core", "app_update", "hotkeys", "share_zip", "model_library", "model_registry",
                   "model_assist", "gimble_assist", "gimble_data", "gimble_model"],
    excludes=["torch", "torchvision", "torchaudio", "ultralytics", "matplotlib", "pandas", "scipy",
              "IPython", "pytest", "tensorboard"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="GeneralClassificationAnnotator-Lite",
    console=False,
    icon=ICON if os.path.exists(ICON) else None,
    upx=False,
)
