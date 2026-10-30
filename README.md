<p align="center"><img src="assets/logo.png" width="128" alt="General Classification Annotator logo"></p>

# General Classification Annotator

A desktop tool for annotating photos for object detection. It imports sessions from the Gimble rig (a LiDAR plus a
camera) and videos, lets you draw boxes, pick LiDAR distances, pre-label with AI models, and publish the finished
images to a shared folder so several people can annotate independently and the data can be collected for training.

Licensed under GPL-3.0 (see `LICENSE`).

## Get it (Windows)

Download from the [Releases page](https://github.com/Bubbletoppoet47/General-Classification-Annotator/releases/latest):

- **GeneralClassificationAnnotator.zip**: the program. Unzip it and run `GeneralClassificationAnnotator.exe` inside the
  folder. It is large (about 280 MB) because it contains PyTorch for the AI pre-labeling.

The ready-made AI models are not inside it. The first time it starts it asks whether you want to download them from
GitHub (just the newest of each size, or all of them, or none). You can do it later from **File > Download models from
GitHub**. They are small, a few MB each.

The first time you run it, Windows may show a blue "Windows protected your PC" box, because the program is not
code-signed. Choose **More info**, then **Run anyway**.

Your project (the `Dataset` folder) is created next to the program the first time it runs. Put the program in a folder
you own, for example `C:\Annotator`, not in Program Files.

## Updating

- **Windows program:** File > Check for updates asks GitHub for the newest release and opens its download page. Download
  the new file, close the program, and run the new one. Your project and settings are not touched.
- **Running from source:** File > Check for updates compares the program files with an `Annotator_Share` folder (File >
  Update folder location picks it), keeps the old files in `_previous`, replaces them, and reopens. Or just `git pull`.

## Run from source

Python 3.10 or newer (Tkinter comes with the normal Windows installer).

```
pip install -r src/requirements.txt        # everything, including the AI parts
python src/annotator_gui.py
```

For an NVIDIA GPU, install PyTorch first
from https://pytorch.org/get-started/locally/ and then run the line above.

`python src/annotator_gui.py --selftest out.txt` builds the window on a throw-away project, tries the AI imports, writes
the result to `out.txt` and exits 0 or 1. (The built programs have no console, so that is how they are checked.)

## Sharing

Published images go into a shared folder, normally a synced Google Drive folder you share with the people who annotate:

```
<shared>/<Name>/index.json
<shared>/<Name>/images/<uid>.jpg
<shared>/<Name>/depth/<uid>.png
```

Publishing only copies what is missing or changed and never deletes anything. Downloading this program gives no access
to anyone's data; access is whoever the Drive folder is shared with. Set the folder with File > Shared folder location.
Models live in a Models folder with `Assist` and `Scratch` inside (File > Models folder location).

## Build the Windows program

GitHub builds it: Actions tab, **Build Windows programs**, Run workflow. Leave **publish** unticked for a trial build
(files are kept on the run page), tick it to also create the release `v<version>`. The version is `APP_VERSION` in
`src/app_update.py`; raise it before each release and add a dated entry on top of `src/UPDATES.txt` (its newest entry
becomes the release notes and the "what's new" text).

To build by hand: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then
`pip install -r src/requirements.txt -r requirements-build.txt` and `pyinstaller packaging/annotator.spec`.

## Tests

```
python tests/test_gimble_zoom.py            # each test prints an OK line and raises on failure
```

Some tests build a window (they need a display; on a headless Linux box use `xvfb-run -a python tests/<name>.py`).
Tests that need PyTorch or sample photos print `SKIPPED` when those are missing.
