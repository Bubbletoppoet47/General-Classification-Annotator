# Changelog

## 1.2.2 - 2026-10-31

- File > Dataset location... (was under Manage): link a dataset folder anywhere, nothing copied. Remembered per computer.
- First start asks you to link or make a dataset instead of creating `Dataset` next to the program.

## 1.2.1 - 2026-10-31

- Fix: "operator torchvision::nms does not exist" when predicting in the Windows program (torchvision installed from
  the same CPU build as torch and its compiled part packed; the self-test now runs the NMS operator).

## 1.2.0 - 2026-10-31

- Small automatic updates for the Windows program: the program's code is now plain `.py` files in an `app` folder beside a
  launcher `.exe`; File > Check for updates downloads only that (about 1 MB), swaps it, keeps the old one in `_previous` and
  restarts. A version that fails to start is rolled back automatically. If the libraries change, the full download is needed.
- Each release now also has `GeneralClassificationAnnotator-code.zip` and `update-manifest.json`.

## 1.1.0 - 2026-10-31

- Tablet / browser mode: File > Open in browser (tablet)... (address, QR code, random 6-digit code per start).
  Pen draws and edits boxes, fingers pan and zoom, Undo / Delete / AI / Publish / Confirm. No LiDAR in the web page.
- Shorter dialog messages and side-panel help text.

## 1.0.2 - 2026-10-29

- One Windows program instead of Lite and AI: `GeneralClassificationAnnotator.zip`.
- Ready-made AI models are downloaded from the project's GitHub `models` release instead of being bundled. A first-start
  question offers the newest of each size, all of them, or none; File > Download models from GitHub does it any time.

## 1.0.1 - 2026-10-29

- New logo and app icon (sorted G, C, A boxes). Used by the Windows programs and the program window.

## 1.0.0 - 2026-10-29

- First public release on GitHub (GPL-3.0).
- Windows programs: Lite (one .exe) and AI (folder with CPU PyTorch), built by GitHub Actions.
- File > Check for updates in the Windows program asks GitHub for the newest release and opens the download page.
  Running from source still updates from an `Annotator_Share` folder.
- `--selftest` option for checking a built program.
- Everything before this (sharing, publish, models folder, hotkeys, zoom, class delete fix) is described in
  `src/UPDATES.txt`.
