# Changelog

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
