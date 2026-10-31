"""
launcher.py - what the packaged .exe actually starts.

The program's own code is NOT inside the .exe. It sits as plain .py files in the `app` folder next to the .exe, so an
update only has to swap that small folder (the big runtime - Python, PyTorch ... - stays as it is).

If a freshly updated version fails to start, this puts the previous `app` folder back (kept in _previous/) and starts
that one instead. Nothing here imports the program's own code until the app folder is on the path.
"""

import os
import shutil
import sys

PENDING = ".pending"          # in app/: "0" = just updated, never started; "1" = a start is in progress / crashed


def exe_dir():
    return os.path.dirname(os.path.abspath(sys.executable))


def restore_previous(base):
    """Puts the newest folder from _previous/ back as `app` (the broken one is kept as app_failed). True if it did."""
    prev = os.path.join(base, "_previous")
    if not os.path.isdir(prev):
        return False
    olds = sorted(d for d in os.listdir(prev) if d.startswith("app_") and d != "app_failed" and os.path.isdir(os.path.join(prev, d)))
    if not olds:
        return False
    app = os.path.join(base, "app")
    failed = os.path.join(prev, "app_failed")
    shutil.rmtree(failed, ignore_errors=True)
    if os.path.isdir(app):
        os.replace(app, failed)
    os.replace(os.path.join(prev, olds[-1]), app)
    return True


def check_pending(base):
    """A just-updated version gets exactly one try: marker 0 -> 1 on start; still 1 on the next start = it crashed."""
    marker = os.path.join(base, "app", PENDING)
    try:
        with open(marker) as f:
            state = f.read().strip()
    except OSError:
        return False
    if state == "1":
        return restore_previous(base)
    with open(marker, "w") as f:
        f.write("1")
    return False


def _clear_marker(base):
    try:
        os.remove(os.path.join(base, "app", PENDING))
    except OSError:
        pass


def _purge_modules(app):
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None) or ""
        if f and os.path.abspath(f).startswith(os.path.abspath(app)):
            del sys.modules[name]


def _error_box(text):
    try:
        import tkinter
        from tkinter import messagebox
        r = tkinter.Tk()
        r.withdraw()
        messagebox.showerror("General Classification Annotator", text)
        r.destroy()
    except Exception:  # noqa: BLE001
        pass


def main():
    base = exe_dir()
    app = os.path.join(base, "app")
    restored = False
    try:
        restored = check_pending(base)
    except OSError:
        pass
    if app not in sys.path:
        sys.path.insert(0, app)
    try:
        import annotator_gui
    except Exception as e:  # noqa: BLE001
        if not restored and restore_previous(base):
            _purge_modules(app)
            try:
                import annotator_gui
                restored = True
            except Exception as e2:  # noqa: BLE001
                _error_box(f"The program couldn't start: {e2}\nPlease download the full zip again.")
                raise
        else:
            _error_box(f"The program couldn't start: {e}\nPlease download the full zip again.")
            raise
    if restored:
        try:
            annotator_gui.UPDATE_ROLLED_BACK = True
        except Exception:  # noqa: BLE001
            pass
    annotator_gui.main()
    _clear_marker(base)           # it ran and closed normally


if __name__ == "__main__":
    main()
