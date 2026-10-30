"""
hotkeys.py - the annotator's keyboard shortcuts and how they are changed.

No tkinter in here (so it is easy to test): a shortcut is stored as a short text such as
"d", "Delete", "Left", "5", "Ctrl+z" or "Shift+d"; Hotkeys keeps only what differs from the defaults and
the annotator saves that next to the project (hotkeys.json). One key can only do one thing: giving a key
that is already in use to another action takes it away from the first one.

Class shortcuts follow the class's POSITION in the list ("class:0" = the first class), the same way the
number keys always did - 1-9 for the first nine classes, 0 for the tenth, none for the rest until you give them one.
"""

ACTIONS = [
    ("delete_box", "Delete selected box", "Delete"),
    ("toggle_pick", "LiDAR dots + pick distance (on/off)", "d"),
    ("toggle_depth", "LiDAR dots + pick distance (second key)", "p"),
    ("prev_image", "Previous image", "Left"),
    ("next_image", "Next image", "Right"),
    ("fit_view", "Fit image to screen (undo zoom)", "f"),
]

MODIFIER_KEYSYMS = {
    "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Meta_L", "Meta_R", "Super_L", "Super_R",
    "Win_L", "Win_R", "Caps_Lock", "Num_Lock", "Scroll_Lock", "ISO_Level3_Shift", "Mode_switch", "??", "",
}

_SHIFT, _CONTROL = 0x1, 0x4

_NICE = {"Left": "←", "Right": "→", "Up": "↑", "Down": "↓", "space": "Space",
         "BackSpace": "Backspace", "Return": "Enter", "Prior": "Page Up", "Next": "Page Down"}


def default_class_key(idx):
    if idx < 9:
        return str(idx + 1)
    if idx == 9:
        return "0"
    return None


def from_event(keysym, state=0):
    """The stored text for a key press (a Tk event's keysym + state), or None for a bare modifier key."""
    if keysym in MODIFIER_KEYSYMS:
        return None
    shift = bool(state & _SHIFT)
    ctrl = bool(state & _CONTROL)
    key = keysym
    if len(key) == 1 and key.isalpha():
        key = key.lower()               # caps lock alone must not change which key it is
    mods = []
    if ctrl:
        mods.append("Ctrl")
    if shift and (len(keysym) > 1 or keysym.isalpha()):
        mods.append("Shift")            # Shift + a letter / F-key / arrow; for 1 -> ! the key itself already says it
    return "+".join(mods + [key])


def split(binding):
    parts = binding.split("+")
    # a "+" key itself would split into empty parts; the last non-empty part is the key
    key = parts[-1] if parts[-1] else "plus"
    return [p for p in parts[:-1] if p], key


def to_tk(binding):
    """Tk event sequence for a stored shortcut, e.g. 'Ctrl+d' -> '<Control-Key-d>'."""
    mods, key = split(binding)
    tk = []
    if "Ctrl" in mods:
        tk.append("Control")
    if "Shift" in mods:
        if len(key) == 1 and key.isalpha():
            key = key.upper()           # Tk tells Shift+d from d by the keysym itself
        else:
            tk.append("Shift")
    tk += ["Key", key]
    return "<" + "-".join(tk) + ">"


def label(binding):
    """What to show for a shortcut ('' when there is none)."""
    if not binding:
        return ""
    mods, key = split(binding)
    k = _NICE.get(key) or (key.upper() if len(key) == 1 and key.isalpha() else key)
    return "+".join(mods + [k])


class Hotkeys:
    def __init__(self, overrides=None):
        self.overrides = {}
        for k, v in (overrides or {}).items():
            if isinstance(k, str) and isinstance(v, str):
                self.overrides[k] = v

    # ------------------------------------------------------------ what the keys are
    @staticmethod
    def default(action):
        if action.startswith("class:"):
            try:
                return default_class_key(int(action[6:]))
            except ValueError:
                return None
        for a, _title, key in ACTIONS:
            if a == action:
                return key
        return None

    def get(self, action):
        if action in self.overrides:
            return self.overrides[action] or None
        return self.default(action)

    def label(self, action):
        return label(self.get(action))

    def actions(self, class_names=()):
        """[(action, title)] in window order: the fixed actions, then one per class."""
        out = [(a, t) for a, t, _k in ACTIONS]
        out += [(f"class:{i}", f"Class {i + 1}: {n}") for i, n in enumerate(class_names)]
        return out

    def bindings(self, n_classes):
        """{tk sequence: action} for everything that currently has a key."""
        out = {}
        for action, _t in self.actions([""] * n_classes):
            b = self.get(action)
            if b:
                out[to_tk(b)] = action
        return out

    # ------------------------------------------------------------ changing them
    def holders(self, binding, n_classes, except_action=None):
        return [a for a, _t in self.actions([""] * n_classes) if a != except_action and self.get(a) == binding]

    def set(self, action, binding, n_classes):
        """Gives `action` this key (None / '' = no key). Any other action that had the key loses it.
        Returns the actions that lost it."""
        lost = []
        if binding:
            for other in self.holders(binding, n_classes, except_action=action):
                self._store(other, None)
                lost.append(other)
        self._store(action, binding)
        return lost

    def _store(self, action, binding):
        if (binding or None) == self.default(action):
            self.overrides.pop(action, None)
        else:
            self.overrides[action] = binding or ""

    def reset(self, action, n_classes):
        return self.set(action, self.default(action), n_classes)

    def reset_all(self):
        self.overrides = {}

    def to_dict(self):
        return dict(self.overrides)
