# ─────────────────────────────────────────────
#  gui/icons.py  —  Centralized Icon Loader
# ─────────────────────────────────────────────
"""
Provides QIcon objects loaded from assets/icons/*.svg.
All paths are resolved relative to this file so the app works
regardless of the current working directory.
"""
import os
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

# Resolve the icons directory (../../assets/icons relative to this file)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ICONS_DIR = os.path.join(_HERE, "..", "assets", "icons")


def _icon(name: str) -> QIcon:
    """Load an SVG icon by base name (without extension)."""
    path = os.path.join(_ICONS_DIR, f"{name}.svg")
    if not os.path.exists(path):
        return QIcon()
    return QIcon(path)


# ── Pre-loaded icon singletons ─────────────────
# Loaded lazily on first access via module-level constants after import.

def get(name: str) -> QIcon:
    """Convenient accessor: get a QIcon by name."""
    return _icon(name)


# ── Named shortcuts ────────────────────────────
APP     = lambda: _icon("app")
CUT     = lambda: _icon("cut")
BLEND   = lambda: _icon("blend")
LOAD    = lambda: _icon("load")
UNDO    = lambda: _icon("undo")
CLEAR   = lambda: _icon("clear")
SAVE    = lambda: _icon("save")
BROWSE  = lambda: _icon("browse")
PREVIEW = lambda: _icon("preview")
OK_IMG  = lambda: _icon("ok_image")
PATCH   = lambda: _icon("patch")
MASK    = lambda: _icon("mask")
AUGMENT = lambda: _icon("augment")
