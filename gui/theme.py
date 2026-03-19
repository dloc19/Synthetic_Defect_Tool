# ─────────────────────────────────────────────
#  gui/theme.py  —  Centralized Dark Theme
# ─────────────────────────────────────────────

# ── Color Palette ─────────────────────────────
BG_DEEP     = "#0d1117"   # main window background
BG_BASE     = "#161b22"   # card / panel background
BG_SURFACE  = "#21262d"   # elevated surface (toolbar, sidebar)
BG_HOVER    = "#30363d"   # hover state background

BORDER      = "#30363d"   # default border
BORDER_MUTED= "#21262d"   # subtle border

ACCENT      = "#58a6ff"   # primary blue accent
ACCENT_DARK = "#1f6feb"   # darker blue (hover)
SUCCESS     = "#3fb950"   # green
SUCCESS_DARK= "#2ea043"
WARNING     = "#d29922"   # amber / yellow
WARNING_DARK= "#bb8009"
DANGER      = "#f85149"   # red
DANGER_DARK = "#da3633"
PURPLE      = "#bc8cff"   # purple
PURPLE_DARK = "#a371f7"

TEXT_PRIMARY   = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
TEXT_MUTED     = "#484f58"

# ── Global Application Stylesheet ─────────────
GLOBAL_QSS = f"""
/* ---- Base ---- */
QWidget {{
    background-color: {BG_DEEP};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background-color: {BG_DEEP};
}}

/* ---- Scroll Area ---- */
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background: {BG_BASE};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BG_HOVER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {BG_BASE};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {BG_HOVER};
    border-radius: 4px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ---- Frame / Separator ---- */
QFrame[frameShape="4"],  /* HLine */
QFrame[frameShape="5"]   /* VLine */
{{
    color: {BORDER};
    max-height: 1px;
}}

/* ---- Label ---- */
QLabel {{
    background: transparent;
    color: {TEXT_PRIMARY};
}}

/* ---- ToolBar ---- */
QToolBar {{
    background-color: {BG_SURFACE};
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 4px 8px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
    color: {TEXT_PRIMARY};
    font-size: 12px;
    font-weight: 600;
}}
QToolBar QToolButton:hover {{
    background-color: {BG_HOVER};
}}
QToolBar QToolButton:pressed {{
    background-color: {BORDER};
}}

/* ---- Status Bar ---- */
QStatusBar {{
    background-color: {BG_SURFACE};
    border-top: 1px solid {BORDER};
    color: {TEXT_SECONDARY};
    font-size: 11px;
    padding: 2px 8px;
}}
QStatusBar::item {{
    border: none;
}}

/* ---- MessageBox ---- */
QMessageBox {{
    background-color: {BG_BASE};
}}
QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
}}

/* ---- FileDialog ---- */
QFileDialog {{
    background-color: {BG_BASE};
}}

/* ---- Progress Bar ---- */
QProgressBar {{
    background-color: {BG_HOVER};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}

/* ---- Tooltip ---- */
QToolTip {{
    background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}}

/* ---- Splitter ---- */
QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 1px;
}}
"""


# ── Button Style Variants ──────────────────────
def btn_style(color: str, color_dark: str, color_pressed: str = None) -> str:
    if color_pressed is None:
        color_pressed = color_dark
    return f"""
        QPushButton {{
            background-color: {color};
            color: #ffffff;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            padding: 8px 16px;
            letter-spacing: 0.3px;
        }}
        QPushButton:hover {{
            background-color: {color_dark};
        }}
        QPushButton:pressed {{
            background-color: {color_pressed};
        }}
        QPushButton:disabled {{
            background-color: {BG_HOVER};
            color: {TEXT_MUTED};
        }}
    """


def btn_ghost_style(color: str) -> str:
    """Transparent button with colored border and text."""
    return f"""
        QPushButton {{
            background-color: transparent;
            color: {color};
            border: 1px solid {color};
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            padding: 6px 14px;
        }}
        QPushButton:hover {{
            background-color: {color}22;
        }}
        QPushButton:pressed {{
            background-color: {color}44;
        }}
        QPushButton:disabled {{
            border-color: {BORDER};
            color: {TEXT_MUTED};
        }}
    """


def card_style() -> str:
    """Style for a rounded card panel."""
    return f"""
        background-color: {BG_BASE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 12px;
    """


# ── Apply Theme to QApplication ───────────────
def apply_theme(app):
    """Apply the dark theme globally to the QApplication."""
    app.setStyleSheet(GLOBAL_QSS)
    app.setApplicationName("Synthetic Defect Tool")
    app.setApplicationVersion("1.0.0")
