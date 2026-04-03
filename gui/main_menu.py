# ─────────────────────────────────────────────
#  gui/main_menu.py  —  Production Main Menu
# ─────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize
from PyQt5.QtGui import QFont, QColor, QPalette

from gui.theme import (
    BG_DEEP, BG_BASE, BG_SURFACE, BORDER,
    ACCENT, SUCCESS, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    btn_style, btn_ghost_style
)
from gui import icons
from gui.cut_mask_window import CutMaskWindow
from gui.blend_window import BlendWindow
from gui.augmentation_window import AugmentationWindow


class _FeatureButton(QPushButton):
    """Large feature card-style button."""

    def __init__(self, icon: str, title: str, subtitle: str, color: str, color_dark: str):
        super().__init__()
        self.setFixedHeight(80)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(16)

        # Icon label
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"font-size: 26px; background: transparent;")
        icon_lbl.setFixedWidth(36)

        # Text block
        text_container = QWidget()
        text_container.setAttribute(Qt.WA_TransparentForMouseEvents)
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        text_container.setStyleSheet("background: transparent;")

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: #ffffff; background: transparent;"
        )

        sub_lbl = QLabel(subtitle)
        sub_lbl.setStyleSheet(
            f"font-size: 11px; color: rgba(255,255,255,0.65); background: transparent;"
        )

        text_layout.addWidget(title_lbl)
        text_layout.addWidget(sub_lbl)

        # Arrow
        arrow = QLabel("›")
        arrow.setStyleSheet("font-size: 20px; color: rgba(255,255,255,0.5); background: transparent;")

        layout.addWidget(icon_lbl)
        layout.addWidget(text_container, 1)
        layout.addWidget(arrow)

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border: none;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: {color_dark};
            }}
            QPushButton:pressed {{
                background-color: {color_dark};
                border: 1px solid rgba(255,255,255,0.15);
            }}
        """)


class MainMenu(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Synthetic Defect Tool")
        self.setFixedSize(440, 660)
        self.setStyleSheet(f"background-color: {BG_DEEP};")
        self.setWindowIcon(icons.APP())

        # ── Central container ──────────────────────
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header bar ────────────────────────────
        header = QWidget()
        header.setFixedHeight(8)
        header.setStyleSheet(f"""
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 {ACCENT},
                stop:0.5 #a371f7,
                stop:1 {SUCCESS}
            );
        """)
        root_layout.addWidget(header)

        # ── Body ──────────────────────────────────
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(32, 40, 32, 28)
        body_layout.setSpacing(0)

        # Logo / Icon area
        logo_row = QHBoxLayout()
        logo_lbl = QLabel("⚡")
        logo_lbl.setStyleSheet("font-size: 40px; background: transparent;")
        logo_row.addWidget(logo_lbl)
        logo_row.addStretch()

        # Version badge
        ver_lbl = QLabel("v1.0.0")
        ver_lbl.setStyleSheet(f"""
            background-color: {BG_SURFACE};
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 2px 10px;
            font-size: 10px;
            font-weight: 600;
        """)
        logo_row.addWidget(ver_lbl)
        body_layout.addLayout(logo_row)

        body_layout.addSpacing(24)

        # Title
        title = QLabel("Synthetic Defect\nGenerator")
        title.setStyleSheet(f"""
            font-size: 26px;
            font-weight: 800;
            color: {TEXT_PRIMARY};
            line-height: 1.3;
            background: transparent;
        """)
        body_layout.addWidget(title)

        body_layout.addSpacing(6)

        subtitle = QLabel("Công cụ chuyên nghiệp hỗ trợ khởi tạo dữ liệu huấn luyện lỗi tổng hợp")
        subtitle.setStyleSheet(f"""
            font-size: 12px;
            color: {TEXT_SECONDARY};
            background: transparent;
        """)
        body_layout.addWidget(subtitle)

        body_layout.addSpacing(32)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {BORDER};")
        body_layout.addWidget(sep)

        body_layout.addSpacing(28)

        # Section label
        section_lbl = QLabel("SELECT TOOL")
        section_lbl.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 700;
            color: {TEXT_MUTED};
            letter-spacing: 1.5px;
            background: transparent;
        """)
        body_layout.addWidget(section_lbl)

        body_layout.addSpacing(12)

        # ── Feature Buttons ────────────────────────
        self.btn_cut = _FeatureButton(
            icon="✂",
            title="Cut Mask NG",
            subtitle="Vẽ đa giác để chiết xuất các mảng lỗi (ng)",
            color="#1f6feb",
            color_dark="#388bfd",
        )
        self.btn_cut.setIcon(icons.CUT())
        self.btn_cut.setIconSize(QSize(20, 20))
        self.btn_cut.setToolTip("Open the polygon mask cutting tool (NG images)")
        body_layout.addWidget(self.btn_cut)

        body_layout.addSpacing(12)

        self.btn_blend = _FeatureButton(
            icon="🎨",
            title="Blend Defect Into OK",
            subtitle="Hòa trộn mảng lỗi vào ảnh OK một cách hoàn hảo",
            color="#238636",
            color_dark="#2ea043",
        )
        self.btn_blend.setIcon(icons.BLEND())
        self.btn_blend.setIconSize(QSize(20, 20))
        self.btn_blend.setToolTip("Open the defect blending tool")
        body_layout.addWidget(self.btn_blend)

        body_layout.addSpacing(12)

        self.btn_augment = _FeatureButton(
            icon="🔄",
            title="Augmentation",
            subtitle="Các phương pháp tăng cường dữ liệu ảnh cơ bản",
            color="#8957e5",
            color_dark="#a371f7",
        )
        self.btn_augment.setToolTip("Open the full image augmentation tool")
        body_layout.addWidget(self.btn_augment)

        body_layout.addStretch()

        # ── Footer ────────────────────────────────
        footer_sep = QFrame()
        footer_sep.setFrameShape(QFrame.HLine)
        footer_sep.setStyleSheet(f"color: {BORDER};")
        body_layout.addWidget(footer_sep)

        body_layout.addSpacing(14)

        footer_row = QHBoxLayout()
        footer_lbl = QLabel("Synthetic Defect Tool  ·  2026")
        footer_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_MUTED}; background: transparent;"
        )
        footer_row.addWidget(footer_lbl)
        footer_row.addStretch()

        hint = QLabel("Developed by dloc19")
        hint.setStyleSheet(
            f"font-size: 10px; color: {TEXT_MUTED}; background: transparent;"
        )
        footer_row.addWidget(hint)
        body_layout.addLayout(footer_row)

        root_layout.addWidget(body)
        self.setCentralWidget(root)

        # ── Connections ───────────────────────────
        self.btn_cut.clicked.connect(self._open_cut)
        self.btn_blend.clicked.connect(self._open_blend)
        self.btn_augment.clicked.connect(self._open_augment)

    # ─── Slots ────────────────────────────────────
    def _open_cut(self):
        self.cut_win = CutMaskWindow()
        self.cut_win.show()

    def _open_blend(self):
        self.blend_win = BlendWindow()
        self.blend_win.show()

    def _open_augment(self):
        self.augment_win = AugmentationWindow()
        self.augment_win.show()