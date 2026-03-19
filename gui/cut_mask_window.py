# ─────────────────────────────────────────────
#  gui/cut_mask_window.py
#  Production Cut Mask Window
# ─────────────────────────────────────────────
import uuid

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QScrollArea,
    QToolBar, QAction, QStatusBar, QFileDialog,
    QMessageBox, QSizePolicy, QDialog
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QKeySequence
from PyQt5.QtCore import Qt, QSize

from gui import icons

import cv2

from gui.advanced_polygon_canvas import AdvancedPolygonCanvas
from gui.theme import (
    BG_DEEP, BG_BASE, BG_SURFACE, BG_HOVER, BORDER,
    ACCENT, SUCCESS, WARNING, DANGER, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    btn_style, btn_ghost_style
)
from core.mask_cutter import polygons_to_mask, cut_patch


# ── Sidebar Panel ─────────────────────────────
class _InfoPanel(QWidget):
    """Bảng điều khiển bên phải (Right-side panel) hiển thị số liệu thống kê đa giác và các nút điều hướng."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_BASE};
                border-left: 1px solid {BORDER};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(16)

        # ── Section: Status ────────────────────
        section_lbl = self._section("DRAWING INFO")
        layout.addWidget(section_lbl)

        self.pts_lbl  = self._stat_row("Active Points", "0")
        self.poly_lbl = self._stat_row("Polygons Done", "0")
        self.file_lbl = self._stat_row("Image", "—")

        for w in (self.pts_lbl, self.poly_lbl, self.file_lbl):
            layout.addWidget(w)

        layout.addWidget(self._separator())

        # ── Section: Guide ─────────────────────
        layout.addWidget(self._section("SHORTCUTS"))

        for icon, key, desc in [
            ("🖱", "L-Click", "Add point"),
            ("🖱", "R-Click", "Close polygon"),
            ("🖱", "Wheel", "Zoom in / out"),
            ("🖱", "Mid-drag", "Pan image"),
        ]:
            row = QHBoxLayout()
            icon_w = QLabel(icon)
            icon_w.setFixedWidth(18)
            icon_w.setStyleSheet(f"background: transparent; font-size: 12px;")
            key_w = QLabel(key)
            key_w.setStyleSheet(f"""
                background-color: {BG_SURFACE};
                color: {TEXT_SECONDARY};
                border: 1px solid {BORDER};
                border-radius: 3px;
                padding: 1px 5px;
                font-size: 10px;
            """)
            desc_w = QLabel(desc)
            desc_w.setStyleSheet(f"background: transparent; color: {TEXT_MUTED}; font-size: 11px;")
            row.addWidget(icon_w)
            row.addWidget(key_w)
            row.addStretch()
            row.addWidget(desc_w)
            container = QWidget()
            container.setStyleSheet("background: transparent;")
            container.setLayout(row)
            layout.addWidget(container)

        layout.addStretch()
        layout.addWidget(self._separator())

        # ── Preview Mask button ───────────────
        self.btn_preview_mask = QPushButton("👁  Preview Mask")
        self.btn_preview_mask.setMinimumHeight(36)
        self.btn_preview_mask.setStyleSheet(btn_ghost_style(ACCENT))
        layout.addWidget(self.btn_preview_mask)

        # ── Save button ───────────────────────
        self.btn_save = QPushButton("  Save Patch && Mask")
        self.btn_save.setMinimumHeight(44)
        self.btn_save.setIcon(icons.SAVE())
        self.btn_save.setIconSize(QSize(16, 16))
        self.btn_save.setStyleSheet(btn_style(SUCCESS, "#2ea043"))
        layout.addWidget(self.btn_save)

    # helpers
    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            font-size: 9px; font-weight: 700;
            color: {TEXT_MUTED}; letter-spacing: 1.5px;
            background: transparent;
        """)
        return lbl

    def _stat_row(self, label: str, value: str) -> QWidget:
        row = QHBoxLayout()
        k = QLabel(label)
        k.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        v = QLabel(value)
        v.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: 700; background: transparent;")
        row.addWidget(k)
        row.addStretch()
        row.addWidget(v)
        c = QWidget()
        c.setStyleSheet("background: transparent;")
        c.setLayout(row)
        c._value_lbl = v
        return c

    def _separator(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setStyleSheet(f"color: {BORDER};")
        return f

    def update_stats(self, pts: int, polys: int):
        self.pts_lbl._value_lbl.setText(str(pts))
        self.poly_lbl._value_lbl.setText(str(polys))

    def set_filename(self, name: str):
        self.file_lbl._value_lbl.setText(name)
        self.file_lbl._value_lbl.setToolTip(name)


# ── Main Window ───────────────────────────────
class CutMaskWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cut Mask NG  —  Synthetic Defect Tool")
        self.setGeometry(80, 80, 1100, 760)
        self.setWindowIcon(icons.CUT())

        self._image = None

        # ── Toolbar ───────────────────────────
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(__import__("PyQt5.QtCore", fromlist=["QSize"]).QSize(16, 16))
        self.addToolBar(toolbar)

        def _tb_action(label, slot, tooltip="", icon_fn=None, bold=False):
            btn = QPushButton(label)
            btn.setToolTip(tooltip)
            if icon_fn is not None:
                btn.setIcon(icon_fn())
                btn.setIconSize(QSize(16, 16))
            h_style = f"""
                QPushButton {{
                    background: transparent;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 14px;
                    color: {TEXT_PRIMARY};
                    font-size: 12px;
                    {'font-weight: 700;' if bold else ''}
                }}
                QPushButton:hover {{ background-color: {BG_HOVER}; }}
                QPushButton:pressed {{ background-color: {BORDER}; }}
            """
            btn.setStyleSheet(h_style)
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)
            return btn

        _tb_action("  Load NG Image", self._load_image, "Open an NG defect image", icons.LOAD)
        self._add_tb_separator(toolbar)
        _tb_action("  Undo Point", self._undo, "Remove the last placed point (Ctrl+Z)", icons.UNDO)
        _tb_action("  Clear All", self._clear, "Clear all polygons and points", icons.CLEAR)
        self._add_tb_separator(toolbar)

        # spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        spacer.setStyleSheet("background: transparent;")
        toolbar.addWidget(spacer)

        # ── Keyboard shortcuts ─────────────────
        from PyQt5.QtWidgets import QShortcut
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Escape"), self, self._clear_active)

        # ── Status Bar ────────────────────────
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_pts = QLabel("Points: 0")
        self._status_poly = QLabel("Polygons: 0")
        self._status_file = QLabel("No image loaded")
        for lbl in (self._status_pts, self._status_poly, self._status_file):
            lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; padding: 0 8px;")
        self._status.addWidget(self._status_file)
        self._status.addPermanentWidget(self._status_pts)
        self._status.addPermanentWidget(self._status_poly)

        # ── Central layout ────────────────────
        central = QWidget()
        central.setStyleSheet(f"background-color: {BG_DEEP};")
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        # Canvas in a scroll area
        self._canvas = AdvancedPolygonCanvas()
        self._canvas.status_changed.connect(self._on_status_changed)

        scroll = QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {BG_DEEP};
                border: none;
            }}
        """)

        h_layout.addWidget(scroll, 1)

        # Sidebar
        self._panel = _InfoPanel()
        self._panel.btn_save.clicked.connect(self._save_patch)
        self._panel.btn_preview_mask.clicked.connect(self._preview_mask)
        h_layout.addWidget(self._panel)

        self.setCentralWidget(central)

    # ── helpers ───────────────────────────────
    def _add_tb_separator(self, tb):
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet(f"color: {BORDER}; margin: 6px 4px;")
        tb.addWidget(sep)

    # ── Slots ─────────────────────────────────
    def _undo(self):
        self._canvas.undo_point()

    def _clear(self):
        self._canvas.clear_all()

    def _clear_active(self):
        """Clear only the in-progress polygon."""
        self._canvas._points = []
        self._canvas._emit_status()
        self._canvas.update()

    def _on_status_changed(self, pts: int, polys: int):
        self._status_pts.setText(f"Points: {pts}")
        self._status_poly.setText(f"Polygons: {polys}")
        self._panel.update_stats(pts, polys)

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load NG Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return

        self._image = cv2.imread(path)
        if self._image is None:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{path}")
            return

        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._canvas.load_image(pix)

        import os
        fname = os.path.basename(path)
        self._status_file.setText(f"📄 {fname}")
        self._panel.set_filename(fname)

    def _save_patch(self):
        if self._image is None:
            QMessageBox.warning(self, "No Image", "Please load an NG image first.")
            return

        all_polys = self._canvas.polygons
        if not all_polys:
            QMessageBox.warning(self, "No Polygons", "Draw at least one polygon (right-click to close).")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return

        polygons_pts = [[(int(p.x()), int(p.y())) for p in poly] for poly in all_polys]
        mask = polygons_to_mask(self._image.shape, polygons_pts)
        patch, patch_mask = cut_patch(self._image, mask)

        name = str(uuid.uuid4())[:8]
        patch_path = f"{folder}/{name}.png"
        mask_path  = f"{folder}/{name}_mask.png"
        cv2.imwrite(patch_path, patch)
        cv2.imwrite(mask_path, patch_mask)

        QMessageBox.information(
            self, "Saved",
            f"✅  Patch saved:\n  {patch_path}\n\n  Mask saved:\n  {mask_path}"
        )

    def _preview_mask(self):
        """Show the current polygon mask overlaid on the loaded image."""
        if self._image is None:
            QMessageBox.warning(self, "No Image", "Load an NG image first.")
            return

        all_polys = self._canvas.polygons
        if not all_polys:
            QMessageBox.warning(self, "No Polygons",
                                "Draw and close at least one polygon (right-click to close).")
            return

        # Build mask from polygons
        polygons_pts = [[(int(p.x()), int(p.y())) for p in poly] for poly in all_polys]
        mask = polygons_to_mask(self._image.shape, polygons_pts)

        # Overlay: green semi-transparent fill
        overlay = self._image.copy()
        green = overlay.copy()
        green[mask > 0] = (0, 200, 80)
        overlay = cv2.addWeighted(overlay, 0.55, green, 0.45, 0)

        # Mask border
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 230, 100), 2)

        # Convert to QPixmap
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # Show in a dialog
        dialog = _MaskPreviewDialog(pix, mask, self)
        dialog.exec_()


# ── Mask Preview Dialog ──────────────────────
from PyQt5.QtWidgets import QDialog


class _MaskPreviewDialog(QDialog):
    """Hiển thị hộp thoại xem trước lớp mặt nạ đa giác đè lên ảnh gốc, kèm theo phần trăm bao phủ."""

    def __init__(self, pixmap: QPixmap, mask, parent=None):
        super().__init__(parent)
        import numpy as np
        self.setWindowTitle("Mask Preview")
        self.setModal(True)
        self.setStyleSheet(f"background: {BG_DEEP};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Image
        lbl = QLabel()
        scaled = pixmap.scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl.setPixmap(scaled)
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)

        # Stats row
        total_px   = mask.size
        masked_px  = int(np.count_nonzero(mask))
        coverage   = 100.0 * masked_px / total_px if total_px else 0.0
        stats = QLabel(
            f"Masked pixels: {masked_px:,}  /  Total: {total_px:,}  —  Coverage: {coverage:.2f}%"
        )
        stats.setAlignment(Qt.AlignCenter)
        stats.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        layout.addWidget(stats)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.setFixedHeight(36)
        btn_close.setStyleSheet(btn_ghost_style(ACCENT))
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        self.adjustSize()