# ─────────────────────────────────────────────
#  gui/blend_window.py  —  Production Blend Window
#  Integrates: blend modes, augmentation, batch generation
# ─────────────────────────────────────────────
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QFileDialog,
    QMessageBox, QProgressBar, QSizePolicy,
    QComboBox, QCheckBox, QDoubleSpinBox,
    QSpinBox, QGroupBox, QScrollArea, QSplitter,
    QRadioButton
)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QObject

import cv2
import numpy as np

from core.blender import blend_defect, BLEND_MODES
from core.augmentor import random_augment
from core.batch_generator import generate_batch
from gui.interactive_blend_canvas import InteractiveBlendCanvas
from gui import icons
from gui.theme import (
    BG_DEEP, BG_BASE, BG_SURFACE, BG_HOVER, BORDER,
    ACCENT, ACCENT_DARK, SUCCESS, SUCCESS_DARK,
    WARNING, DANGER, PURPLE, PURPLE_DARK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    btn_style, btn_ghost_style
)


# ── Workers ───────────────────────────────────
class _PreviewWorker(QThread):
    """Luồng xử lý nền (Worker thread) khởi tạo bản xem trước (preview) lúc hòa trộn."""
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, ok_img, patch, mask, blend_mode, aug_config, position=None):
        super().__init__()
        self._ok, self._patch, self._mask = ok_img, patch, mask
        self._mode = blend_mode
        self._aug  = aug_config
        self._position = position

    def run(self):
        try:
            p, m = self._patch.copy(), self._mask.copy()
            if self._aug:
                p, m = random_augment(p, m, self._aug)
            result = blend_defect(self._ok, p, m, blend_mode=self._mode, position=self._position)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class _BatchWorker(QThread):
    """Luồng xử lý lô dữ liệu (Batch Worker) chạy ngầm để sinh ảnh tổng hợp hàng loạt."""
    progress = pyqtSignal(int, int)   # (đã_hoàn_thành, tổng_số)
    finished = pyqtSignal(int)        # tổng_số_ảnh_đã_lưu
    error    = pyqtSignal(str)

    def __init__(self, ok_img, patch, mask, count,
                 blend_mode, aug_config, output_dir, save_mask, prefix):
        super().__init__()
        self._ok, self._patch, self._mask = ok_img, patch, mask
        self._count  = count
        self._mode   = blend_mode
        self._aug    = aug_config
        self._outdir = output_dir
        self._save_mask = save_mask
        self._prefix = prefix

    def run(self):
        try:
            saved = 0
            for idx, _ in generate_batch(
                self._ok, self._patch, self._mask,
                count=self._count,
                blend_mode=self._mode,
                aug_config=self._aug if self._aug else None,
                output_dir=self._outdir,
                save_mask=self._save_mask,
                prefix=self._prefix,
            ):
                saved = idx
                self.progress.emit(idx, self._count)
            self.finished.emit(saved)
        except Exception as e:
            self.error.emit(str(e))


# ── Load Panel Widget ─────────────────────────
class _LoadPanel(QWidget):
    """
    Một Bảng điều khiển (Panel) rõ nét chuyên đảm nhiệm tải một tệp dữ liệu duy nhất.
    Bao gồm: icon + Tiêu đề (header), Nút tải File cỡ lớn, ảnh thu nhỏ (thumbnail), nhãn tên tệp.
    """
    def __init__(self, title: str, icon_fn, btn_label: str,
                 btn_color: str, btn_dark: str, is_mask: bool = False):
        super().__init__()
        self._is_mask = is_mask
        self.setObjectName("LoadPanel")
        self.setStyleSheet(f"""
            QWidget#LoadPanel {{
                background-color: {BG_SURFACE};
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(6)

        # ── Header row: icon + title ────────────
        header = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setFixedSize(18, 18)
        icon_lbl.setStyleSheet("background: transparent;")
        if icon_fn:
            icon_lbl.setPixmap(icon_fn().pixmap(18, 18))
        header.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size: 11px; font-weight: 700; color: {TEXT_PRIMARY}; background: transparent;"
        )
        header.addWidget(title_lbl, 1)

        # Status dot (grey = not loaded, green = loaded)
        self._dot = QLabel("●")
        self._dot.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; background: transparent;")
        header.addWidget(self._dot)
        root.addLayout(header)

        # ── Middle row: thumbnail + file name ──
        mid = QHBoxLayout()
        mid.setSpacing(10)

        self._thumb = QLabel()
        self._thumb.setFixedSize(48, 48)
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setStyleSheet(f"""
            background: {BG_DEEP};
            border: 1px dashed {BORDER};
            border-radius: 6px;
            color: {TEXT_MUTED};
            font-size: 18px;
        """)
        self._thumb.setText("?")
        mid.addWidget(self._thumb)

        info_col = QVBoxLayout()
        info_col.setSpacing(4)
        self._file_lbl = QLabel("No file loaded")
        self._file_lbl.setWordWrap(True)
        self._file_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_MUTED}; background: transparent;"
        )
        info_col.addWidget(self._file_lbl)
        info_col.addStretch()
        mid.addLayout(info_col, 1)
        root.addLayout(mid)

        # ── Load button — full width ─────────────
        self.btn = QPushButton(btn_label)
        self.btn.setMinimumHeight(34)
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {btn_color};
                color: #ffffff;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 700;
                padding: 6px;
            }}
            QPushButton:hover  {{ background-color: {btn_dark}; }}
            QPushButton:pressed {{ background-color: {btn_dark}; border: 1px solid rgba(255,255,255,0.2); }}
        """)
        root.addWidget(self.btn)

    # public
    def set_loaded(self, path: str, img_bgr=None):
        """Update label, status dot, and thumbnail."""
        fname = os.path.basename(path)
        self._file_lbl.setText(fname)
        self._file_lbl.setToolTip(path)
        self._file_lbl.setStyleSheet(
            f"font-size: 10px; color: {SUCCESS}; background: transparent;"
        )
        self._dot.setStyleSheet(f"color: {SUCCESS}; font-size: 10px; background: transparent;")
        self.btn.setText("✅  Loaded — Change…")

        # Generate thumbnail
        if img_bgr is not None:
            try:
                import numpy as _np
                thumb = img_bgr.copy()
                # Grayscale mask → colorize green
                if self._is_mask or len(thumb.shape) == 2:
                    thumb_show = cv2.applyColorMap(
                        thumb if len(thumb.shape) == 2 else cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY),
                        cv2.COLORMAP_BONE
                    )
                else:
                    thumb_show = thumb
                rgb = cv2.cvtColor(thumb_show, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(
                    48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self._thumb.setPixmap(pix)
                self._thumb.setText("")
            except Exception:
                self._thumb.setText("🖼")


# ── Section helpers ───────────────────────────
def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"""
        font-size: 9px; font-weight: 700; letter-spacing: 1.5px;
        color: {TEXT_MUTED}; background: transparent;
    """)
    return lbl


def _hsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color: {BORDER};")
    return f


# ── Main Window ───────────────────────────────
class BlendWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blend Defect Into OK  —  Synthetic Defect Tool")
        self.setGeometry(120, 100, 1280, 820)
        self.setWindowIcon(icons.BLEND())

        self._ok     = None
        self._patch  = None
        self._mask   = None
        self._result = None
        self._worker = None

        # ── Root split: controls | preview ────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        # ── LEFT: scrollable controls ──────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(320)
        ctrl_scroll.setMaximumWidth(450)
        ctrl_scroll.setStyleSheet(f"""
            QScrollArea {{ background: {BG_BASE}; border: none; border-right: 1px solid {BORDER}; }}
        """)

        ctrl_inner = QWidget()
        ctrl_inner.setStyleSheet(f"background: {BG_BASE};")
        ctrl_layout = QVBoxLayout(ctrl_inner)
        ctrl_layout.setContentsMargins(16, 16, 16, 16)
        ctrl_layout.setSpacing(12)
        ctrl_scroll.setWidget(ctrl_inner)

        # ── 1. Input Files ───────────────────
        ctrl_layout.addWidget(_section_label("INPUT FILES"))

        self._card_ok = _LoadPanel(
            title="OK Image", icon_fn=icons.OK_IMG,
            btn_label="📷  Load OK Image",
            btn_color=ACCENT, btn_dark=ACCENT_DARK,
        )
        self._card_patch = _LoadPanel(
            title="Defect Patch", icon_fn=icons.PATCH,
            btn_label="🔧  Load Defect Patch",
            btn_color=PURPLE, btn_dark=PURPLE_DARK,
        )
        self._card_mask = _LoadPanel(
            title="Mask Image", icon_fn=icons.MASK,
            btn_label="🎭  Load Mask",
            btn_color=WARNING, btn_dark="#bb8009",
            is_mask=True,
        )

        self._card_ok.btn.clicked.connect(self._load_ok)
        self._card_patch.btn.clicked.connect(self._load_patch)
        self._card_mask.btn.clicked.connect(self._load_mask)

        ctrl_layout.addWidget(self._card_ok)
        ctrl_layout.addWidget(self._card_patch)
        ctrl_layout.addWidget(self._card_mask)
        ctrl_layout.addWidget(_hsep())

        # ── 2. Blend Mode ────────────────────
        ctrl_layout.addWidget(_section_label("BLEND MODE"))

        self._blend_mode_combo = QComboBox()
        for name in BLEND_MODES:
            self._blend_mode_combo.addItem(name)
        self._blend_mode_combo.setStyleSheet(f"""
            QComboBox {{
                background: {BG_SURFACE}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER}; border-radius: 6px;
                padding: 6px 10px; font-size: 12px;
            }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
            QComboBox QAbstractItemView {{
                background: {BG_SURFACE}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER}; selection-background-color: {ACCENT};
            }}
        """)
        ctrl_layout.addWidget(self._blend_mode_combo)
        ctrl_layout.addWidget(_hsep())

        # ── 3. Augmentation ──────────────────
        ctrl_layout.addWidget(_section_label("AUGMENTATION"))

        def _checkbox(text: str, tooltip: str = "") -> QCheckBox:
            cb = QCheckBox(text)
            cb.setToolTip(tooltip)
            cb.setStyleSheet(f"""
                QCheckBox {{ color: {TEXT_PRIMARY}; font-size: 12px; background: transparent; spacing: 6px; }}
                QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {BORDER}; border-radius: 3px; background: {BG_SURFACE}; }}
                QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
            """)
            return cb

        def _spinbox(mn, mx, val, decimals=1, step=5.0) -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setRange(mn, mx)
            sb.setValue(val)
            sb.setDecimals(decimals)
            sb.setSingleStep(step)
            sb.setStyleSheet(f"""
                QDoubleSpinBox {{
                    background: {BG_SURFACE}; color: {TEXT_PRIMARY};
                    border: 1px solid {BORDER}; border-radius: 4px;
                    padding: 3px 6px; font-size: 11px;
                }}
            """)
            return sb

        def _row(label, widget, unit="") -> QHBoxLayout:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet(f"font-size: 11px; color: {TEXT_SECONDARY}; background: transparent; min-width: 90px;")
            row.addWidget(lbl)
            row.addWidget(widget)
            if unit:
                u = QLabel(unit)
                u.setStyleSheet(f"font-size: 10px; color: {TEXT_MUTED}; background: transparent;")
                row.addWidget(u)
            return row

        # Rotation
        self._aug_rotate = _checkbox("Rotation", "Randomly rotate the patch")
        ctrl_layout.addWidget(self._aug_rotate)
        self._aug_rot_min = _spinbox(-180, 0, -30, step=5)
        self._aug_rot_max = _spinbox(0, 180, 30, step=5)
        rot_row = QHBoxLayout()
        rot_row.addLayout(_row("Min angle", self._aug_rot_min, "°"))
        rot_row.addLayout(_row("Max angle", self._aug_rot_max, "°"))
        ctrl_layout.addLayout(rot_row)

        # Flip
        flip_row = QHBoxLayout()
        self._aug_flip_h = _checkbox("Flip H", "Horizontal flip")
        self._aug_flip_v = _checkbox("Flip V", "Vertical flip")
        flip_row.addWidget(self._aug_flip_h)
        flip_row.addWidget(self._aug_flip_v)
        ctrl_layout.addLayout(flip_row)

        # Scale
        self._aug_scale = _checkbox("Scale", "Randomly resize patch")
        ctrl_layout.addWidget(self._aug_scale)
        self._aug_scale_min = _spinbox(0.3, 1.0, 0.8, step=0.1)
        self._aug_scale_max = _spinbox(1.0, 3.0, 1.2, step=0.1)
        sc_row = QHBoxLayout()
        sc_row.addLayout(_row("Min scale", self._aug_scale_min, "×"))
        sc_row.addLayout(_row("Max scale", self._aug_scale_max, "×"))
        ctrl_layout.addLayout(sc_row)

        # Brightness
        self._aug_brightness = _checkbox("Brightness", "Random brightness shift")
        ctrl_layout.addWidget(self._aug_brightness)
        self._aug_br_min = _spinbox(-100, 0, -30, step=10)
        self._aug_br_max = _spinbox(0, 100, 30, step=10)
        br_row = QHBoxLayout()
        br_row.addLayout(_row("Δ Min", self._aug_br_min))
        br_row.addLayout(_row("Δ Max", self._aug_br_max))
        ctrl_layout.addLayout(br_row)

        ctrl_layout.addWidget(_hsep())

        # ── 4. Batch Generation ──────────────
        ctrl_layout.addWidget(_section_label("BATCH GENERATION"))

        self._batch_count = QSpinBox()
        self._batch_count.setRange(1, 9999)
        self._batch_count.setValue(10)
        self._batch_count.setStyleSheet(f"""
            QSpinBox {{
                background: {BG_SURFACE}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER}; border-radius: 4px;
                padding: 3px 6px; font-size: 12px;
            }}
        """)
        ctrl_layout.addLayout(_row("Count", self._batch_count, "images"))

        self._batch_save_mask = _checkbox("Save annotation mask per image")
        ctrl_layout.addWidget(self._batch_save_mask)

        self._batch_folder = None
        self._batch_folder_lbl = QLabel("No folder selected")
        self._batch_folder_lbl.setStyleSheet(
            f"font-size: 10px; color: {TEXT_MUTED}; background: transparent;"
        )
        ctrl_layout.addWidget(self._batch_folder_lbl)

        self._btn_pick_folder = QPushButton("📁  Select Output Folder")
        self._btn_pick_folder.setMinimumHeight(34)
        self._btn_pick_folder.setStyleSheet(btn_ghost_style(ACCENT))
        self._btn_pick_folder.clicked.connect(self._pick_batch_folder)
        ctrl_layout.addWidget(self._btn_pick_folder)

        self._batch_progress = QProgressBar()
        self._batch_progress.setRange(0, 100)
        self._batch_progress.setValue(0)
        self._batch_progress.setFixedHeight(6)
        self._batch_progress.setTextVisible(False)
        self._batch_progress.hide()
        ctrl_layout.addWidget(self._batch_progress)

        self._btn_batch = QPushButton("⚡  Generate Batch")
        self._btn_batch.setMinimumHeight(44)
        self._btn_batch.setStyleSheet(btn_style(WARNING, WARNING_DARK))
        self._btn_batch.clicked.connect(self._run_batch)
        ctrl_layout.addWidget(self._btn_batch)

        ctrl_layout.addWidget(_hsep())

        # ── 4.5. Placement Mode ──────────────
        ctrl_layout.addWidget(_section_label("PLACEMENT"))

        self._radio_random = QRadioButton("Random Position")
        self._radio_manual = QRadioButton("Manual (Drag on Canvas)")
        self._radio_manual.setChecked(True)

        placement_row = QVBoxLayout()
        for rd in (self._radio_random, self._radio_manual):
            rd.setStyleSheet(f"QRadioButton {{ color: {TEXT_PRIMARY}; font-size: 11px; }}")
            placement_row.addWidget(rd)
        ctrl_layout.addLayout(placement_row)

        self._radio_manual.toggled.connect(self._on_placement_toggled)

        ctrl_layout.addWidget(_hsep())

        # ── 5. Single Preview + Save ─────────
        ctrl_layout.addWidget(_section_label("SINGLE PREVIEW"))

        self._preview_progress = QProgressBar()
        self._preview_progress.setRange(0, 0)
        self._preview_progress.setFixedHeight(4)
        self._preview_progress.setTextVisible(False)
        self._preview_progress.hide()
        ctrl_layout.addWidget(self._preview_progress)

        self._btn_preview = QPushButton("  Preview Blend")
        self._btn_preview.setMinimumHeight(48)
        self._btn_preview.setIcon(icons.PREVIEW())
        self._btn_preview.setIconSize(QSize(18, 18))
        self._btn_preview.setStyleSheet(btn_style(SUCCESS, SUCCESS_DARK))
        self._btn_preview.clicked.connect(self._run_preview)
        ctrl_layout.addWidget(self._btn_preview)

        self._btn_save = QPushButton("  Save Result")
        self._btn_save.setMinimumHeight(40)
        self._btn_save.setEnabled(False)
        self._btn_save.setIcon(icons.SAVE())
        self._btn_save.setIconSize(QSize(16, 16))
        self._btn_save.setStyleSheet(btn_style(ACCENT, ACCENT_DARK))
        self._btn_save.clicked.connect(self._save_result)
        ctrl_layout.addWidget(self._btn_save)

        ctrl_layout.addStretch()

        splitter.addWidget(ctrl_scroll)

        # ── RIGHT: preview area ────────────────
        right_panel = QWidget()
        right_panel.setStyleSheet(f"background: {BG_DEEP};")
        right_vbox = QVBoxLayout(right_panel)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)

        # Header bar
        header = QWidget()
        header.setFixedHeight(48)
        header.setStyleSheet(f"background: {BG_SURFACE}; border-bottom: 1px solid {BORDER};")
        h_row = QHBoxLayout(header)
        h_row.setContentsMargins(16, 0, 16, 0)
        h_lbl = QLabel("Preview")
        h_lbl.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {TEXT_PRIMARY}; background: transparent;")
        h_row.addWidget(h_lbl)
        h_row.addStretch()
        self._ph_status = QLabel("Waiting for input…")
        self._ph_status.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED}; background: transparent;")
        h_row.addWidget(self._ph_status)
        right_vbox.addWidget(header)

        # Interactive Canvas
        self._canvas = InteractiveBlendCanvas()
        
        self._canvas_scroll = QScrollArea()
        self._canvas_scroll.setWidget(self._canvas)
        self._canvas_scroll.setWidgetResizable(False)
        self._canvas_scroll.setAlignment(Qt.AlignCenter)
        self._canvas_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {BG_DEEP};
                border: none;
            }}
        """)
        right_vbox.addWidget(self._canvas_scroll, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([340, 940])

        self.setCentralWidget(splitter)

    # ── aug_config helper ──────────────────────
    def _build_aug_config(self) -> dict | None:
        any_aug = any([
            self._aug_rotate.isChecked(),
            self._aug_flip_h.isChecked(),
            self._aug_flip_v.isChecked(),
            self._aug_scale.isChecked(),
            self._aug_brightness.isChecked(),
        ])
        if not any_aug:
            return None
        return {
            "rotate":         self._aug_rotate.isChecked(),
            "rotate_min":     self._aug_rot_min.value(),
            "rotate_max":     self._aug_rot_max.value(),
            "flip_h":         self._aug_flip_h.isChecked(),
            "flip_v":         self._aug_flip_v.isChecked(),
            "scale":          self._aug_scale.isChecked(),
            "scale_min":      self._aug_scale_min.value(),
            "scale_max":      self._aug_scale_max.value(),
            "brightness":     self._aug_brightness.isChecked(),
            "brightness_min": self._aug_br_min.value(),
            "brightness_max": self._aug_br_max.value(),
        }

    def _current_blend_mode(self) -> int:
        name = self._blend_mode_combo.currentText()
        return BLEND_MODES[name]

    def _on_placement_toggled(self, checked):
        self._canvas.set_manual_mode(checked)

    # ── File loaders ───────────────────────────
    def _open_file(self, title, as_gray=False):
        path, _ = QFileDialog.getOpenFileName(
            self, title, "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not path:
            return None, None
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR)
        return path, img

    def _load_ok(self):
        path, img = self._open_file("Select OK Image")
        if img is not None:
            self._ok = img
            self._card_ok.set_loaded(path, img)
            
            # Show in canvas
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self._canvas.load_background(QPixmap.fromImage(qimg))
            # Also reset preview text if it exists
            self._btn_save.setEnabled(False)

    def _load_patch(self):
        path, img = self._open_file("Select Defect Patch")
        if img is not None:
            self._patch = img
            self._card_patch.set_loaded(path, img)
            if self._mask is not None:
                self._canvas.set_patch(self._patch, self._mask)
            else:
                self._canvas.set_patch(self._patch)

    def _load_mask(self):
        path, img = self._open_file("Select Mask", as_gray=True)
        if img is not None:
            self._mask = img
            self._card_mask.set_loaded(path, img)
            if self._patch is not None:
                self._canvas.set_patch(self._patch, self._mask)

    def _check_inputs(self) -> bool:
        if any(x is None for x in [self._ok, self._patch, self._mask]):
            missing = [n for n, x in [("OK Image", self._ok),
                                       ("Defect Patch", self._patch),
                                       ("Mask", self._mask)] if x is None]
            QMessageBox.warning(self, "Missing Files",
                                "Load all three files first.\n\nMissing: " + ", ".join(missing))
            return False
        return True

    # ── Preview ────────────────────────────────
    def _run_preview(self):
        if not self._check_inputs():
            return
        self._btn_preview.setEnabled(False)
        self._btn_preview.setText("Processing…")
        self._preview_progress.show()
        self._ph_status.setText("Running blend…")

        position = None
        if self._radio_manual.isChecked() and self._canvas.patch_pos:
            position = (int(self._canvas.patch_pos.x()), int(self._canvas.patch_pos.y()))

        self._worker = _PreviewWorker(
            self._ok, self._patch, self._mask,
            self._current_blend_mode(),
            self._build_aug_config(),
            position
        )
        self._worker.finished.connect(self._on_preview_done)
        self._worker.error.connect(self._on_preview_error)
        self._worker.start()

    def _on_preview_done(self, result):
        self._result = result
        self._btn_preview.setEnabled(True)
        self._btn_preview.setText("  Preview Blend")
        self._btn_preview.setIcon(icons.PREVIEW())
        self._btn_preview.setIconSize(QSize(18, 18))
        self._preview_progress.hide()
        self._ph_status.setText("Done  ✅")
        self._btn_save.setEnabled(True)
        self._show_image(result)

    def _on_preview_error(self, msg):
        self._btn_preview.setEnabled(True)
        self._btn_preview.setText("  Preview Blend")
        self._btn_preview.setIcon(icons.PREVIEW())
        self._btn_preview.setIconSize(QSize(18, 18))
        self._preview_progress.hide()
        self._ph_status.setText("Error ✗")
        QMessageBox.critical(self, "Blend Error", f"Error:\n\n{msg}")

    def _show_image(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._canvas.set_preview(QPixmap.fromImage(qimg))

    def _save_result(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Blended Image", "blended_result.png",
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self._result)
            QMessageBox.information(self, "Saved", f"✅  Saved:\n{path}")

    # ── Batch ──────────────────────────────────
    def _pick_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._batch_folder = folder
            short = folder if len(folder) < 40 else "…" + folder[-37:]
            self._batch_folder_lbl.setText(f"📂  {short}")
            self._batch_folder_lbl.setToolTip(folder)
            self._batch_folder_lbl.setStyleSheet(
                f"font-size: 10px; color: {SUCCESS}; background: transparent;"
            )

    def _run_batch(self):
        if not self._check_inputs():
            return
        if not self._batch_folder:
            QMessageBox.warning(self, "No Folder", "Please select an output folder first.")
            return

        count = self._batch_count.value()
        self._btn_batch.setEnabled(False)
        self._btn_batch.setText(f"Generating 0 / {count}…")
        self._batch_progress.setRange(0, count)
        self._batch_progress.setValue(0)
        self._batch_progress.show()
        self._ph_status.setText(f"Batch: 0 / {count}")

        self._batch_worker = _BatchWorker(
            self._ok, self._patch, self._mask,
            count=count,
            blend_mode=self._current_blend_mode(),
            aug_config=self._build_aug_config(),
            output_dir=self._batch_folder,
            save_mask=self._batch_save_mask.isChecked(),
            prefix="synth",
        )
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_done)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.start()

    def _on_batch_progress(self, done: int, total: int):
        self._batch_progress.setValue(done)
        self._btn_batch.setText(f"Generating {done} / {total}…")
        self._ph_status.setText(f"Batch: {done} / {total}")

    def _on_batch_done(self, saved: int):
        self._btn_batch.setEnabled(True)
        self._btn_batch.setText("⚡  Generate Batch")
        self._batch_progress.hide()
        self._ph_status.setText(f"Batch done — {saved} images saved  ✅")
        QMessageBox.information(
            self, "Batch Done",
            f"✅  Generated {saved} synthetic images.\n\nSaved to:\n{self._batch_folder}"
        )

    def _on_batch_error(self, msg: str):
        self._btn_batch.setEnabled(True)
        self._btn_batch.setText("⚡  Generate Batch")
        self._batch_progress.hide()
        self._ph_status.setText("Batch error ✗")
        QMessageBox.critical(self, "Batch Error", f"Error during batch:\n\n{msg}")


# theme import needed for WARNING_DARK
from gui.theme import WARNING_DARK
