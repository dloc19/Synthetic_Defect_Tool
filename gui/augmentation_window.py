# ─────────────────────────────────────────────
#  gui/augmentation_window.py  —  Augmentation Window
#  Cửa sổ augmentation cơ bản
# ─────────────────────────────────────────────
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QFileDialog,
    QMessageBox, QProgressBar,
    QCheckBox, QDoubleSpinBox, QSpinBox,
    QScrollArea, QSplitter, QGridLayout,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import cv2
import numpy as np

from core.image_augmentor import (
    AUGMENTATION_REGISTRY, get_registry_by_category, run_pipeline,
)
from gui import icons
from gui.theme import (
    BG_DEEP, BG_BASE, BG_SURFACE, BG_HOVER, BORDER,
    ACCENT, ACCENT_DARK, SUCCESS, SUCCESS_DARK,
    WARNING, WARNING_DARK, DANGER, DANGER_DARK,
    PURPLE, PURPLE_DARK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    btn_style, btn_ghost_style,
)


# ── Category Colors ─────────────────────────
_CAT_COLORS = {
    "Geometric":    ("#1f6feb", "#388bfd"),
    "Color":        ("#8957e5", "#a371f7"),
    "Noise & Blur": ("#238636", "#2ea043"),
}

_CAT_ICONS = {
    "Geometric":    "📐",
    "Color":        "🎨",
    "Noise & Blur": "💧",
}


# ═══════════════════════════════════════════════
#  Workers
# ═══════════════════════════════════════════════

class _PreviewWorker(QThread):
    finished = pyqtSignal(object, object)
    error    = pyqtSignal(str)

    def __init__(self, img, enabled_keys, params_override):
        super().__init__()
        self._img, self._keys, self._params = img, enabled_keys, params_override

    def run(self):
        try:
            result = run_pipeline(self._img, self._keys, self._params)
            self.finished.emit(self._img, result)
        except Exception as e:
            self.error.emit(str(e))


class _BatchWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(int)
    error    = pyqtSignal(str)

    def __init__(self, input_paths, count_per_image, enabled_keys, params_override, output_dir):
        super().__init__()
        self._paths, self._count = input_paths, count_per_image
        self._keys, self._params, self._outdir = enabled_keys, params_override, output_dir

    def run(self):
        try:
            total = len(self._paths) * self._count
            done = 0
            for path in self._paths:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    done += self._count
                    self.progress.emit(done, total)
                    continue
                basename = os.path.splitext(os.path.basename(path))[0]
                for i in range(self._count):
                    result = run_pipeline(img, self._keys, self._params)
                    cv2.imwrite(os.path.join(self._outdir, f"{basename}_aug_{i:04d}.png"), result)
                    done += 1
                    self.progress.emit(done, total)
            self.finished.emit(done)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════
#  Algorithm Card
# ═══════════════════════════════════════════════

class _AugCard(QWidget):
    def __init__(self, entry: dict, color: str):
        super().__init__()
        self.entry = entry
        self.setObjectName("AugCard")
        self.setStyleSheet(f"""
            QWidget#AugCard {{
                background-color: {BG_SURFACE};
                border: 1px solid {BORDER};
                border-radius: 6px;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(4)

        self.checkbox = QCheckBox(entry["name"])
        self.checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {TEXT_PRIMARY}; font-size: 12px;
                font-weight: 600; background: transparent; spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {color}; border-radius: 3px;
                background: {BG_DEEP};
            }}
            QCheckBox::indicator:checked {{
                background: {color}; border-color: {color};
            }}
        """)
        root.addWidget(self.checkbox)

        self._spinboxes: dict[str, QDoubleSpinBox | QSpinBox] = {}
        params = entry.get("params", [])

        if params:
            params_container = QWidget()
            params_container.setStyleSheet("background: transparent;")
            p_layout = QGridLayout(params_container)
            p_layout.setContentsMargins(20, 2, 0, 2)
            p_layout.setSpacing(4)
            p_layout.setColumnStretch(1, 1)

            for row, (p_name, p_label, p_type, p_default, p_min, p_max, p_step) in enumerate(params):
                lbl = QLabel(p_label)
                lbl.setStyleSheet(f"font-size: 10px; color: {TEXT_SECONDARY}; background: transparent;")
                p_layout.addWidget(lbl, row, 0)

                if p_type == "int":
                    sb = QSpinBox()
                    sb.setRange(int(p_min), int(p_max))
                    sb.setValue(int(p_default))
                    sb.setSingleStep(int(p_step))
                else:
                    sb = QDoubleSpinBox()
                    sb.setRange(float(p_min), float(p_max))
                    sb.setValue(float(p_default))
                    sb.setSingleStep(float(p_step))
                    sb.setDecimals(1)

                sb.setStyleSheet(f"""
                    QSpinBox, QDoubleSpinBox {{
                        background: {BG_DEEP}; color: {TEXT_PRIMARY};
                        border: 1px solid {BORDER}; border-radius: 3px;
                        padding: 2px 4px; font-size: 10px;
                        min-width: 60px; max-height: 22px;
                    }}
                """)
                sb.setEnabled(False)
                p_layout.addWidget(sb, row, 1)
                self._spinboxes[p_name] = sb

            root.addWidget(params_container)
            self.checkbox.toggled.connect(
                lambda checked: [sb.setEnabled(checked) for sb in self._spinboxes.values()]
            )

    def is_enabled(self) -> bool:
        return self.checkbox.isChecked()

    def get_params(self) -> dict:
        return {name: sb.value() for name, sb in self._spinboxes.items()}


# ═══════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════

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


def _cv2_to_pixmap(img_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ═══════════════════════════════════════════════
#  Augmentation Window
# ═══════════════════════════════════════════════

class AugmentationWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Augmentation  —  Synthetic Defect Tool")
        self.setGeometry(100, 80, 1300, 800)
        self.setWindowIcon(icons.APP())

        self._input_img = None
        self._input_paths: list[str] = []
        self._result_img = None
        self._worker = None
        self._aug_cards: list[_AugCard] = []

        # ── Root splitter ─────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        # ══════════════════════════════════════
        #  LEFT — Controls
        # ══════════════════════════════════════
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(340)
        ctrl_scroll.setMaximumWidth(460)
        ctrl_scroll.setStyleSheet(f"""
            QScrollArea {{
                background: {BG_BASE}; border: none;
                border-right: 1px solid {BORDER};
            }}
        """)

        ctrl_inner = QWidget()
        ctrl_inner.setStyleSheet(f"background: {BG_BASE};")
        ctrl = QVBoxLayout(ctrl_inner)
        ctrl.setContentsMargins(16, 16, 16, 16)
        ctrl.setSpacing(10)
        ctrl_scroll.setWidget(ctrl_inner)

        # ── Input ──────────────────────────────
        ctrl.addWidget(_section_label("INPUT"))

        self._btn_load_single = QPushButton("📷  Load Single Image")
        self._btn_load_single.setMinimumHeight(36)
        self._btn_load_single.setCursor(Qt.PointingHandCursor)
        self._btn_load_single.setStyleSheet(btn_style(ACCENT, ACCENT_DARK))
        self._btn_load_single.clicked.connect(self._load_single)
        ctrl.addWidget(self._btn_load_single)

        self._btn_load_folder = QPushButton("📁  Load Folder (Batch)")
        self._btn_load_folder.setMinimumHeight(36)
        self._btn_load_folder.setCursor(Qt.PointingHandCursor)
        self._btn_load_folder.setStyleSheet(btn_ghost_style(ACCENT))
        self._btn_load_folder.clicked.connect(self._load_folder)
        ctrl.addWidget(self._btn_load_folder)

        self._input_status = QLabel("No input loaded")
        self._input_status.setStyleSheet(f"font-size: 10px; color: {TEXT_MUTED}; background: transparent;")
        ctrl.addWidget(self._input_status)
        ctrl.addWidget(_hsep())

        # ── Algorithms ─────────────────────────
        ctrl.addWidget(_section_label("AUGMENTATION ALGORITHMS"))

        quick = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.setFixedHeight(28)
        btn_all.setCursor(Qt.PointingHandCursor)
        btn_all.setStyleSheet(btn_ghost_style(SUCCESS))
        btn_all.clicked.connect(self._select_all)
        quick.addWidget(btn_all)

        btn_none = QPushButton("Deselect All")
        btn_none.setFixedHeight(28)
        btn_none.setCursor(Qt.PointingHandCursor)
        btn_none.setStyleSheet(btn_ghost_style(DANGER))
        btn_none.clicked.connect(self._deselect_all)
        quick.addWidget(btn_none)
        ctrl.addLayout(quick)

        for cat_name, entries in get_registry_by_category().items():
            color = _CAT_COLORS.get(cat_name, (ACCENT, ACCENT_DARK))[0]
            icon_emoji = _CAT_ICONS.get(cat_name, "📦")

            cat_lbl = QLabel(f"{icon_emoji}  {cat_name.upper()}")
            cat_lbl.setStyleSheet(f"""
                font-size: 10px; font-weight: 700; letter-spacing: 1px;
                color: {color}; background: transparent; padding-top: 6px;
            """)
            ctrl.addWidget(cat_lbl)

            for entry in entries:
                card = _AugCard(entry, color)
                self._aug_cards.append(card)
                ctrl.addWidget(card)

        ctrl.addWidget(_hsep())

        # ── Batch Settings ─────────────────────
        ctrl.addWidget(_section_label("BATCH SETTINGS"))

        count_row = QHBoxLayout()
        count_lbl = QLabel("Images per input:")
        count_lbl.setStyleSheet(f"font-size: 11px; color: {TEXT_SECONDARY}; background: transparent;")
        count_row.addWidget(count_lbl)
        self._batch_count = QSpinBox()
        self._batch_count.setRange(1, 9999)
        self._batch_count.setValue(5)
        self._batch_count.setStyleSheet(f"""
            QSpinBox {{
                background: {BG_SURFACE}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER}; border-radius: 4px;
                padding: 3px 6px; font-size: 12px; min-width: 70px;
            }}
        """)
        count_row.addWidget(self._batch_count)
        ctrl.addLayout(count_row)

        self._batch_folder = None
        self._batch_folder_lbl = QLabel("No output folder selected")
        self._batch_folder_lbl.setStyleSheet(f"font-size: 10px; color: {TEXT_MUTED}; background: transparent;")
        ctrl.addWidget(self._batch_folder_lbl)

        self._btn_pick_folder = QPushButton("📁  Select Output Folder")
        self._btn_pick_folder.setMinimumHeight(34)
        self._btn_pick_folder.setCursor(Qt.PointingHandCursor)
        self._btn_pick_folder.setStyleSheet(btn_ghost_style(ACCENT))
        self._btn_pick_folder.clicked.connect(self._pick_output_folder)
        ctrl.addWidget(self._btn_pick_folder)
        ctrl.addWidget(_hsep())

        # ── Actions ────────────────────────────
        ctrl.addWidget(_section_label("ACTIONS"))

        self._preview_progress = QProgressBar()
        self._preview_progress.setRange(0, 0)
        self._preview_progress.setFixedHeight(4)
        self._preview_progress.setTextVisible(False)
        self._preview_progress.hide()
        ctrl.addWidget(self._preview_progress)

        self._btn_preview = QPushButton("👁  Preview")
        self._btn_preview.setMinimumHeight(44)
        self._btn_preview.setCursor(Qt.PointingHandCursor)
        self._btn_preview.setStyleSheet(btn_style(SUCCESS, SUCCESS_DARK))
        self._btn_preview.clicked.connect(self._run_preview)
        ctrl.addWidget(self._btn_preview)

        self._batch_progress = QProgressBar()
        self._batch_progress.setRange(0, 100)
        self._batch_progress.setValue(0)
        self._batch_progress.setFixedHeight(6)
        self._batch_progress.setTextVisible(False)
        self._batch_progress.hide()
        ctrl.addWidget(self._batch_progress)

        self._btn_batch = QPushButton("⚡  Generate Batch")
        self._btn_batch.setMinimumHeight(44)
        self._btn_batch.setCursor(Qt.PointingHandCursor)
        self._btn_batch.setStyleSheet(btn_style(WARNING, WARNING_DARK))
        self._btn_batch.clicked.connect(self._run_batch)
        ctrl.addWidget(self._btn_batch)

        self._btn_save = QPushButton("💾  Save Preview")
        self._btn_save.setMinimumHeight(38)
        self._btn_save.setEnabled(False)
        self._btn_save.setCursor(Qt.PointingHandCursor)
        self._btn_save.setStyleSheet(btn_style(PURPLE, PURPLE_DARK))
        self._btn_save.clicked.connect(self._save_result)
        ctrl.addWidget(self._btn_save)

        ctrl.addStretch()
        splitter.addWidget(ctrl_scroll)

        # ══════════════════════════════════════
        #  RIGHT — Preview
        # ══════════════════════════════════════
        right = QWidget()
        right.setStyleSheet(f"background: {BG_DEEP};")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        # Header
        hdr = QWidget()
        hdr.setFixedHeight(48)
        hdr.setStyleSheet(f"background: {BG_SURFACE}; border-bottom: 1px solid {BORDER};")
        hr = QHBoxLayout(hdr)
        hr.setContentsMargins(16, 0, 16, 0)
        hr.addWidget(QLabel("🔄  Augmentation Preview"))
        hr.itemAt(0).widget().setStyleSheet(f"font-size: 13px; font-weight: 700; color: {TEXT_PRIMARY}; background: transparent;")
        hr.addStretch()
        self._status_lbl = QLabel("Waiting for input…")
        self._status_lbl.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED}; background: transparent;")
        hr.addWidget(self._status_lbl)
        rv.addWidget(hdr)

        # Compare header
        ch = QWidget()
        ch.setFixedHeight(32)
        ch.setStyleSheet(f"background: {BG_BASE}; border-bottom: 1px solid {BORDER};")
        chr_ = QHBoxLayout(ch)
        chr_.setContentsMargins(16, 0, 16, 0)
        o_lbl = QLabel("ORIGINAL")
        o_lbl.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {TEXT_MUTED}; letter-spacing: 1.5px; background: transparent;")
        chr_.addWidget(o_lbl)
        chr_.addStretch()
        a_lbl = QLabel("AUGMENTED")
        a_lbl.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {SUCCESS}; letter-spacing: 1.5px; background: transparent;")
        chr_.addWidget(a_lbl)
        rv.addWidget(ch)

        # Preview images
        preview_area = QWidget()
        preview_area.setStyleSheet(f"background: {BG_DEEP};")
        pl = QHBoxLayout(preview_area)
        pl.setContentsMargins(16, 16, 16, 16)
        pl.setSpacing(12)

        self._lbl_original = QLabel("Original\n\nLoad an image to begin")
        self._lbl_original.setAlignment(Qt.AlignCenter)
        self._lbl_original.setMinimumSize(300, 300)
        self._lbl_original.setStyleSheet(f"""
            background: {BG_BASE}; border: 1px dashed {BORDER};
            border-radius: 8px; color: {TEXT_MUTED}; font-size: 14px;
        """)

        self._lbl_augmented = QLabel("Augmented\n\nClick Preview to see result")
        self._lbl_augmented.setAlignment(Qt.AlignCenter)
        self._lbl_augmented.setMinimumSize(300, 300)
        self._lbl_augmented.setStyleSheet(f"""
            background: {BG_BASE}; border: 1px dashed {BORDER};
            border-radius: 8px; color: {TEXT_MUTED}; font-size: 14px;
        """)

        pl.addWidget(self._lbl_original, 1)
        pl.addWidget(self._lbl_augmented, 1)
        rv.addWidget(preview_area, 1)

        # Info bar
        ib = QWidget()
        ib.setFixedHeight(32)
        ib.setStyleSheet(f"background: {BG_SURFACE}; border-top: 1px solid {BORDER};")
        ibr = QHBoxLayout(ib)
        ibr.setContentsMargins(16, 0, 16, 0)
        self._info_lbl = QLabel("Select augmentation algorithms and click Preview")
        self._info_lbl.setStyleSheet(f"font-size: 10px; color: {TEXT_MUTED}; background: transparent;")
        ibr.addWidget(self._info_lbl)
        rv.addWidget(ib)

        splitter.addWidget(right)
        splitter.setSizes([380, 920])
        self.setCentralWidget(splitter)

    # ═══════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════

    def _get_enabled_keys(self) -> list[str]:
        return [c.entry["key"] for c in self._aug_cards if c.is_enabled()]

    def _get_params_override(self) -> dict:
        return {c.entry["key"]: c.get_params()
                for c in self._aug_cards if c.is_enabled() and c.get_params()}

    def _select_all(self):
        for c in self._aug_cards:
            c.checkbox.setChecked(True)

    def _deselect_all(self):
        for c in self._aug_cards:
            c.checkbox.setChecked(False)

    def _fit_pixmap(self, label: QLabel, pixmap: QPixmap):
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ═══════════════════════════════════════════
    #  Input
    # ═══════════════════════════════════════════

    def _load_single(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)")
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Error", f"Cannot read image:\n{path}")
            return
        self._input_img = img
        self._input_paths = [path]
        fname = os.path.basename(path)
        self._input_status.setText(f"✅  {fname}")
        self._input_status.setStyleSheet(f"font-size: 10px; color: {SUCCESS}; background: transparent;")
        self._btn_load_single.setText(f"📷  {fname} — Change…")
        self._status_lbl.setText("Image loaded")
        self._fit_pixmap(self._lbl_original, _cv2_to_pixmap(img))
        self._lbl_original.setText("")

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                        if os.path.splitext(f)[1].lower() in exts])
        if not paths:
            QMessageBox.warning(self, "No Images", "No image files found.")
            return
        self._input_paths = paths
        self._input_img = cv2.imread(paths[0], cv2.IMREAD_COLOR)
        short = folder if len(folder) < 35 else "…" + folder[-32:]
        self._input_status.setText(f"📁  {short}  ({len(paths)} images)")
        self._input_status.setStyleSheet(f"font-size: 10px; color: {SUCCESS}; background: transparent;")
        self._status_lbl.setText(f"{len(paths)} images loaded")
        if self._input_img is not None:
            self._fit_pixmap(self._lbl_original, _cv2_to_pixmap(self._input_img))
            self._lbl_original.setText("")

    # ═══════════════════════════════════════════
    #  Preview
    # ═══════════════════════════════════════════

    def _run_preview(self):
        if self._input_img is None:
            QMessageBox.warning(self, "No Input", "Please load an image first.")
            return
        enabled = self._get_enabled_keys()
        if not enabled:
            QMessageBox.warning(self, "No Augmentations", "Please enable at least one algorithm.")
            return
        self._btn_preview.setEnabled(False)
        self._btn_preview.setText("Processing…")
        self._preview_progress.show()
        self._status_lbl.setText("Running…")
        self._worker = _PreviewWorker(self._input_img, enabled, self._get_params_override())
        self._worker.finished.connect(self._on_preview_done)
        self._worker.error.connect(self._on_preview_error)
        self._worker.start()

    def _on_preview_done(self, original, augmented):
        self._result_img = augmented
        self._btn_preview.setEnabled(True)
        self._btn_preview.setText("👁  Preview")
        self._preview_progress.hide()
        self._status_lbl.setText("Done  ✅")
        self._btn_save.setEnabled(True)
        self._fit_pixmap(self._lbl_original, _cv2_to_pixmap(original))
        self._fit_pixmap(self._lbl_augmented, _cv2_to_pixmap(augmented))
        self._lbl_original.setText("")
        self._lbl_augmented.setText("")
        n = len(self._get_enabled_keys())
        self._info_lbl.setText(
            f"Applied {n} augmentation(s)  |  "
            f"Original: {original.shape[1]}×{original.shape[0]}  →  "
            f"Result: {augmented.shape[1]}×{augmented.shape[0]}")

    def _on_preview_error(self, msg):
        self._btn_preview.setEnabled(True)
        self._btn_preview.setText("👁  Preview")
        self._preview_progress.hide()
        self._status_lbl.setText("Error ✗")
        QMessageBox.critical(self, "Error", f"Error:\n\n{msg}")

    # ═══════════════════════════════════════════
    #  Save / Batch
    # ═══════════════════════════════════════════

    def _save_result(self):
        if self._result_img is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Augmented Image", "augmented_result.png",
            "PNG (*.png);;JPEG (*.jpg)")
        if path:
            cv2.imwrite(path, self._result_img)
            QMessageBox.information(self, "Saved", f"✅  Saved:\n{path}")

    def _pick_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._batch_folder = folder
            short = folder if len(folder) < 35 else "…" + folder[-32:]
            self._batch_folder_lbl.setText(f"📂  {short}")
            self._batch_folder_lbl.setToolTip(folder)
            self._batch_folder_lbl.setStyleSheet(f"font-size: 10px; color: {SUCCESS}; background: transparent;")

    def _run_batch(self):
        if not self._input_paths:
            QMessageBox.warning(self, "No Input", "Please load image(s) first.")
            return
        enabled = self._get_enabled_keys()
        if not enabled:
            QMessageBox.warning(self, "No Augmentations", "Please enable at least one algorithm.")
            return
        if not self._batch_folder:
            QMessageBox.warning(self, "No Folder", "Please select an output folder first.")
            return
        total = len(self._input_paths) * self._batch_count.value()
        self._btn_batch.setEnabled(False)
        self._btn_batch.setText(f"Generating 0 / {total}…")
        self._batch_progress.setRange(0, total)
        self._batch_progress.setValue(0)
        self._batch_progress.show()
        self._status_lbl.setText(f"Batch: 0 / {total}")
        self._batch_worker = _BatchWorker(
            self._input_paths, self._batch_count.value(),
            enabled, self._get_params_override(), self._batch_folder)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_done)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.start()

    def _on_batch_progress(self, done, total):
        self._batch_progress.setValue(done)
        self._btn_batch.setText(f"Generating {done} / {total}…")
        self._status_lbl.setText(f"Batch: {done} / {total}")

    def _on_batch_done(self, saved):
        self._btn_batch.setEnabled(True)
        self._btn_batch.setText("⚡  Generate Batch")
        self._batch_progress.hide()
        self._status_lbl.setText(f"Batch done — {saved} images  ✅")
        QMessageBox.information(self, "Done", f"✅  Generated {saved} images.\n\nSaved to:\n{self._batch_folder}")

    def _on_batch_error(self, msg):
        self._btn_batch.setEnabled(True)
        self._btn_batch.setText("⚡  Generate Batch")
        self._batch_progress.hide()
        self._status_lbl.setText("Batch error ✗")
        QMessageBox.critical(self, "Error", f"Error:\n\n{msg}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._input_img is not None and self._lbl_original.pixmap():
            self._fit_pixmap(self._lbl_original, _cv2_to_pixmap(self._input_img))
        if self._result_img is not None and self._lbl_augmented.pixmap():
            self._fit_pixmap(self._lbl_augmented, _cv2_to_pixmap(self._result_img))
