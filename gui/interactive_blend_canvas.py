# ─────────────────────────────────────────────
#  gui/interactive_blend_canvas.py
#  Canvas for previewing blends and manual patch placement
# ─────────────────────────────────────────────
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QPointF, pyqtSignal, QRectF
import cv2
import numpy as np


class InteractiveBlendCanvas(QLabel):
    """
    Bảng vẽ có thể phóng to, cuộn ảnh dùng để xem trước kết quả hòa trộn (blend)
    và hỗ trợ việc kéo thả nhãn lỗi (patch) thủ công.
    - Lăn chuột (Scroll-wheel): Phóng to / Thu nhỏ (Zoom)
    - Giữ chuột giữa (Middle-click drag): Cuộn, di chuyển khung nhìn (Pan)
    - Giữ chuột trái (Chỉ trong chế độ Manual): Di chuyển nhãn (Move patch)
    """
    position_changed = pyqtSignal(int, int)  # Cấp tín hiệu mỗi khi nhãn di chuyển đặt xuống (x, y)

    def __init__(self):
        super().__init__()
        self._bg_pixmap: QPixmap | None = None
        self._preview_pixmap: QPixmap | None = None

        self._patch_pixmap: QPixmap | None = None
        self._patch_pos: QPointF | None = None  # Top-left of patch in image coords

        self._scale_factor: float = 1.0
        self._min_scale: float = 0.05
        self._max_scale: float = 10.0
        self._panning: bool = False
        self._last_pan_pos: QPoint | None = None

        self._manual_mode: bool = True
        self._dragging_patch: bool = False

        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background: transparent;")

    # ── Public API ────────────────────────────
    def load_background(self, pixmap: QPixmap, reset_view: bool = True):
        self._bg_pixmap = pixmap
        # the preview is cleared until successfully blended, or we can just show bg
        self._preview_pixmap = None
        if reset_view:
            self._scale_factor = 1.0
        self._update_view()

    def set_preview(self, pixmap: QPixmap):
        """Show the rendered blend result."""
        self._preview_pixmap = pixmap
        self._update_view()

    def set_patch(self, patch_bgr: np.ndarray, mask: np.ndarray | None = None):
        # Convert BGR to BGRA
        bgra = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2BGRA)
        if mask is not None:
            # Apply mask to alpha channel (0 = transparent, 255 = opaque)
            bgra[:, :, 3] = mask
        else:
            bgra[:, :, 3] = 255

        h, w, ch = bgra.shape
        qimg = QImage(bgra.data, w, h, ch * w, QImage.Format_ARGB32)
        self._patch_pixmap = QPixmap.fromImage(qimg)
        # Default to center of bg if available
        if self._bg_pixmap:
            bw, bh = self._bg_pixmap.width(), self._bg_pixmap.height()
            self._patch_pos = QPointF(bw / 2 - w / 2, bh / 2 - h / 2)
        else:
            self._patch_pos = QPointF(0, 0)
        self.update()

    def set_manual_mode(self, enabled: bool):
        self._manual_mode = enabled
        self.update()

    @property
    def patch_pos(self) -> QPointF | None:
        return self._patch_pos

    # ── Internal ──────────────────────────────
    def _update_view(self):
        base = self._preview_pixmap if self._preview_pixmap else self._bg_pixmap
        if base is None:
            return
        
        # Đảm bảo trục X và Y luôn giãn chuẩn tuyệt đối theo tỷ lệ, tránh tỷ lệ KeepAspectRatio nội bộ của Qt gây sai lệch 1 pixel
        new_w = max(1, int(round(base.width() * self._scale_factor)))
        new_h = max(1, int(round(base.height() * self._scale_factor)))
        
        scaled = base.scaled(
            new_w, new_h,
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.setFixedSize(scaled.size())

    def _to_view(self, pt: QPointF) -> QPointF:
        s = self._scale_factor
        return QPointF(pt.x() * s, pt.y() * s)

    def _to_img(self, pt: QPoint) -> QPointF:
        s = self._scale_factor
        return QPointF(pt.x() / s, pt.y() / s)

    def _safe_patch_rect(self) -> QRectF | None:
        if not self._patch_pixmap or not self._patch_pos:
            return None
        return QRectF(
            self._patch_pos.x(), self._patch_pos.y(),
            self._patch_pixmap.width(), self._patch_pixmap.height()
        )

    # ── Events ────────────────────────────────
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 0.89
        new_scale = self._scale_factor * factor
        self._scale_factor = max(self._min_scale, min(self._max_scale, new_scale))
        self._update_view()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

        elif event.button() == Qt.LeftButton and self._manual_mode:
            rect = self._safe_patch_rect()
            if rect:
                img_pt = self._to_img(event.pos())
                if rect.contains(img_pt):
                    self._dragging_patch = True
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    # teleport center of patch to click
                    pw, ph = rect.width(), rect.height()
                    # Khóa cứng tọa độ vào lưới pixel nguyên (tránh bị lệch vị trí hòa trộn OpenCV khi zoom to màn hình)
                    tgt_x = int(round(img_pt.x() - pw / 2))
                    tgt_y = int(round(img_pt.y() - ph / 2))
                    self._patch_pos = QPointF(tgt_x, tgt_y)
                    self.position_changed.emit(tgt_x, tgt_y)
                    self.update()

    def mouseMoveEvent(self, event):
        if self._panning and self._last_pan_pos is not None:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.move(self.pos() + delta)

        elif self._dragging_patch and self._manual_mode and self._patch_pixmap:
            img_pt = self._to_img(event.pos())
            pw, ph = self._patch_pixmap.width(), self._patch_pixmap.height()
            
            # Chốt tọa độ về số nguyên tuyệt đối (Int)
            tgt_x = int(round(img_pt.x() - pw / 2))
            tgt_y = int(round(img_pt.y() - ph / 2))
            
            self._patch_pos = QPointF(tgt_x, tgt_y)
            self.position_changed.emit(tgt_x, tgt_y)
            self.update()

        elif self._manual_mode:
            rect = self._safe_patch_rect()
            if rect:
                img_pt = self._to_img(event.pos())
                if rect.contains(img_pt):
                    self.setCursor(Qt.OpenHandCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton:
            if self._dragging_patch:
                self._dragging_patch = False
                self.setCursor(Qt.OpenHandCursor)

    # ── Paint ─────────────────────────────────
    def paintEvent(self, event):
        super().paintEvent(event)

        # Draw the manual placement preview of the patch overlay
        if self._manual_mode and self._patch_pixmap and self._patch_pos:
            # DO NOT draw the patch if we already have a generated preview for exactly this position!
            # But wait, it's easier to always draw it if manual mode is enabled. If they drag it, it moves.
            # actually we can draw it playfully semi-transparent.
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)

            s = self._scale_factor
            v_x = self._patch_pos.x() * s
            v_y = self._patch_pos.y() * s
            v_w = self._patch_pixmap.width() * s
            v_h = self._patch_pixmap.height() * s

            # Optional: restrict to background bounds if we want to clip, but QPainter handles overflow
            painter.setOpacity(0.55 if self._dragging_patch else 0.8)
            
            # Vẽ patch bằng QRectF số thực chống rung, dịch pixel và đảm bảo nội suy chính xác (Smooth Transform)
            target_rect = QRectF(v_x, v_y, v_w, v_h)
            source_rect = QRectF(0, 0, self._patch_pixmap.width(), self._patch_pixmap.height())
            
            painter.drawPixmap(target_rect, self._patch_pixmap, source_rect)

            # border
            pen = QPen(QColor("#3fb950"), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(target_rect)

            # zoom indicator
            zoom_pct = f"{int(self._scale_factor * 100)}%"
            painter.setPen(QColor("#8b949e"))
            painter.setOpacity(1.0)
            painter.setFont(painter.font())
            painter.drawText(self.rect().adjusted(0, 0, -8, -6), Qt.AlignBottom | Qt.AlignRight, zoom_pct)
