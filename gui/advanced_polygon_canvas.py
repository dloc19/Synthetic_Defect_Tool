# ─────────────────────────────────────────────
#  gui/advanced_polygon_canvas.py
#  Production canvas with rubber-band, fill, zoom
# ─────────────────────────────────────────────
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath,
    QCursor, QPixmap
)
from PyQt5.QtCore import Qt, QPoint, QPointF, pyqtSignal


class AdvancedPolygonCanvas(QLabel):
    """
    Bảng vẽ tương tác (Interactive canvas) hiển thị và cho phép vẽ vùng đa giác.

    Chuột trái   → Thêm điểm neo (add point)
    Chuột phải   → Đóng vòng và hoàn thành 1 đa giác (yêu cầu ít nhất 3 điểm)
    Giữ chuột giữa → Kéo thả để cuộn / di chuyển ảnh (pan)
    Lăn chuột    → Phóng to / Thu nhỏ ảnh (zoom in / out)
    """

    # Phát tín hiệu báo cáo trạng thái mỗi khi số điểm hoặc số đa giác thay đổi
    status_changed = pyqtSignal(int, int)   # (số_điểm_đang_vẽ, số_đa_giác_hoàn_thành)

    # ── Colors ────────────────────────────────
    _C_ACTIVE_LINE  = QColor("#f85149")        # red  — current polygon lines
    _C_ACTIVE_PT    = QColor("#ffa657")        # orange — intermediate points
    _C_FIRST_PT     = QColor("#58a6ff")        # blue  — first point highlight
    _C_DONE_FILL    = QColor(63, 185, 80, 55)  # green translucent fill
    _C_DONE_BORDER  = QColor("#3fb950")        # green border
    _C_RUBBER       = QColor("#d29922")        # amber rubber-band line

    def __init__(self):
        super().__init__()

        self._points: list[QPointF] = []      # current active polygon (image coords)
        self._polygons: list[list[QPointF]] = []  # finished polygons (image coords)
        self._mouse_pos: QPoint | None = None    # live cursor position

        self._pixmap_original: QPixmap | None = None
        self._scale_factor: float = 1.0
        self._min_scale: float = 0.05
        self._max_scale: float = 10.0
        self._panning: bool = False
        self._last_pan_pos: QPoint | None = None

        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)

        self._update_placeholder()

    # ── Public API ────────────────────────────
    def load_image(self, pixmap: QPixmap):
        self._pixmap_original = pixmap
        self._scale_factor = 1.0
        self._update_view()

    def undo_point(self):
        if self._points:
            self._points.pop()
            self._emit_status()
            self.update()

    def clear_all(self):
        self._points = []
        self._polygons = []
        self._emit_status()
        self.update()

    @property
    def polygons(self) -> list[list[QPointF]]:
        return self._polygons

    # ── Internal helpers ──────────────────────
    def _update_placeholder(self):
        if self._pixmap_original is None:
            self.setText(
                "📂  Chưa tải ảnh lên\n\n"
                "Hãy tải một ảnh lỗi NG để bắt đầu khoanh vùng đa giác"
            )
            self.setStyleSheet("""
                QLabel {
                    color: #484f58;
                    font-size: 13px;
                    border: 2px dashed #30363d;
                    border-radius: 8px;
                    background: #0d1117;
                }
            """)

    def _update_view(self):
        if self._pixmap_original is None:
            return
        
        # Chốt cứng kích thước bằng phép chiếu tỉ lệ tròn số, bỏ qua sai số khung của tỷ lệ KeepAspectRatio
        new_w = max(1, int(round(self._pixmap_original.width() * self._scale_factor)))
        new_h = max(1, int(round(self._pixmap_original.height() * self._scale_factor)))
        
        scaled = self._pixmap_original.scaled(
            new_w, new_h,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.setFixedSize(scaled.size())

    def _emit_status(self):
        self.status_changed.emit(len(self._points), len(self._polygons))

    # ── Events ────────────────────────────────
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.12 if delta > 0 else 0.89
        new_scale = self._scale_factor * factor
        self._scale_factor = max(self._min_scale, min(self._max_scale, new_scale))
        self._update_view()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Làm tròn tọa độ thành số nguyên pixel để đảm bảo mask lưu xuống nguyên vẹn, bất kể đang zoom
            cx = int(round(event.pos().x() / self._scale_factor))
            cy = int(round(event.pos().y() / self._scale_factor))
            pt = QPointF(cx, cy)
            self._points.append(pt)
            self._emit_status()
            self.update()

        elif event.button() == Qt.RightButton:
            if len(self._points) >= 3:
                self._polygons.append(list(self._points))
                self._points = []
                self._emit_status()
            self.update()

        elif event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        self._mouse_pos = event.pos()
        if self._panning and self._last_pan_pos is not None:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.move(self.pos() + delta)
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CrossCursor)

    def _to_view(self, pt: QPointF) -> QPointF:
        s = self._scale_factor
        return QPointF(pt.x() * s, pt.y() * s)

    # ── Paint ─────────────────────────────────
    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Draw finished polygons (fill + border)
        for poly in self._polygons:
            if len(poly) < 2:
                continue
            path = QPainterPath()
            path.moveTo(self._to_view(poly[0]))
            for pt in poly[1:]:
                path.lineTo(self._to_view(pt))
            path.closeSubpath()

            painter.fillPath(path, QBrush(self._C_DONE_FILL))
            pen = QPen(self._C_DONE_BORDER, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)

            # vertex dots
            painter.setPen(Qt.NoPen)
            painter.setBrush(self._C_DONE_BORDER)
            for pt in poly:
                painter.drawEllipse(self._to_view(pt), 4, 4)

        pts = self._points
        if pts:
            # 2. Draw active polygon lines
            pen_active = QPen(self._C_ACTIVE_LINE, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen_active)
            painter.setBrush(Qt.NoBrush)
            for i in range(len(pts) - 1):
                painter.drawLine(self._to_view(pts[i]), self._to_view(pts[i + 1]))

            # 3. Rubber-band line to current cursor
            if self._mouse_pos is not None and len(pts) >= 1:
                pen_rubber = QPen(self._C_RUBBER, 1, Qt.DashLine)
                painter.setPen(pen_rubber)
                painter.drawLine(self._to_view(pts[-1]), QPointF(self._mouse_pos))

            # 4. Draw vertex points
            painter.setPen(Qt.NoPen)
            for i, pt in enumerate(pts):
                v_pt = self._to_view(pt)
                if i == 0:
                    # First point: large circle as "close here" hint
                    painter.setBrush(self._C_FIRST_PT)
                    painter.drawEllipse(v_pt, 7, 7)
                    # inner dot
                    painter.setBrush(QColor("#ffffff"))
                    painter.drawEllipse(v_pt, 2.5, 2.5)
                else:
                    painter.setBrush(self._C_ACTIVE_PT)
                    painter.drawEllipse(v_pt, 5, 5)

        # 5. Zoom label bottom-right
        zoom_pct = f"{int(self._scale_factor * 100)}%"
        painter.setPen(QColor("#8b949e"))
        painter.setFont(painter.font())
        rect = self.rect().adjusted(0, 0, -8, -6)
        painter.drawText(rect, Qt.AlignBottom | Qt.AlignRight, zoom_pct)