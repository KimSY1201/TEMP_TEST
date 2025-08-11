import sys
import os
import subprocess
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QGridLayout, QMessageBox, QPushButton,
                             QSlider, QSpinBox, QDoubleSpinBox, QCheckBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint, QThread
from PyQt6.QtGui import QColor, QFont, QScreen, QMouseEvent, QPainter, QPen, QBrush
import queue
import time
from scipy.ndimage import gaussian_filter

# 기존 스타일시트 유지
GLOBAL_STYLESHEET = """
/* 전체 위젯 기본 스타일 */
QWidget {
    color: #E0E0E0;
    font-family: 'Malgun Gothic', 'Segoe UI', 'Arial';
}

#MainWindow {
    background-color: #2E2E2E;
}

#TopPanel, #LeftPanel {
    background-color: #3C3C3C;
    border-radius: 8px;
    padding: 10px;
}

QLabel[class="TitleLabel"] {
    font-size: 16px;
    font-weight: bold;
    color: #FFFFFF;
    padding: 2px;
}

QLabel[class="InfoLabel"] {
    font-size: 12px;
    font-weight: bold;
}

#AnomalyCountLabel {
    font-size: 16px;
    font-weight: bold;
    color: #F44336;
    padding: 5px;
}

QLabel[class="HeatmapCell"] {
    font-weight: bold;
    border: none;
}

QPushButton {
    background-color: #555555;
    color: #FFFFFF;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #6A6A6A;
}
QPushButton:pressed {
    background-color: #4A4A4A;
}

QPushButton[class="AdjustButton"] {
    font-size: 16px;
    font-weight: bold;
    padding: 4px 8px;
    min-width: 30px;
}

QDoubleSpinBox {
    background-color: #2E2E2E;
    color: #FFFFFF;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 5px;
    font-size: 16px;
    font-weight: bold;
}

QLabel[styleClass="Indicator"] {
    min-width: 20px;
    max-width: 20px;
    min-height: 20px;
    max-height: 20px;
    border-radius: 10px;
    border: 1px solid rgba(0, 0, 0, 100);
}

QLabel[styleClass="Indicator"][state="stable"] {
    background-color: qlineargradient(spread:pad, x1:0.145, y1:0.16, x2:0.92, y2:0.915,
                                     stop:0 rgba(0, 255, 0, 255),
                                     stop:1 rgba(0, 128, 0, 255));
}
QLabel[styleClass="Indicator"][state="detected"] {
    background-color: qlineargradient(spread:pad, x1:0.145, y1:0.16, x2:0.92, y2:0.915,
                                     stop:0 rgba(255, 0, 0, 255),
                                     stop:1 rgba(128, 0, 0, 255));
}
"""

class DataSignal(QObject):
    """GUI 업데이트를 위한 시그널 클래스"""
    update_data_signal = pyqtSignal(dict)

class DraggableNumberLabel(QLabel):
    """마우스 드래그로 숫자를 변경할 수 있는 사용자 정의 QLabel 위젯"""
    valueChanged = pyqtSignal(float)

    def __init__(self, initial_value: float = 0.0, step: float = 0.1, parent=None):
        super().__init__(parent)
        self._value = initial_value
        self._step = step
        self._last_mouse_pos = QPoint()
        self._is_dragging = False

        self.setText(f"{self._value:.1f}")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.setMouseTracking(True)

        self.setStyleSheet("""
            DraggableNumberLabel {
                border: 2px solid #555;
                border-radius: 5px;
                background-color: #3C3C3C;
                color: #FFFFFF;
                padding: 5px;
            }
            DraggableNumberLabel:hover {
                border: 2px solid #888;
            }
        """)

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_value: float):
        if self._value != new_value:
            self._value = new_value
            self.setText(f"{self._value:.1f}")
            self.valueChanged.emit(self._value)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._last_mouse_pos = event.globalPosition().toPoint()
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_dragging:
            current_mouse_pos = event.globalPosition().toPoint()
            delta_x = current_mouse_pos.x() - self._last_mouse_pos.x()
            change = (delta_x / 10.0) * self._step
            self.value += change
            self._last_mouse_pos = current_mouse_pos
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = False
            self.unsetCursor()
        super().mouseReleaseEvent(event)
        
        
class HeatmapOverlay(QWidget):
    """열원 감지 오버레이만을 담당하는 위젯"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hotspots = []
        self.cell_size = 0
        self.interpolated_grid_size = 24
        self.original_grid_size = 8
        
        # 투명 배경 설정
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

    def set_hotspots(self, hotspots, cell_size, grid_size):
        """열원 정보와 그리드 정보 업데이트"""
        self.hotspots = hotspots or []
        self.cell_size = cell_size
        self.interpolated_grid_size = grid_size
        self.update()  # 다시 그리기 요청

    def paintEvent(self, event):
        """열원 감지 결과를 오버레이로 표시"""
        super().paintEvent(event)
        
        if not self.hotspots or self.cell_size == 0:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 스케일 팩터 계산 (8x8 -> interpolated_grid_size)
        scale_factor = self.interpolated_grid_size / self.original_grid_size
        
        for hotspot in self.hotspots:
            try:
                self._draw_hotspot(painter, hotspot, scale_factor)
            except Exception as e:
                print(f"Error drawing hotspot {hotspot}: {e}")

    def _draw_hotspot(self, painter, hotspot, scale_factor):
        """개별 열원을 그리기"""
        center_x, center_y = hotspot['center']
        
        # 8x8 좌표를 interpolated_grid_size 좌표로 변환
        scaled_x = center_x * scale_factor
        scaled_y = center_y * scale_factor
        
        # 픽셀 좌표로 변환
        pixel_x = int(scaled_x * self.cell_size + self.cell_size / 2)
        pixel_y = int(scaled_y * self.cell_size + self.cell_size / 2)
        
        # 감지 방법과 검증 상태에 따른 스타일 결정
        style = self._get_hotspot_style(hotspot)
        
        # 열원 영역 그리기 (원형)
        self._draw_hotspot_circle(painter, pixel_x, pixel_y, style)
        
        # 열원 정보 텍스트 그리기
        self._draw_hotspot_info(painter, pixel_x, pixel_y, hotspot, style)
        
        # 열원 좌표들을 개별 셀로 그리기 (선택사항)
        if style.get('show_cells', False):
            self._draw_hotspot_cells(painter, hotspot, scale_factor, style)

    def _get_hotspot_style(self, hotspot):
        """열원 타입에 따른 스타일 설정 반환"""
        detection_method = hotspot.get('detection_method', 'unknown')
        validation_status = hotspot.get('validation_status', 'unknown')
        cross_validation = hotspot.get('cross_validation', False)
        detection_count = hotspot.get('detection_count', 0)
        consensus_score = hotspot.get('consensus_score', 0.0)
        model_confidence = hotspot.get('model_confidence', 0.0)
        
        # 기본 스타일
        style = {
            'circle_radius': max(30, min(50, hotspot.get('size', 1) * 5)),
            'show_cells': False,
            'text_size': 10,
            'show_confidence': False
        }
        
        # 감지 방법과 검증 상태에 따른 색상 및 스타일 결정
        if cross_validation and validation_status == 'consensus_validated':
            # 교차검증 통과한 최고 신뢰도 열원
            style.update({
                'fill_color': QColor(220, 20, 20, 180),      # 반투명 빨간색
                'border_color': QColor(255, 255, 255, 255),   # 흰색 테두리
                'border_width': 3,
                'text_color': QColor(255, 255, 255),
                'priority': 5,
                'show_confidence': True
            })
        elif detection_method == 'lgbm_model':
            # LightGBM 모델 기반 감지
            if model_confidence > 0.8:
                style.update({
                    'fill_color': QColor(255, 100, 0, 160),       # 진한 주황색
                    'border_color': QColor(255, 255, 0, 255),     # 노란색 테두리
                    'border_width': 2,
                    'text_color': QColor(255, 255, 255),
                    'priority': 4,
                    'show_confidence': True
                })
            else:
                style.update({
                    'fill_color': QColor(255, 165, 0, 140),       # 연한 주황색
                    'border_color': QColor(255, 200, 0, 255),     # 연한 노란색 테두리
                    'border_width': 2,
                    'text_color': QColor(255, 255, 255),
                    'priority': 3,
                    'show_confidence': True
                })
        elif detection_method == 'temperature':
            # 온도 기반 감지
            style.update({
                'fill_color': QColor(255, 69, 0, 140),        # 오렌지 레드
                'border_color': QColor(255, 140, 0, 255),     # 다크 오렌지 테두리
                'border_width': 2,
                'text_color': QColor(255, 255, 255),
                'priority': 2
            })
        elif detection_method == 'MOG2':
            # MOG2 기반 감지
            style.update({
                'fill_color': QColor(30, 144, 255, 140),      # 도저블루
                'border_color': QColor(0, 191, 255, 255),     # 딥스카이블루 테두리
                'border_width': 2,
                'text_color': QColor(255, 255, 255),
                'priority': 2
            })
        elif validation_status == 'history_validated':
            # 이력 검증 통과
            style.update({
                'fill_color': QColor(34, 139, 34, 140),       # 포레스트 그린
                'border_color': QColor(0, 255, 0, 255),       # 라임 테두리
                'border_width': 2,
                'text_color': QColor(255, 255, 255),
                'priority': 3
            })
        else:
            # 기본 열원
            style.update({
                'fill_color': QColor(128, 128, 128, 120),     # 회색
                'border_color': QColor(255, 255, 255, 200),   # 흰색 테두리
                'border_width': 1,
                'text_color': QColor(255, 255, 255),
                'priority': 1
            })
        
        # 지속 감지 횟수에 따른 강조 효과
        if detection_count >= 5:
            style['border_width'] += 1
            style['circle_radius'] += 2
            style['text_size'] += 1
        
        return style

    def _draw_hotspot_circle(self, painter, pixel_x, pixel_y, style):
        """열원을 원형으로 그리기"""
        radius = style['circle_radius']
        
        # 채우기 색상 설정
        painter.setBrush(QBrush(style['fill_color']))
        
        # 테두리 설정
        pen = QPen(style['border_color'], style['border_width'])
        painter.setPen(pen)
        
        # 원 그리기
        painter.drawEllipse(pixel_x - radius, pixel_y - radius, radius * 2, radius * 2)
        
        # 중심점 표시
        painter.setPen(QPen(QColor(0, 0, 0), 3))
        painter.drawPoint(pixel_x, pixel_y)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawPoint(pixel_x, pixel_y)

    def _draw_hotspot_info(self, painter, pixel_x, pixel_y, hotspot, style):
        """열원 정보 텍스트 그리기"""
        painter.setPen(QPen(style['text_color']))
        font = QFont("Arial", style['text_size'], QFont.Weight.Bold)
        painter.setFont(font)
        
        # 기본 정보 텍스트
        hotspot_id = hotspot.get('tracker_id', hotspot.get('id', 0))
        max_temp = hotspot.get('max_temp', 0)
        detection_method = hotspot.get('detection_method', 'unknown')
        
        # 텍스트 내용 구성
        lines = []
        lines.append(f"H{hotspot_id}")
        lines.append(f"{max_temp:.1f}°C")
        
        # 감지 방법 표시
        method_short = {
            'lgbm_model': 'ML',
            'temperature': 'TEMP',
            'MOG2': 'MOG',
            'fallback_threshold': 'FB'
        }.get(detection_method, 'UNK')
        lines.append(f"[{method_short}]")
        
        # 신뢰도 표시 (해당하는 경우)
        if style.get('show_confidence', False):
            model_confidence = hotspot.get('model_confidence', 0.0)
            consensus_score = hotspot.get('consensus_score', 0.0)
            if model_confidence > 0:
                lines.append(f"ML:{model_confidence:.2f}")
            if consensus_score > 0:
                lines.append(f"CS:{consensus_score:.2f}")
        
        # 검증 상태 표시
        validation_status = hotspot.get('validation_status', 'unknown')
        cross_validation = hotspot.get('cross_validation', False)
        if cross_validation:
            lines.append("✓CV")
        elif validation_status == 'history_validated':
            lines.append("✓HV")
        
        # 텍스트 그리기 위치 계산
        radius = style['circle_radius']
        text_x = pixel_x + radius + 8
        text_y = pixel_y - len(lines) * 6  # 텍스트 높이에 따라 조정
        
        # 배경 사각형 그리기 (가독성 향상)
        text_width = 80
        text_height = len(lines) * 12 + 4
        bg_rect = QRect(text_x - 2, text_y - 2, text_width, text_height)
        painter.fillRect(bg_rect, QColor(0, 0, 0, 150))  # 반투명 검은 배경
        
        # 각 줄별로 텍스트 그리기
        for i, line in enumerate(lines):
            painter.drawText(text_x, text_y + (i + 1) * 12, line)

    def _draw_hotspot_cells(self, painter, hotspot, scale_factor, style):
        """열원을 구성하는 개별 셀들을 강조 표시"""
        if 'coordinates' not in hotspot:
            return
        
        cell_fill_color = QColor(style['fill_color'])
        cell_fill_color.setAlpha(80)  # 더 투명하게
        painter.setBrush(QBrush(cell_fill_color))
        painter.setPen(QPen(style['border_color'], 1))
        
        for orig_x, orig_y in hotspot['coordinates']:
            # 원본 8x8 좌표를 interpolated grid 좌표로 변환
            for i_interp in range(int(orig_y * scale_factor), int((orig_y + 1) * scale_factor)):
                for j_interp in range(int(orig_x * scale_factor), int((orig_x + 1) * scale_factor)):
                    if (i_interp < self.interpolated_grid_size and 
                        j_interp < self.interpolated_grid_size):
                        
                        cell_x = j_interp * self.cell_size
                        cell_y = i_interp * self.cell_size
                        
                        # 셀 사각형 그리기
                        painter.drawRect(cell_x, cell_y, self.cell_size - 1, self.cell_size - 1)


class HeatmapWidget(QWidget):
    """히트맵과 열원 감지 오버레이를 표시하는 커스텀 위젯"""
    
    def __init__(self, max_width, max_height, parent=None):
        super().__init__(parent)
        self.max_height = max_height - 40
        self.max_width = max_width        
        self.grid_cells = []
        self.hotspots = []
        self.interpolated_grid_size = 24
        self.original_grid_size = 8
        self.cell_size = self.max_height // self.interpolated_grid_size
        self.gaussian_sigma = 0.9
        self.min_temp = 19.0
        self.max_temp = 32.0
        
        self.display_temperature = False
        self.current_values = []
        
        # 메인 레이아웃 설정
        self.main_layout = QGridLayout(self)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0,0,0,0)
        
        # 오버레이 위젯 생성
        self.overlay = HeatmapOverlay(self)
        
        self.init_heatmap()

    def resizeEvent(self, event):
        """위젯 크기 변경 시 오버레이 크기도 조정"""
        super().resizeEvent(event)
        if hasattr(self, 'overlay'):
            self.overlay.resize(self.size())

    def init_heatmap(self):
        """히트맵 그리드 초기화"""
        self.clear_grid()
        
        self.grid_cells = []
        
        for i in range(self.interpolated_grid_size):
            row_cells = []
            for j in range(self.interpolated_grid_size):
                cell = QLabel()
                cell.setFixedSize(self.cell_size, self.cell_size)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setProperty("class", "HeatmapCell")
                cell.setStyleSheet("background-color: lightgray; font-size: 10px")
                self.main_layout.addWidget(cell, i, j)
                row_cells.append(cell)
            self.grid_cells.append(row_cells)
        
        # 오버레이 위젯을 맨 위에 배치
        self.overlay.resize(self.size())
        self.overlay.raise_()
        
    def clear_grid(self):
        """기존 그리드 셀들 제거"""
        if hasattr(self, 'grid_cells'):
            for row in self.grid_cells:
                for cell in row:
                    cell.deleteLater()
            self.grid_cells = []

    def update_heatmap(self, values, hotspots=None):
        """히트맵 업데이트 (열원은 오버레이로만 표시)"""
        if len(values) != 64:
            return
            
        self.current_values = values
        self.hotspots = hotspots or []
        
        # 8x8 배열로 변환
        original_data = np.array(values).reshape((self.original_grid_size, self.original_grid_size))
        
        # 가우시안 필터 적용
        filtered_data = gaussian_filter(original_data, sigma=self.gaussian_sigma)
        
        # 보간을 통해 확대
        scale_factor = self.interpolated_grid_size // self.original_grid_size
        interpolated_grid = np.zeros((self.interpolated_grid_size, self.interpolated_grid_size))
        
        for i_orig in range(self.original_grid_size):
            for j_orig in range(self.original_grid_size):
                value = filtered_data[i_orig, j_orig]
                for i_interp in range(i_orig * scale_factor, (i_orig + 1) * scale_factor):
                    for j_interp in range(j_orig * scale_factor, (j_orig + 1) * scale_factor):
                        if i_interp < self.interpolated_grid_size and j_interp < self.interpolated_grid_size:
                            interpolated_grid[i_interp, j_interp] = value
        
        # 추가 가우시안 필터 적용 (부드러운 표시)
        interpolated_grid = gaussian_filter(interpolated_grid, sigma=self.gaussian_sigma)
        
        # 셀 업데이트 (모든 셀을 일반 스타일로 처리)
        font_size = max(8, self.cell_size // 4)
        
        for i in range(self.interpolated_grid_size):
            for j in range(self.interpolated_grid_size):
                if i < len(self.grid_cells) and j < len(self.grid_cells[i]):
                    value = interpolated_grid[i, j]
                    color = self.get_color_from_value(value)
                    cell = self.grid_cells[i][j]
                    
                    text = f"{value:.1f}" if self.display_temperature else ""
                    
                    # 모든 셀을 일반 스타일로 처리 (열원 강조 제거)
                    normal_style = f"""
                        background-color: {color.name()}; 
                        color: black; 
                        font-weight: bold; 
                        font-size: {font_size}px; 
                        border: none;
                    """
                    cell.setText(text)
                    cell.setStyleSheet(normal_style)
                    cell.setToolTip("")  # 툴팁 제거
        
        # 오버레이에 열원 정보 전달
        self.overlay.set_hotspots(self.hotspots, self.cell_size, self.interpolated_grid_size)

    def get_color_from_value(self, value):
        """온도 값에 따른 색상 반환"""
        temp = float(value)
        if (self.max_temp - self.min_temp) == 0:
            normalized_temp = 0.0
        else:
            normalized_temp = (temp - self.min_temp) / (self.max_temp - self.min_temp)

        if normalized_temp <= 0.2:
            red, green, blue = 0, int(255 * (5 * normalized_temp)), 255
        elif normalized_temp <= 0.4:
            red, green, blue = 0, 255, int(255 * (1 - (5 * (normalized_temp - 0.2))))
        elif normalized_temp <= 0.6:
            red, green, blue = int(255 * (5 * (normalized_temp - 0.4))), 255, 0
        elif normalized_temp <= 0.8:
            red, green, blue = 255, int(255 * (1 - (5 * (normalized_temp - 0.6)))), 0
        else:
            red, green, blue = 255, 0, 0

        r = int(max(0, min(255, red)))
        g = int(max(0, min(255, green)))
        b = int(max(0, min(255, blue)))

        return QColor(r, g, b)

    def set_display_temperature(self, display):
        """온도 표시 여부 설정"""
        self.display_temperature = display
        if self.current_values:
            self.update_heatmap(self.current_values, self.hotspots)

    def set_temp_range(self, min_temp, max_temp):
        """온도 범위 설정"""
        self.min_temp = min_temp
        self.max_temp = max_temp
        if self.current_values:
            self.update_heatmap(self.current_values, self.hotspots)

    def set_grid_size(self, size):
        """그리드 크기 설정"""
        if size != self.interpolated_grid_size:
            self.interpolated_grid_size = size
            self.cell_size = self.max_height // self.interpolated_grid_size
            self.init_heatmap()
            if self.current_values:
                self.update_heatmap(self.current_values, self.hotspots)

class OutputModule(QWidget):
    """메인 GUI 모듈"""
    
    ORIGINAL_GRID_SIZE = 8
    DEFAULT_INTERPOLATED_GRID_SIZE = 24

    def __init__(self, data_signal_obj, available_rect):
        super().__init__()
        self.max_width = available_rect.width()
        self.max_height = available_rect.height()
        self.data_signal = data_signal_obj
        self.anomaly_count = 0
        self.log_filename = "./detected_values.txt"
        self.fire_alert_triggered = False
        self.smoke_alert_triggered = False
        
        # 데이터 신호 연결
        self.data_signal.update_data_signal.connect(self.update_display)
        
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        self.setObjectName("MainWindow")
        self.setWindowTitle('통합 관제 시스템')

        main_layout = QGridLayout(self)
        main_layout.setSpacing(15)

        
        
        # 왼쪽 패널 생성
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 0,0 ,2 ,1)

        # 히트맵 위젯 생성
        self.heatmap_widget = HeatmapWidget(self.max_width, self.max_height)
        main_layout.addWidget(self.heatmap_widget, 0, 1, 2, 1)

        # main_layout.setColumnStretch(1, 1)
        # main_layout.setRowStretch(0, 1)

    def create_left_panel(self):
        """왼쪽 정보 패널 생성"""
        left_panel = QWidget()
        left_panel.setObjectName("LeftPanel")
        # left_panel.setFixedWidth(self.max_width - self.max_height)
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setSpacing(5)

        # 정보 라벨들
        self.sensor_temp_label = QLabel("센서 온도: N/A")
        self.sensor_temp_label.setProperty("class", "TitleLabel")
        
        self.max_temp_label = QLabel("최고 온도: N/A")
        self.max_temp_label.setProperty("class", "TitleLabel")
        
        self.avg_temp_label = QLabel("평균 온도: N/A")
        self.avg_temp_label.setProperty("class", "TitleLabel")
        
        self.etc_label = QLabel("기타: N/A")
        self.etc_label.setProperty("class", "TitleLabel")
        
        self.anomaly_count_label = QLabel(f"이상 감지: {self.anomaly_count} 회")
        self.anomaly_count_label.setObjectName("AnomalyCountLabel")

        # 상태 표시기
        fire_layout, self.fire_indicator = self._create_status_row("화재 감지")
        smoke_layout, self.smoke_indicator = self._create_status_row("연기 감지")

        self.humidity_label = QLabel("습도: 55.0%")
        self.humidity_label.setProperty("class", "InfoLabel")

        # 열원 정보 라벨
        self.hotspot_info_label = QLabel("감지된 열원: 0개")
        self.hotspot_info_label.setProperty("class", "InfoLabel")
        
        # 온도 표시 체크박스
        display_degree = QCheckBox("온도 표시", self)
        display_degree.stateChanged.connect(self.display_degree_control)

        # 온도 범위 조절 위젯
        temp_range_group_box = self.create_temp_range_widget()
        
        # 그리드 크기 조절 위젯
        grid_size_group_box = self.create_grid_size_widget()

        # 로그 버튼
        self.log_button = QPushButton("이상 감지 정보 자세히 보기")
        self.log_button.clicked.connect(self.open_log_file)

        # 시간 라벨
        self.time_label = QLabel("시간: N/A")
        self.time_label.setFont(QFont("Arial", 9))

        # 레이아웃에 위젯 추가
        left_panel_layout.addWidget(self.sensor_temp_label)
        left_panel_layout.addWidget(self.max_temp_label)
        left_panel_layout.addWidget(self.avg_temp_label)
        left_panel_layout.addWidget(self.etc_label)
        left_panel_layout.addWidget(self.anomaly_count_label)
        left_panel_layout.addLayout(fire_layout)
        left_panel_layout.addLayout(smoke_layout)
        left_panel_layout.addWidget(self.humidity_label)
        left_panel_layout.addWidget(self.hotspot_info_label)
        left_panel_layout.addWidget(display_degree)
        left_panel_layout.addWidget(temp_range_group_box)
        left_panel_layout.addWidget(grid_size_group_box)
        # left_panel_layout.addStretch(1)
        left_panel_layout.addWidget(self.log_button)
        left_panel_layout.addWidget(self.time_label)

        return left_panel

    def create_temp_range_widget(self):
        """온도 범위 조절 위젯 생성"""
        temp_range_group_box = QWidget()
        temp_range_group_box.setObjectName("TopPanel")
        temp_range_layout = QVBoxLayout(temp_range_group_box)
        
        temp_range_title = QLabel("히트맵 온도 범위")
        temp_range_title.setProperty("class", "TitleLabel")
        
        # 최저/최고 온도 조절 UI
        min_temp_layout, self.min_temp_spinbox = self._create_temp_control_row("최저", 19.0)
        max_temp_layout, self.max_temp_spinbox = self._create_temp_control_row("최고", 32.0)

        # 시그널 연결
        self.min_temp_spinbox.valueChanged.connect(lambda value: self._update_temp_range('min', value))
        self.max_temp_spinbox.valueChanged.connect(lambda value: self._update_temp_range('max', value))

        temp_range_layout.addWidget(temp_range_title)
        temp_range_layout.addLayout(min_temp_layout)
        temp_range_layout.addLayout(max_temp_layout)
        
        return temp_range_group_box

    def create_grid_size_widget(self):
        """그리드 크기 조절 위젯 생성"""
        grid_size_group_box = QWidget()
        grid_size_group_box.setObjectName("TopPanel")
        grid_size_layout = QVBoxLayout(grid_size_group_box)
        grid_size_layout.setSpacing(0)
        grid_size_up_layout = QHBoxLayout()
        grid_size_up_layout.setSpacing(0)

        self.grid_label_prefix = QLabel("그리드 해상도:")
        self.grid_label_prefix.setProperty("class", "TitleLabel")

        self.grid_size_spinbox = QSpinBox()
        self.grid_size_spinbox.setRange(self.ORIGINAL_GRID_SIZE, self.ORIGINAL_GRID_SIZE * 8)
        self.grid_size_spinbox.setSingleStep(self.ORIGINAL_GRID_SIZE)
        self.grid_size_spinbox.setValue(self.DEFAULT_INTERPOLATED_GRID_SIZE)
        self.grid_size_spinbox.valueChanged.connect(self._update_grid_size_from_spinbox)
        self.grid_size_spinbox.setStyleSheet("background-color:black")
        self.grid_size_spinbox.setFont(QFont("Arial", 14))

        grid_size_up_layout.addWidget(self.grid_label_prefix)
        grid_size_up_layout.addStretch(1)
        grid_size_up_layout.addWidget(self.grid_size_spinbox)

        grid_size_layout.addLayout(grid_size_up_layout)
        
        return grid_size_group_box

    def _create_status_row(self, text):
        """상태 표시기 행 생성"""
        layout = QHBoxLayout()
        label = QLabel(text)
        label.setProperty("class", "InfoLabel")
        indicator = QLabel()
        indicator.setProperty("styleClass", "Indicator")
        indicator.setProperty("state", "stable")
        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(indicator)
        return layout, indicator

    def _create_temp_control_row(self, label_text, initial_value):
        """온도 조절 UI 행 생성"""
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setProperty("class", "InfoLabel")

        dec_button = QPushButton("-")
        dec_button.setProperty("class", "AdjustButton")

        spin_box = QDoubleSpinBox()
        spin_box.setRange(-50.0, 100.0)
        spin_box.setSingleStep(0.1)
        spin_box.setDecimals(1)
        spin_box.setValue(initial_value)

        inc_button = QPushButton("+")
        inc_button.setProperty("class", "AdjustButton")

        # 버튼 클릭 시그널 연결
        dec_button.clicked.connect(lambda: self._adjust_temp(spin_box, -0.1))
        inc_button.clicked.connect(lambda: self._adjust_temp(spin_box, 0.1))

        layout.addWidget(label)
        layout.addStretch(1)
        layout.addWidget(dec_button)
        layout.addWidget(spin_box)
        layout.addWidget(inc_button)

        return layout, spin_box

    def display_degree_control(self, state):
        """온도 표시 제어"""
        self.heatmap_widget.set_display_temperature(state)

    def _adjust_temp(self, spin_box, amount):
        """온도 조절 버튼 처리"""
        current_value = spin_box.value()
        spin_box.setValue(current_value + amount)

    def _update_temp_range(self, temp_type, value):
        """온도 범위 업데이트"""
        if temp_type == 'min':
            min_temp = value
            max_temp = self.max_temp_spinbox.value()
        else:
            min_temp = self.min_temp_spinbox.value()
            max_temp = value
        
        self.heatmap_widget.set_temp_range(min_temp, max_temp)

    def _update_grid_size_from_spinbox(self, value):
        """그리드 크기 업데이트"""
        self.heatmap_widget.set_grid_size(value)

    def update_display(self, data_package):
        """디스플레이 업데이트 (Detector로부터 데이터 받음)"""
        sensor_degree = data_package.get('sensor_degree', 'N/A')
        current_time = data_package.get('time', 'N/A')
        values = data_package.get('values', [])
        etc = data_package.get('etc', [])
        fire_detected = data_package.get('fire_detected', False)
        smoke_detected = data_package.get('smoke_detected', False)
        hotspots = data_package.get('hotspots', [])
        detection_stats = data_package.get('detection_stats', {})

        # 시간 업데이트
        self.time_label.setText(f"시간: {current_time}")

        # 온도 정보 업데이트
        if values and detection_stats:
            self.sensor_temp_label.setText(f"센서 온도: {sensor_degree:.1f}°C")
            self.max_temp_label.setText(f"최고 온도: {detection_stats.get('max_temp', 0):.1f}°C")
            self.avg_temp_label.setText(f"평균 온도: {detection_stats.get('avg_temp', 0):.1f}°C")
            self.etc_label.setText(f"기타: {etc}")
            # HeatmapWidget.avg_temp = f"{detection_stats.get('avg_temp', 0):.1f}"
        else:
            self.max_temp_label.setText("최고 온도: N/A")
            self.avg_temp_label.setText("평균 온도: N/A")

        # 이상 감지 횟수 업데이트
        anomaly_count = data_package.get('anomaly_count', 0)
        if anomaly_count != self.anomaly_count:
            self.anomaly_count = anomaly_count
            self.anomaly_count_label.setText(f"이상 감지: {self.anomaly_count} 회")

        # 상태 표시기 업데이트
        self.update_indicator_status(self.fire_indicator, fire_detected)
        self.update_indicator_status(self.smoke_indicator, smoke_detected)

        # 열원 정보 업데이트
        if hotspots:
            # print(hotspots)
            hotspot_info = f"감지된 열원: {len(hotspots)}개"
            if len(hotspots) > 0:
                max_temp_hotspot = max(hotspots, key=lambda x: x['max_temp'])
                hotspot_info += f" (최고: {max_temp_hotspot['max_temp']:.1f}°C)"
                # hotspot_info += f" (최고: {max_temp_hotspot['max_temp']:.1f}°C, center: {hotspots[0]['center'][0]:.1f},{hotspots[0]['center'][1]:.1f})"
            self.hotspot_info_label.setText(hotspot_info)
        else:
            self.hotspot_info_label.setText("감지된 열원: 0개")

        # 경고 처리
        # is_anomaly = fire_detected or smoke_detected
        is_anomaly = None
        if is_anomaly:
            if fire_detected and not self.fire_alert_triggered:
                self.handle_anomaly("화재", current_time)
                self.fire_alert_triggered = True
            if smoke_detected and not self.smoke_alert_triggered:
                self.handle_anomaly("연기", current_time)
                self.smoke_alert_triggered = True

        if not fire_detected:
            self.fire_alert_triggered = False
        if not smoke_detected:
            self.smoke_alert_triggered = False

        # 히트맵 업데이트
        if values:
            self.heatmap_widget.update_heatmap(values, hotspots)

    def handle_anomaly(self, anomaly_type, current_time):
        """이상 상황 처리"""
        QApplication.processEvents()
        self.show_alert_popup(f"{anomaly_type}가 감지되었습니다!", 
                             f"시간: {current_time}\n시스템 로그를 확인하세요.")

    def update_indicator_status(self, indicator_widget, detected):
        """상태 표시기 업데이트"""
        new_state = "detected" if detected else "stable"
        if indicator_widget.property("state") != new_state:
            indicator_widget.setProperty("state", new_state)
            indicator_widget.style().unpolish(indicator_widget)
            indicator_widget.style().polish(indicator_widget)

    def show_alert_popup(self, title, message):
        """경고 팝업 표시"""
        alert = QMessageBox(self)
        alert.setIcon(QMessageBox.Icon.Warning)
        alert.setWindowTitle("! 경고 !")
        alert.setText(title)
        alert.setInformativeText(message)
        alert.setStandardButtons(QMessageBox.StandardButton.Ok)
        alert.exec()

    def open_log_file(self):
        """로그 파일 열기"""
        if not os.path.exists(self.log_filename):
            self.show_alert_popup("파일 없음", f"로그 파일({self.log_filename})이 아직 생성되지 않았습니다.")
            return
        if sys.platform == "win32":
            os.startfile(self.log_filename)
        elif sys.platform == "darwin":
            subprocess.run(["open", self.log_filename])
        else:
            subprocess.run(["xdg-open", self.log_filename])