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
        # print(f"Overlay: Setting {len(self.hotspots)} hotspots, cell_size: {cell_size}")
        self.update()  # 다시 그리기 요청

    def paintEvent(self, event):
        """열원 감지 결과를 오버레이로 표시"""
        super().paintEvent(event)
        
        # print(f"Overlay paintEvent called, hotspots: {len(self.hotspots)}")
        
        if not self.hotspots or self.cell_size == 0:
            return
        
        if self.hotspots[0]['size'] < 6 or self.hotspots[0]['size'] > 20 :
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 스케일 팩터 계산 (8x8 -> interpolated_grid_size)
        scale_factor = self.interpolated_grid_size / self.original_grid_size
        
        for hotspot in self.hotspots:
            try:
                center_x, center_y = hotspot['center']
                
                # 8x8 좌표를 interpolated_grid_size 좌표로 변환
                scaled_x = center_x * scale_factor
                scaled_y = center_y * scale_factor
                
                # 픽셀 좌표로 변환
                pixel_x = int(scaled_x * self.cell_size + self.cell_size / 2)
                pixel_y = int(scaled_y * self.cell_size + self.cell_size / 2)
                
                # 열원 크기에 따른 원 크기 결정
                circle_radius = max(100, min(100, hotspot.get('size', 1) * 5))
                
                # 온도에 따른 색상 결정
                max_temp = hotspot.get('max_temp', 25)
                if max_temp > 35:
                    color = QColor(255, 0, 0, 150)  # 빨간색 (고온)
                elif max_temp > 30:
                    color = QColor(255, 165, 0, 150)  # 주황색 (중온)
                else:
                    color = QColor(255, 255, 0, 150)  # 노란색 (저온)
                
                # 외곽선과 채우기 설정
                painter.setPen(QPen(QColor(255, 255, 255), 3))
                painter.setBrush(QBrush(color))
                
                # 원 그리기
                painter.drawEllipse(pixel_x - circle_radius, pixel_y - circle_radius, 
                                  circle_radius * 2, circle_radius * 2)
                
                # 중심점 표시
                painter.setPen(QPen(QColor(0, 0, 0), 4))
                painter.drawPoint(pixel_x, pixel_y)
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawPoint(pixel_x, pixel_y)
                
                # 열원 ID와 온도 표시
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                text = f"H{hotspot.get('id', 0)}\n{max_temp:.1f}°C"
                painter.drawText(pixel_x + circle_radius + 5, pixel_y - 5, text)
                
                # print(f"Drew hotspot {hotspot.get('id', 0)} at ({pixel_x}, {pixel_y}) with radius {circle_radius}")
                
            except Exception as e:
                print(f"Error drawing hotspot {hotspot}: {e}")

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
        """히트맵 업데이트"""
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
        
        # 셀 업데이트
        avg_value = interpolated_grid.mean()
        font_size = max(8, self.cell_size // 4)
        for i in range(self.interpolated_grid_size):
            for j in range(self.interpolated_grid_size):
                if i < len(self.grid_cells) and j < len(self.grid_cells[i]):
                    value = interpolated_grid[i, j]
                    color = self.get_color_from_value(value)
                    cell = self.grid_cells[i][j]
                    
                    text = f"{value:.1f}" if self.display_temperature else ""
                    cell.setText(text)
                    cell.setStyleSheet(f"""
                        background-color: {color.name()}; 
                        color: black; 
                        font-weight: bold; 
                        font-size: {font_size}px; 
                        border: none;
                    """)

                    if text != "":
                        if float(text) < float(avg_value) + 0.5:
                            cell.setText(text)
                            cell.setStyleSheet(f"""
                                background-color: black; 
                                color: white; 
                                font-weight: bold; 
                                font-size: {font_size}px; 
                                border: none;
                            """)
        
        # 오버레이 위젯에 열원 정보 전달
        # print(f"Updating overlay with {len(self.hotspots)} hotspots")
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
            # print('size', size)
            self.interpolated_grid_size = size
            self.cell_size = self.max_height // self.interpolated_grid_size
            # print(self.cell_size)
            self.init_heatmap()
            # print('where')
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
        self.log_filename = "detected_values.txt"
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
        is_anomaly = fire_detected or smoke_detected
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