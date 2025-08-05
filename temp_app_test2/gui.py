""" 
250804 리팩토링 완료:
- 열원 필터링, 가중치 적용, 보간 처리를 detector에서 수행
- GUI는 처리된 데이터를 받아서 표시만 담당
- 파라미터 변경 시 detector에 업데이트 신호 전송

250805 추가 수정사항:
        - 중앙 센서와 별개로, 모서리 센서의 경우 별도의 보정값 필요함.
        - 1~8 단계로 센서값을 부여할 필요
        - 5.5 열 까지는 적당히 측정되지만, 이후부터는 아슬아슬하게 감지됨
        - 7,7열의 경우 높이 170cm 쯤에서 활성화됨.
        - 센서의 정반대, 가장 먼 곳의 3칸은 측정되지 않음.
"""

import sys
import os
import subprocess
import numpy as np
import random
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QGridLayout, QMessageBox, QPushButton,
                             QSlider, QSpinBox, QDoubleSpinBox, QCheckBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint, QThread
from PyQt6.QtGui import QColor, QFont, QScreen, QMouseEvent
import queue
import time

# === 1920x1080 해상도에 맞춰 폰트 크기 상향 조정 ===
GLOBAL_STYLESHEET = """
/* 전체 위젯 기본 스타일 */
QWidget {
    color: #E0E0E0;
    font-family: 'Malgun Gothic', 'Segoe UI', 'Arial';
}

/* 메인 윈도우 스타일 */
#MainWindow {
    background-color: #2E2E2E;
}

/* 정보 패널(카드) 스타일 */
#TopPanel, #LeftPanel {
    background-color: #3C3C3C;
    border-radius: 8px;
    padding: 10px;
}

/* 제목 레이블 스타일 */
QLabel[class="TitleLabel"] {
    font-size: 12px;
    font-weight: bold;
    color: #FFFFFF;
    padding: 2px;
}

/* 일반 정보 레이블 스타일 */
QLabel[class="InfoLabel"] {
    font-size: 12px;
    font-weight: bold;
}

/* 이상 감지 카운트 레이블 특별 스타일 */
#AnomalyCountLabel {
    font-size: 16px;
    font-weight: bold;
    color: #F44336;
    padding: 2px;
}

/* 히트맵 셀 스타일 */
QLabel[class="HeatmapCell"] {
    font-weight: bold;
    border: none;
}

/* 버튼 스타일 */
QPushButton {
    background-color: #555555;
    color: #FFFFFF;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #6A6A6A;
}
QPushButton:pressed {
    background-color: #4A4A4A;
}

/* 온도 조절 버튼 작은 스타일 */
QPushButton[class="AdjustButton"] {
    font-size: 12px;
    font-weight: bold;
    padding: 4px 8px;
    min-width: 30px;
}

/* QDoubleSpinBox 스타일 */
QDoubleSpinBox {
    background-color: #2E2E2E;
    color: #FFFFFF;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 5px;
    font-size: 12px;
    font-weight: bold;
}

/* 상태 표시기(LED) 스타일 */
QLabel[styleClass="Indicator"] {
    min-width: 20px;
    max-width: 20px;
    min-height: 20px;
    max-height: 20px;
    border-radius: 10px;
    border: 1px solid rgba(0, 0, 0, 100);
}

/* 상태에 따른 색상 */
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
    update_data_signal = pyqtSignal(dict)
    # DetectionModule로 파라미터 업데이트를 전송하는 시그널
    parameter_update_signal = pyqtSignal(dict)
    anomaly_detected_signal = pyqtSignal(int)

class DraggableNumberLabel(QLabel):
    """
    마우스 드래그로 숫자를 변경할 수 있는 사용자 정의 QLabel 위젯.
    """
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


class OutputModule(QWidget):
    ORIGINAL_GRID_SIZE = 8  # 원본 데이터 그리드 크기 (고정)
    DEFAULT_INTERPOLATED_GRID_SIZE = 8  # 초기 보간된 그리드 크기

    def __init__(self, data_signal_obj, available_rect):
        super().__init__()
        self.max_width = available_rect.width()
        self.max_height = available_rect.height() - 120

        # 클래스 변수 대신 인스턴스 변수로 관리
        self.original_grid_size = self.ORIGINAL_GRID_SIZE
        self.interpolated_grid_size = self.DEFAULT_INTERPOLATED_GRID_SIZE
        self.cell_size = self.max_height // self.interpolated_grid_size
        self.font_size = self.cell_size // 3

        self.anomaly_count = 0
        self.log_filename = "detected_values.txt"
        self.fire_alert_triggered = False
        self.smoke_alert_triggered = False
        self.data_signal = data_signal_obj
        self.grid_cells = []
        self.heatmap_layout = None
        self.current_data_package = None
        
        # 시그널 연결
        self.data_signal.update_data_signal.connect(self.update_display)
        self.data_signal.anomaly_detected_signal.connect(self.update_anomaly_count)

        # === 히트맵 온도 범위 인스턴스 변수 (detector와 동기화) ===
        self.min_temp = 19.0
        self.max_temp = 32.0
        self.avg_temp = 0
        self.filter_temp_add = 5
        self.filter_temp = 0
        self.weight_list = [3.8, 4, 4.3, 5]

        # == 온도 출력 on off == 
        self.display_degree = True
        
        self.init_ui()

    def update_anomaly_count(self, count):
        self.anomaly_count = count
        self.anomaly_count_label.setText(f"이상 감지: {self.anomaly_count} 회")

    def init_ui(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle('temp_app_test 통합 관제 시스템')

        main_layout = QGridLayout(self)
        main_layout.setSpacing(15)

        left_panel = QWidget()
        left_panel.setObjectName("LeftPanel")
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setSpacing(5)

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

        fire_layout, self.fire_indicator = self._create_status_row("화재 감지")
        smoke_layout, self.smoke_indicator = self._create_status_row("연기 감지")
        
        self.object_detection_label = QLabel("객체: N/A")
        self.object_detection_label.setProperty("class", "TitleLabel")
        
        self.test_label = QLabel("test: N/A")
        self.test_label.setProperty("class", "TitleLabel")

        self.humidity_label = QLabel("습도: 55.0%")
        self.humidity_label.setProperty("class", "InfoLabel")
        
        # -- 온도 표시 위젯 --
        display_degree = QCheckBox("온도 표시", self)
        display_degree.stateChanged.connect(self.display_degree_control)
                
        # --- 히트맵 온도 범위 조절 위젯 ---
        temp_range_group_box = QWidget()
        temp_range_group_box.setObjectName("TopPanel")
        temp_range_layout = QVBoxLayout(temp_range_group_box)
        
        temp_range_title = QLabel("히트맵 온도 범위")
        temp_range_title.setProperty("class", "TitleLabel")
        
        # 최저 온도 조절 UI
        min_temp_layout, self.min_temp_spinbox = self._create_temp_control_row("최저", self.min_temp)
        # 최고 온도 조절 UI
        max_temp_layout, self.max_temp_spinbox = self._create_temp_control_row("최고", self.max_temp)
        
        # 필터링 적용 온도 조절 UI
        filter_temp_layout, self.filter_temp_spinbox = self._create_temp_control_row("기준", self.filter_temp_add)
        # 가중치 조절 UI
        weight_layout1, self.weight1_spinbox = self._create_temp_control_row("1", self.weight_list[0])
        weight_layout2, self.weight2_spinbox = self._create_temp_control_row("2", self.weight_list[1])
        weight_layout3, self.weight3_spinbox = self._create_temp_control_row("3", self.weight_list[2])
        weight_layout4, self.weight4_spinbox = self._create_temp_control_row("4", self.weight_list[3])
        

        # Signal 연결 - detector에 파라미터 업데이트 전송
        self.min_temp_spinbox.valueChanged.connect(lambda value: self._update_temp_range('min', value))
        self.max_temp_spinbox.valueChanged.connect(lambda value: self._update_temp_range('max', value))
        
        self.filter_temp_spinbox.valueChanged.connect(lambda value: self._update_filter_weight('filter_add', value))
        self.weight1_spinbox.valueChanged.connect(lambda value: self._update_filter_weight('w1', value))
        self.weight2_spinbox.valueChanged.connect(lambda value: self._update_filter_weight('w2', value))
        self.weight3_spinbox.valueChanged.connect(lambda value: self._update_filter_weight('w3', value))
        self.weight4_spinbox.valueChanged.connect(lambda value: self._update_filter_weight('w4', value))

        temp_range_layout.addWidget(temp_range_title)
        temp_range_layout.addLayout(min_temp_layout)
        temp_range_layout.addLayout(max_temp_layout)
        temp_range_layout.addLayout(filter_temp_layout)
        temp_range_layout.addLayout(weight_layout1)
        temp_range_layout.addLayout(weight_layout2)
        temp_range_layout.addLayout(weight_layout3)
        temp_range_layout.addLayout(weight_layout4)

        # --- 그리드 크기 조절 위젯 ---
        grid_size_group_box = QWidget()
        grid_size_group_box.setObjectName("TopPanel")
        grid_size_layout = QVBoxLayout(grid_size_group_box)
        grid_size_up_layout = QHBoxLayout()

        self.grid_label_prefix = QLabel("그리드 해상도:")
        self.grid_label_prefix.setProperty("class", "TitleLabel")

        self.grid_size_spinbox = QSpinBox()
        self.grid_size_spinbox.setRange(self.ORIGINAL_GRID_SIZE, self.ORIGINAL_GRID_SIZE * 8)
        self.grid_size_spinbox.setSingleStep(self.ORIGINAL_GRID_SIZE)
        self.grid_size_spinbox.setValue(self.interpolated_grid_size)
        self.grid_size_spinbox.valueChanged.connect(self._update_grid_size_from_spinbox)
        self.grid_size_spinbox.setStyleSheet(f"background-color:black")
        self.grid_size_spinbox.setFont(QFont("Arial", 14))

        self.grid_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.grid_size_slider.setRange(self.ORIGINAL_GRID_SIZE, self.ORIGINAL_GRID_SIZE * 8)
        self.grid_size_slider.setSingleStep(self.ORIGINAL_GRID_SIZE)
        self.grid_size_slider.setValue(self.interpolated_grid_size)
        self.grid_size_slider.valueChanged.connect(self._update_grid_size_from_slider)

        grid_size_up_layout.addWidget(self.grid_label_prefix)
        grid_size_up_layout.addStretch(1)
        grid_size_up_layout.addWidget(self.grid_size_spinbox)

        grid_size_layout.addLayout(grid_size_up_layout)

        self.log_button = QPushButton("이상 감지 정보 자세히 보기")
        self.log_button.clicked.connect(self.open_log_file)

        self.time_label = QLabel("시간: N/A")
        self.time_label.setFont(QFont("Arial", 9))

        left_panel_layout.addWidget(self.sensor_temp_label)
        left_panel_layout.addWidget(self.max_temp_label)
        left_panel_layout.addWidget(self.avg_temp_label)
        left_panel_layout.addWidget(self.etc_label)
        left_panel_layout.addWidget(self.anomaly_count_label)
        left_panel_layout.addLayout(fire_layout)
        left_panel_layout.addLayout(smoke_layout)
        left_panel_layout.addWidget(self.object_detection_label)
        left_panel_layout.addWidget(self.test_label)
        left_panel_layout.addWidget(self.humidity_label)
        left_panel_layout.addWidget(display_degree)
        left_panel_layout.addWidget(temp_range_group_box)
        left_panel_layout.addWidget(grid_size_group_box)
        left_panel_layout.addStretch(1)
        left_panel_layout.addWidget(self.log_button)
        left_panel_layout.addWidget(self.time_label)

        main_layout.addWidget(left_panel, 0, 0, 2, 1)

        self.heatmap_layout = QGridLayout()
        self.heatmap_layout.setSpacing(0)
        self.heatmap_layout.setContentsMargins(0,0,0,0)
        main_layout.addLayout(self.heatmap_layout, 0, 1, 2, 1)

        # 초기 히트맵 생성
        self._create_heatmap_cells()

        main_layout.setColumnStretch(1, 1)
        main_layout.setRowStretch(0, 1)
    
    def display_degree_control(self, state):
        self.display_degree = state
        
    # === 온도 조절 UI 생성 헬퍼 함수 ===
    def _create_temp_control_row(self, label_text, initial_value):
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
    
    def _adjust_temp(self, spin_box, amount):
        current_value = spin_box.value()
        spin_box.setValue(current_value + amount)

    def _update_temp_range(self, temp_type, value):
        """온도 범위 업데이트 시 detector에 파라미터 전송"""
        if temp_type == 'min':
            self.min_temp = value
        elif temp_type == 'max':
            self.max_temp = value

        # detector에 파라미터 업데이트 전송
        params = {
            'min_temp': self.min_temp,
            'max_temp': self.max_temp
        }
        self.data_signal.parameter_update_signal.emit(params)
    
    def _update_filter_weight(self, name, value):
        """필터링 파라미터 업데이트 시 detector에 전송"""
        if name == 'filter_add':
           self.filter_temp_add = value
           print(f'filter_add {value} 갱신')
            
        if name == 'w1':
            self.weight_list[0] = value
            print(f'w1 {value} 갱신')
        elif name == 'w2':
            self.weight_list[1] = value
            print(f'w2 {value} 갱신')
        elif name == 'w3':
            self.weight_list[2] = value
            print(f'w3 {value} 갱신')
        elif name == 'w4':
            self.weight_list[3] = value
            print(f'w4 {value} 갱신')

        # detector에 파라미터 업데이트 전송
        params = {
            'filter_temp_add': self.filter_temp_add,
            'weight_list': self.weight_list.copy()
        }
        self.data_signal.parameter_update_signal.emit(params)

    def _create_heatmap_cells(self):
        # 기존 셀들을 모두 제거
        if self.grid_cells:
            for row in self.grid_cells:
                for cell in row:
                    self.heatmap_layout.removeWidget(cell)
                    cell.deleteLater()
                    
        # grid_cells 리스트 초기화
        self.grid_cells = []
        
        # 현재 화면 높이에 맞춰 셀 크기 재계산
        self.cell_size = self.max_height // self.interpolated_grid_size
        self.font_size = self.cell_size // 3 
        
        for i in range(self.interpolated_grid_size):
            row_cells = []
            for j in range(self.interpolated_grid_size):
                cell = QLabel()
                cell.setFixedSize(self.cell_size, self.cell_size)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setProperty("class", "HeatmapCell")
                
                # 셀 스타일 설정
                cell.setStyleSheet(
                    f"background-color: lightgray; "
                    f"font-size: {self.font_size}px; "
                    f"border: 1px solid #333;"
                )
                
                self.heatmap_layout.addWidget(cell, i, j)
                row_cells.append(cell)
            self.grid_cells.append(row_cells)
        
        # 그리드 레이아웃의 마지막 행에 확장 공간을 추가
        self.heatmap_layout.setRowStretch(self.interpolated_grid_size, 1)
        
    def _update_grid_size_from_spinbox(self, value):
        self.grid_size_slider.setValue(value)
        self._set_grid_size(value)
        
    def _update_grid_size_from_slider(self, value):
        self.grid_size_spinbox.setValue(value)
        self._set_grid_size(value)

    def _set_grid_size(self, new_size):
        if self.interpolated_grid_size != new_size:
            self.interpolated_grid_size = new_size
            self._create_heatmap_cells()

            # detector에 그리드 크기 업데이트 전송
            params = {'interpolated_grid_size': new_size}
            self.data_signal.parameter_update_signal.emit(params)

            # 현재 데이터로 히트맵 다시 업데이트
            if self.current_data_package:
                processed_values = self.current_data_package.get('processed_values', [])
                if processed_values:
                    self.update_heatmap(processed_values)

    def _create_status_row(self, text):
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

    def update_display(self, data_package):
        """detector에서 처리된 데이터를 받아서 표시"""
        # 현재 데이터 패키지를 저장
        self.current_data_package = data_package

        sensor_degree = data_package.get('sensor_degree', 'N/A')
        current_time = data_package.get('time', 'N/A')
        values = data_package.get('values', [])  # 원본 값
        processed_values = data_package.get('processed_values', [])  # 처리된 값
        etc = data_package.get('etc', [])
        object_detection = data_package.get('object_detection', [])
        detection_stats = data_package.get('detection_stats', {})
        processing_params = data_package.get('processing_params', {})
        
        # detector에서 처리된 화재/연기 감지 결과
        fire_detected = data_package.get('fire_detected', False)
        smoke_detected = data_package.get('smoke_detected', False)
        
        # 그리드 크기 업데이트 (detector에서 처리된 크기)
        detector_grid_size = data_package.get('interpolated_grid_size', self.interpolated_grid_size)
        if detector_grid_size != self.interpolated_grid_size:
            self.interpolated_grid_size = detector_grid_size
            self._create_heatmap_cells()

        self.time_label.setText(f"시간: {current_time}")

        if values:
            # detector에서 계산된 통계 사용
            max_temp = detection_stats.get('max_temp', np.max(values))
            avg_temp = detection_stats.get('avg_temp', np.mean(values))
            filter_temp = processing_params.get('filter_temp', 0)
            
            self.sensor_temp_label.setText(f"센서 온도: {sensor_degree:.1f}°C")
            self.max_temp_label.setText(f"최고 온도: {max_temp:.1f}°C")
            self.avg_temp_label.setText(f"평균 온도: {avg_temp:.1f}°C")
            self.etc_label.setText(f"기타: {etc}")
            self.object_detection_label.setText(f"객체: {object_detection}")
            self.test_label.setText(f"필터 온도: {filter_temp:.1f}°C")
        else:
            self.max_temp_label.setText(f"최고 온도: N/A")
            self.avg_temp_label.setText("평균 온도: N/A")

        # 상태 표시기 업데이트
        self.update_indicator_status(self.fire_indicator, fire_detected)
        self.update_indicator_status(self.smoke_indicator, smoke_detected)

        # 경고 처리
        is_anomaly = fire_detected or smoke_detected
        if is_anomaly:
            if fire_detected and not self.fire_alert_triggered:
                self.handle_anomaly("화재", current_time)
                self.fire_alert_triggered = True
            if smoke_detected and not self.smoke_alert_triggered:
                self.handle_anomaly("연기", current_time)
                self.smoke_alert_triggered = True

        if not fire_detected: self.fire_alert_triggered = False
        if not smoke_detected: self.smoke_alert_triggered = False

        # 히트맵 업데이트 (detector에서 처리된 값 사용)
        if processed_values:
            self.update_heatmap(processed_values)

    def handle_anomaly(self, anomaly_type, current_time):
        QApplication.processEvents()
        self.show_alert_popup(f"{anomaly_type}가 감지되었습니다!", f"시간: {current_time}\n시스템 로그를 확인하세요.")

    def update_indicator_status(self, indicator_widget, detected):
        new_state = "detected" if detected else "stable"
        if indicator_widget.property("state") != new_state:
            indicator_widget.setProperty("state", new_state)
            indicator_widget.style().unpolish(indicator_widget)
            indicator_widget.style().polish(indicator_widget)

    def update_heatmap(self, processed_values):
        """
        detector에서 처리된 값을 받아서 히트맵 업데이트
        더 이상 GUI에서 보간이나 필터링을 하지 않음
        """
        if not processed_values:
            return
            
        # processed_values는 이미 적절한 크기로 처리됨
        expected_size = self.interpolated_grid_size * self.interpolated_grid_size
        if len(processed_values) != expected_size:
            print(f"Warning: Expected {expected_size} values, got {len(processed_values)}")
            return

        processed_array = np.array(processed_values).reshape(self.interpolated_grid_size, self.interpolated_grid_size)
        
        # 히트맵 셀 업데이트
        if self.grid_cells is None or self.interpolated_grid_size != len(self.grid_cells):
            self._create_heatmap_cells()
        
        for i in range(self.interpolated_grid_size):
            for j in range(self.interpolated_grid_size):
                value = processed_array[i, j]
                color = self.get_color_from_value(value)
                cell = self.grid_cells[i][j]
                
                # 셀 텍스트 업데이트
                cell.setText(f"{value:.1f}" if self.display_degree else "")
                
                # 셀 스타일 업데이트
                cell.setStyleSheet(f"background-color: {color.name()}; color:black; font-weight: bold; font-size: {self.font_size}px; border: none;")

    def show_alert_popup(self, title, message):
        return  # 팝업 비활성화
        alert = QMessageBox(self)
        alert.setIcon(QMessageBox.Icon.Warning)
        alert.setWindowTitle("! 경고 !")
        alert.setText(title)
        alert.setInformativeText(message)
        alert.setStandardButtons(QMessageBox.StandardButton.Ok)
        alert.exec()

    def log_anomaly(self, message):
        pass

    def open_log_file(self):
        if not os.path.exists(self.log_filename):
            self.show_alert_popup("파일 없음", f"로그 파일({self.log_filename})이 아직 생성되지 않았습니다.")
            return
        if sys.platform == "win32":
            os.startfile(self.log_filename)
        elif sys.platform == "darwin":
            subprocess.run(["open", self.log_filename])
        else:
            subprocess.run(["xdg-open", self.log_filename])

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

# 실제 센서 데이터를 생성하고 큐에 넣는 역할을 하는 스레드
class DataGeneratorThread(QThread):
    def __init__(self, data_queue, data_signal_obj):
        super().__init__()
        self.data_queue = data_queue
        self.signal = data_signal_obj
        self.running = True
        self.time_counter = 0

    def run(self):
        while self.running:
            self.time_counter += 1
            # 8x8 원본 데이터 생성 (평균 25도, 가끔 35도 이상 값 발생)
            base_temps = np.random.normal(25.0, 2.0, OutputModule.ORIGINAL_GRID_SIZE**2)
            if random.random() < 0.2:  # 20% 확률로 이상 고온 발생
                anomaly_index = random.randint(0, 63)
                base_temps[anomaly_index] = random.uniform(35.0, 45.0)

            data_package = {
                'time': datetime.now().strftime("%H:%M:%S"),
                'values': base_temps.tolist(),
                'sensor_degree': np.mean(base_temps) + random.uniform(-0.5, 0.5),
                'etc': [round(random.uniform(20,80),1), round(random.uniform(20,80),1)],
            }
            # DetectionModule로 데이터 전송 (처리를 위해)
            self.data_queue.put(data_package)
            
            time.sleep(1)  # 1초 대기

    def stop(self):
        self.running = False
        self.wait()

# GUI 큐에서 데이터를 받아서 GUI에 전달하는 스레드
class GUIUpdateThread(QThread):
    def __init__(self, gui_queue, data_signal):
        super().__init__()
        self.gui_queue = gui_queue
        self.data_signal = data_signal
        self.running = True
        
    def run(self):
        while self.running:
            if not self.gui_queue.empty():
                data_package = self.gui_queue.get()
                self.data_signal.update_data_signal.emit(data_package)
            else:
                time.sleep(0.1)
                
    def stop(self):
        self.running = False
        self.wait()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    
    primary_screen = app.primaryScreen()
    available_rect = primary_screen.availableGeometry()

    # 1. 데이터 통신을 위한 큐와 시그널 객체 생성
    detection_queue = queue.Queue()
    gui_queue = queue.Queue()
    data_signal = DataSignal()
    
    # 2. GUI 인스턴스 생성
    output_gui = OutputModule(data_signal, available_rect)
    output_gui.showMaximized()

    # 3. DetectionModule 임포트 및 인스턴스 생성
    try:
        from detector import DetectionModule
        detection_module = DetectionModule(detection_queue, gui_queue, threshold=10.0)
        
        # 4. 파라미터 업데이트 시그널 연결
        data_signal.parameter_update_signal.connect(detection_module.update_detection_parameters)
        
        # 5. GUI 업데이트 스레드 생성
        gui_update_thread = GUIUpdateThread(gui_queue, data_signal)
        
        # 6. 데이터 생성 스레드 인스턴스 생성
        data_generator = DataGeneratorThread(detection_queue, data_signal)
        
        # 7. 프로그램 종료 시 스레드가 안전하게 종료되도록 설정
        app.aboutToQuit.connect(data_generator.stop)
        app.aboutToQuit.connect(detection_module.stop)
        app.aboutToQuit.connect(gui_update_thread.stop)

        # 8. 스레드 시작
        data_generator.start()
        detection_module.start()
        gui_update_thread.start()
        
        print("모든 모듈이 성공적으로 시작되었습니다.")
        
    except ImportError as e:
        print(f"DetectionModule을 찾을 수 없습니다: {e}")
        print("detector.py 파일이 같은 디렉토리에 있는지 확인하세요.")
        
        # DetectionModule 없이 GUI만 실행 (테스트용)
        class DummyDataGenerator:
            def __init__(self, signal_obj):
                self.signal = signal_obj
                self.timer = QTimer()
                self.timer.timeout.connect(self.generate_data)
                self.time_counter = 0

            def start(self):
                self.timer.start(1000)

            def generate_data(self):
                self.time_counter += 1
                random_values = np.random.uniform(15.0, 35.0, OutputModule.ORIGINAL_GRID_SIZE**2)
                fire_detected = random.random() < 0.05
                smoke_detected = random.random() < 0.02

                data_package = {
                    'time': f"00:00:{self.time_counter:02d}",
                    'values': random_values.tolist(),
                    'processed_values': random_values.tolist(),  # 단순히 동일한 값 사용
                    'fire_detected': fire_detected,
                    'smoke_detected': smoke_detected,
                    'detection_stats': {
                        'max_temp': np.max(random_values),
                        'avg_temp': np.mean(random_values)
                    },
                    'processing_params': {
                        'filter_temp': np.mean(random_values) + 5
                    },
                    'object_detection': ['test_object'],
                    'sensor_degree': np.mean(random_values),
                    'etc': [50.0, 60.0]
                }
                self.signal.update_data_signal.emit(data_package)

        dummy_gen = DummyDataGenerator(data_signal)
        dummy_gen.start()
        print("테스트 모드로 실행 중 (DetectionModule 없음)")

    sys.exit(app.exec())