
""" 
250804 세이브

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
from scipy.interpolate import RectBivariateSpline
import queue
import time

# === [수정됨] 1920x1080 해상도에 맞춰 폰트 크기 상향 조정 ===
GLOBAL_STYLESHEET = """
/* 전체 위젯 기본 스타일 */
QWidget {
    color: #E0E0E0; /* 기본 글자색 (밝은 회색) */
    font-family: 'Malgun Gothic', 'Segoe UI', 'Arial'; /* 폰트 우선순위 */
}

/* 메인 윈도우 스타일 */
#MainWindow {
    background-color: #2E2E2E; /* 메인 배경색 (어두운 회색) */
}

/* 정보 패널(카드) 스타일 */
#TopPanel, #LeftPanel {
    background-color: #3C3C3C; /* 패널 배경색 */
    border-radius: 8px; /* 둥근 모서리 */
    padding: 10px;
}

/* 제목 레이블 스타일 */
QLabel[class="TitleLabel"] {
    font-size: 12px; /* 12px -> 20px */
    font-weight: bold;
    color: #FFFFFF; /* 흰색 */
    padding: 2px;
}

/* 일반 정보 레이블 스타일 */
QLabel[class="InfoLabel"] {
    font-size: 12px; /* 14px -> 12px */
    font-weight: bold;
}

/* 이상 감지 카운트 레이블 특별 스타일 */
#AnomalyCountLabel {
    font-size: 16px; /* 12px -> 20px */
    font-weight: bold;
    color: #F44336; /* 밝은 빨강 */
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
    font-size: 12px; /* 12px -> 14px */
    font-weight: bold;
}
QPushButton:hover {
    background-color: #6A6A6A; /* 마우스 올렸을 때 */
}
QPushButton:pressed {
    background-color: #4A4A4A; /* 눌렀을 때 */
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

/* 상태에 따른 색상 (동적 속성 셀렉터 사용) */
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

# scipy.ndimage에서 gaussian_filter 임포트
from scipy.ndimage import gaussian_filter

class DataSignal(QObject):
    update_data_signal = pyqtSignal(dict)
    # [추가] DetectionModule에서 GUI로 이상 감지 횟수를 전달하기 위한 시그널
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
    ORIGINAL_GRID_SIZE = 8 # 원본 데이터 그리드 크기 (고정)
    DEFAULT_INTERPOLATED_GRID_SIZE = 8 # 초기 보간된 그리드 크기

    def __init__(self, data_signal_obj, available_rect):
        super().__init__()
        self.max_width = available_rect.width()
        self.max_height = available_rect.height() - 120

        # 클래스 변수 대신 인스턴스 변수로 관리
        self.original_grid_size = self.ORIGINAL_GRID_SIZE
        self.interpolated_grid_size = self.DEFAULT_INTERPOLATED_GRID_SIZE
        # cell_size는 interpolated_grid_size에 따라 동적으로 계산되도록
        self.cell_size = self.max_height // self.interpolated_grid_size
        self.font_size = self.cell_size//3

        self.anomaly_count = 0
        self.log_filename = "detected_values.txt"
        self.fire_alert_triggered = False
        self.smoke_alert_triggered = False
        self.data_signal = data_signal_obj
        self.grid_cells = []
        self.heatmap_layout = None # 히트맵 레이아웃을 저장할 변수 추가
        self.current_data_package = None # 마지막으로 받은 데이터 패키지를 저장
        self.data_signal.update_data_signal.connect(self.update_display)
        self.data_signal.anomaly_detected_signal.connect(self.update_anomaly_count)

        self.gaussian_sigma = 0.5 # 가우시안 필터의 시그마(표준편차) 값. 조절 가능

        
        # === 히트맵 온도 범위 인스턴스 변수 ===
        
        self.min_temp = 19.0
        self.max_temp = 32.0
        
        self.avg_temp = 0
        
        self.filter_temp_add = 5
        
        self.filter_temp = 0
        self.weight_list = [3.8, 4, 4.3, 5]

        # == 온도 출력 on off == 
        self.display_degree = True
        
        self.init_ui()
    # [추가] DetectionModule로부터 받은 카운트로 라벨을 업데이트하는 슬롯
    def update_anomaly_count(self, count):
        self.anomaly_count = count
        self.anomaly_count_label.setText(f"이상 감지: {self.anomaly_count} 회")
        # 필요하다면 여기서 경고 팝업을 띄울 수도 있습니다.
        # self.show_alert_popup(...)

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
        # display_degree.toggle()
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
        # 최고 온도 조절 UI
        weight_layout1, self.weight1_spinbox = self._create_temp_control_row("1", self.weight_list[0])
        weight_layout2, self.weight2_spinbox = self._create_temp_control_row("2", self.weight_list[1])
        weight_layout3, self.weight3_spinbox = self._create_temp_control_row("3", self.weight_list[2])
        weight_layout4, self.weight4_spinbox = self._create_temp_control_row("4", self.weight_list[3])
        

        # Signal 연결
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
        # 원본 그리드 크기의 배수로 설정
        self.grid_size_spinbox.setRange(self.ORIGINAL_GRID_SIZE, self.ORIGINAL_GRID_SIZE * 8) # 예시: 8x8 ~ 64x64
        self.grid_size_spinbox.setSingleStep(self.ORIGINAL_GRID_SIZE) # 8단위로 변경
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
        # grid_size_layout.addWidget(self.grid_size_slider)
        # --- 그리드 크기 조절 위젯 끝 ---

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
        left_panel_layout.addWidget(display_degree) # 온도 조절 위젯 추가
        left_panel_layout.addWidget(temp_range_group_box) # 온도 조절 위젯 추가

        left_panel_layout.addWidget(grid_size_group_box) # QWidget 자체를 추가
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
        spin_box.setRange(-50.0, 100.0) # 범위 설정
        spin_box.setSingleStep(0.1)
        spin_box.setDecimals(1) # 소수점 한 자리
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
    
    # === [추가됨] +/- 버튼으로 온도 조절하는 슬롯 ===
    def _adjust_temp(self, spin_box, amount):
        current_value = spin_box.value()
        spin_box.setValue(current_value + amount)

    # === [추가됨] 스핀박스 값 변경 시 실제 변수 업데이트 및 히트맵 갱신 ===
    def _update_temp_range(self, temp_type, value):
        if temp_type == 'min':
            self.min_temp = value
        elif temp_type == 'max':
            self.max_temp = value

        # 현재 데이터가 있으면 히트맵을 다시 그림
        self._redraw_heatmap()
    
    # === [추가됨] 스핀박스 값 변경 시 실제 변수 업데이트 및 히트맵 갱신 ===
    def _update_filter_weight(self, name, value):
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

        # 현재 데이터가 있으면 히트맵을 다시 그림
        self._redraw_heatmap()
        
    # === [추가됨] 현재 데이터로 히트맵을 다시 그리는 함수 ===
    def _redraw_heatmap(self):
        if self.current_data_package:
            self.update_heatmap(self.current_data_package.get('values', []))
            self.showMaximized() # 현재 윈도우를 최대화합니다.

    def _create_heatmap_cells(self):
        # 기존 셀들을 모두 제거
        if self.grid_cells:
            for row in self.grid_cells:
                for cell in row:
                    # removeWidget으로 레이아웃에서 위젯 제거
                    self.heatmap_layout.removeWidget(cell)
                    # deleteLater()로 위젯 객체를 메모리에서 해제
                    cell.deleteLater()
                    
        # grid_cells 리스트 초기화
        self.grid_cells = []
        
        # 현재 화면 높이에 맞춰 셀 크기 재계산
        # interpolated_grid_size를 사용하여 전체 크기를 나눕니다.
        self.cell_size = self.max_height // self.interpolated_grid_size
        # 폰트 크기 계산 시 정수 나눗셈을 사용합니다.
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
                    f"border: 1px solid #333;" # 경계를 추가하여 셀을 구분
                )
                
                # 레이아웃에 위젯을 추가합니다.
                self.heatmap_layout.addWidget(cell, i, j)
                row_cells.append(cell)
            self.grid_cells.append(row_cells)
        
        # 그리드 레이아웃의 마지막 행에 확장 공간을 추가하여 위젯들을 위로 정렬
        self.heatmap_layout.setRowStretch(self.interpolated_grid_size, 1)
        
        
    def _update_grid_size_from_spinbox(self, value):
        # 스핀박스 값이 변경되면 슬라이더 값도 변경
        self.grid_size_slider.setValue(value)
        # 실제 그리드 크기 업데이트 로직 호출
        self._set_grid_size(value)
        
        
    def _update_grid_size_from_slider(self, value):
        # 슬라이더 값이 변경되면 스핀박스 값도 변경
        self.grid_size_spinbox.setValue(value)
        # 실제 그리드 크기 업데이트 로직 호출
        self._set_grid_size(value)

    def _set_grid_size(self, new_size):
        # 현재 그리드 크기와 다를 경우에만 업데이트 및 다시 그리기
        if self.interpolated_grid_size != new_size:
            self.interpolated_grid_size = new_size
            self._create_heatmap_cells() # 히트맵 셀 다시 생성

            # 그리드 크기 변경 후, 현재 가지고 있는 데이터로 히트맵을 다시 업데이트
            if self.current_data_package:
                self.update_heatmap(self.current_data_package.get('values', []))
                

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
        # 현재 데이터 패키지를 저장 (그리드 크기 변경 시 다시 사용하기 위해)
        self.current_data_package = data_package

        sensor_degree = data_package.get('sensor_degree', 'N/A')
        current_time = data_package.get('time', 'N/A')
        values = data_package.get('values', [])
        etc = data_package.get('etc', [])
        object_detection = data_package.get('object_detection', [])
        
        fire_detected = data_package.get('fire_detected', False)
        smoke_detected = data_package.get('smoke_detected', False)

        self.time_label.setText(f"시간: {current_time}")

        if values:
            # print(f'avg 값 {self.avg_temp}')
            self.avg_temp = np.mean(values)
            # print(f'avg 갱신 {self.avg_temp}')
            self.filter_temp = self.avg_temp + self.filter_temp_add
            
            max_temp = np.max(values)
            
            self.sensor_temp_label.setText(f"센서 온도: {sensor_degree:.1f}°C")
            self.max_temp_label.setText(f"최고 온도: {max_temp:.1f}°C")
            self.avg_temp_label.setText(f"평균 온도: {self.avg_temp:.1f}°C")
            self.etc_label.setText(f"기타: {etc[0], etc[1]}")
            self.object_detection_label.setText(f"객체: {object_detection}")
            self.test_label.setText(f"test: {self.avg_temp:.1f} + {self.filter_temp_add:.1f}")
        else:
            self.max_temp_label.setText(f"최고 온도: N/A")
            self.avg_temp_label.setText("평균 온도: N/A")

        self.update_indicator_status(self.fire_indicator, fire_detected)
        self.update_indicator_status(self.smoke_indicator, smoke_detected)

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

        if values:
            # values = np.array(values, np.float32).reshape(8,8)
            self.update_heatmap(values) # 매번 최신 데이터를 기반으로 히트맵 업데이트

    def handle_anomaly(self, anomaly_type, current_time):
        # self.anomaly_count += 1
        # self.anomaly_count_label.setText(f"이상 감지: {self.anomaly_count} 회")
        QApplication.processEvents()
        self.show_alert_popup(f"{anomaly_type}가 감지되었습니다!", f"시간: {current_time}\n시스템 로그를 확인하세요.")

    def update_indicator_status(self, indicator_widget, detected):
        new_state = "detected" if detected else "stable"
        if indicator_widget.property("state") != new_state:
            indicator_widget.setProperty("state", new_state)
            indicator_widget.style().unpolish(indicator_widget)
            indicator_widget.style().polish(indicator_widget)

    def update_heatmap(self, values):
        """
        8x8 원본 데이터를 보간 및 필터링하여 히트맵을 업데이트합니다.
        높은 값은 유지하고 낮은 값만 부드럽게 만듭니다.
        """
        values = np.array(values, np.float32).reshape(8,8)
        if values is None or values.shape != (self.original_grid_size, self.original_grid_size):
            print("Invalid input values shape.")
            return

        threshold_temp = self.max_temp * 0.8 # 최고 온도의 90%를 기준으로 설정
        

        for i in range(8):  # 행 (row)
            for j in range(8):  # 열 (column)
                # 구역 0: 중앙 2x2 구역
                if (i >= 3 and i <= 4) and (j >= 3 and j <= 4) and values[i][j] >= self.filter_temp:
                    values[i][j] = values[i][j] * self.weight_list[0]
                    break
                # 구역 1: 중앙 4x4에서 1번 구역을 제외한 구역
                elif (i >= 2 and i <= 5) and (j >= 2 and j <= 5) and values[i][j] >= self.filter_temp:
                    values[i][j] = values[i][j] * self.weight_list[1]
                    break
                # 구역 2: 중앙 6x6에서 2번, 1번 구역을 제외한 구역
                elif (i >= 1 and i <= 6) and (j >= 1 and j <= 6) and values[i][j] >= self.filter_temp:
                    values[i][j] = values[i][j] * self.weight_list[2]
                    break
                # 구역 3: 가장 바깥쪽 테두리 (위의 모든 구역을 제외한 나머지)
                elif values[i][j] >= self.filter_temp:
                    values[i][j] = values[i][j] * self.weight_list[3]
                # 해당 사항 없을 시.    
                else:
                    pass
                    
        
               
        now_grid_size = self.interpolated_grid_size//self.original_grid_size
        interpolated_values = values.repeat(now_grid_size, axis=0).repeat(now_grid_size, axis=1)
        
        # 2. 높은 값은 유지하고 낮은 값만 가우시안 필터링 (Gaussian Filtering)
        
        #    - 임계값 설정: 이 온도보다 높은 값은 필터링하지 않습니다.
        # threshold_temp = self.max_temp * 0.9 # 최고 온도의 90%를 기준으로 설정
       
        # #    - 필터링 결과가 들어갈 배열을 원본 데이터로 초기화
        # filtered_values = interpolated_values.copy()

        # #    - 필터링할 영역(낮은 온도)의 마스크 생성
        # high_temp_mask = interpolated_values > threshold_temp
        # low_temp_mask = interpolated_values < threshold_temp

        # #    - 마스크를 적용하여 낮은 온도 값만 추출
        # high_temp_data = filtered_values[high_temp_mask]
        # low_temp_data = filtered_values[low_temp_mask]
        
        # # 높은 온도에는 가중치 곱함(임시)
        # multiple_high_temp= high_temp_data * 5
        
        # #    - 낮은 온도 데이터에만 가우시안 필터 적용
        # if low_temp_data.size > 0:
        #     # 필터링 강도는 self.gaussian_sigma 변수를 사용합니다.
        #     # blurred_low_temp = gaussian_filter(low_temp_data, sigma=self.gaussian_sigma)
        #     blurred_low_temp = low_temp_data
            
        #     #    - 필터링 결과를 원본 배열의 해당 위치에 다시 삽입
        #     filtered_values[low_temp_mask] = blurred_low_temp
        #     filtered_values[high_temp_mask] = multiple_high_temp

        
        # 3. 히트맵 셀(cell) 업데이트
        #    - filtered_values 배열을 사용하여 셀의 색상과 텍스트를 업데이트합니다.
        #    - grid_cells 변수를 사용하여 각 셀 위젯에 접근합니다.
        if self.grid_cells is None or self.interpolated_grid_size != len(self.grid_cells):
            self._create_heatmap_cells() # 그리드 크기가 변경되었을 경우 셀을 재생성
        
        for i in range(self.interpolated_grid_size):
            for j in range(self.interpolated_grid_size):
                value = interpolated_values[i, j]
                color = self.get_color_from_value(value)
                cell = self.grid_cells[i][j]
                
                # 셀 텍스트 업데이트 (온도 표시 여부에 따라)
                cell.setText(f"{value:.1f}" if self.display_degree else "")
                
                # 셀 스타일 업데이트
                cell.setStyleSheet(f"background-color: {color.name()}; color:black; font-weight: bold; font-size: {self.font_size}px; border: none;")

    

    def show_alert_popup(self, title, message):
        return
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

# [추가] 실제 센서 데이터를 생성하고 큐에 넣는 역할을 하는 스레드
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
            if random.random() < 0.2: # 20% 확률로 이상 고온 발생
                anomaly_index = random.randint(0, 63)
                base_temps[anomaly_index] = random.uniform(35.0, 45.0)

            data_package = {
                'time': datetime.now().strftime("%H:%M:%S"),
                'values': base_temps.tolist(),
                'sensor_degree': np.mean(base_temps) + random.uniform(-0.5, 0.5),
                'etc': (round(random.uniform(20,80),1), round(random.uniform(20,80),1)),
                'fire_detected': random.random() < 0.05,
                'smoke_detected': random.random() < 0.02,
            }
            # DetectionModule로 데이터 전송
            self.data_queue.put(data_package)
            # GUI로 데이터 전송
            self.signal.update_data_signal.emit(data_package)
            
            time.sleep(1) # 1초 대기

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
    data_signal = DataSignal()
    
    # 2. GUI 인스턴스 생성
    output_gui = OutputModule(data_signal, available_rect)
    output_gui.showMaximized()

    # 3. 데이터 생성 스레드와 데이터 감지 스레드 인스턴스 생성
    data_generator = DataGeneratorThread(detection_queue, data_signal)
    # DetectionModule에 data_signal 객체 전달
    # detection_module = DetectionModule(detection_queue, data_signal, threshold=10.0)
    
    # 4. 프로그램 종료 시 스레드가 안전하게 종료되도록 설정
    app.aboutToQuit.connect(data_generator.stop)
    # app.aboutToQuit.connect(detection_module.stop)

    # 5. 스레드 시작
    data_generator.start()
    # detection_module.start()

    sys.exit(app.exec())

    class DataGenerator:
        def __init__(self, signal_obj):
            self.signal = signal_obj
            self.timer = QTimer()
            self.timer.timeout.connect(self.generate_data)
            self.time_counter = 0

        def start(self):
            self.timer.start(1000)

        def generate_data(self):
            self.time_counter += 1
            # ORIGINAL_GRID_SIZE는 고정된 8x8 데이터를 생성
            random_values = np.random.uniform(15.0, 35.0, OutputModule.ORIGINAL_GRID_SIZE**2)
            fire_detected = random.random() < 0.05
            smoke_detected = random.random() < 0.02

            data_package = {
                'time': f"00:00:{self.time_counter:02d}",
                'values': random_values.tolist(),
                'fire_detected': fire_detected,
                'smoke_detected': smoke_detected,
            }
            self.signal.update_data_signal.emit(data_package)

    data_gen = DataGenerator(data_signal)
    data_gen.start()
