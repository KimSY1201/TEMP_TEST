"""  250804 리팩토링 완료:
- 열원 필터링, 가중치 적용, 보간 처리를 detector에서 수행
- GUI는 처리된 데이터를 받아서 표시만 담당
- 파라미터 변경 시 detector에 업데이트 신호 전송
- 센서 위치 변경 시 가중치 UI 동적 업데이트 추가

"""

""" 250807 평균온도 필터링 추가
"""

""" 2025/01/XX 리팩토링:
- GUI 우선 로딩 및 포트 선택 기능 추가
- 포트 연결/해제 상태 표시 추가
- 동적 포트 전환 기능 구현
"""

import sys
import os
import subprocess
import numpy as np
import random
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QGridLayout, QMessageBox, QPushButton, QRadioButton,
                             QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QButtonGroup,
                             QComboBox, QGroupBox, QFrame, QTextBrowser, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint, QThread, QVariantAnimation, QAbstractAnimation
from PyQt6.QtGui import QColor, QFont, QScreen, QMouseEvent
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl
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
#TopPanel {
    background-color: #080F25;
    border-radius: 8px;
    padding: 10px;
}

#LeftPanel {
    border-radius: 8px;
    padding: 10px;
}

/* 연결 상태 패널 스타일 */
#ConnectionPanel {
    background-color: #4A4A4A;
    border-radius: 6px;
    padding: 8px;
    border: 2px solid #666666;
}

#ConnectionPanel[connected="true"] {
    border-color: #4CAF50;
    background-color: #2E4A2E;
}

#ConnectionPanel[connected="false"] {
    border-color: #F44336;
    background-color: #4A2E2E;
}

.Panel {
    background-color: #353535;
    border-color: #F44336;
    border-radius: 6px;
    padding: 8px;
}

/* 제목 레이블 스타일 */
QLabel[class="TitleLabel"] {
    font-size: 16pt;
    font-weight: bold;
    color: #FFFFFF;
    padding: 2px;
}

/* 일반 정보 레이블 스타일 */
QLabel[class="InfoLabel"] {
    font-size: 14pt;
    font-weight: bold;
}

/* 연결 상태 레이블 스타일 */
QLabel[class="ConnectionLabel"] {
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

/* 연결 버튼 특별 스타일 */
QPushButton[buttonType="connect"] {
    background-color: #4CAF50;
}
QPushButton[buttonType="connect"]:hover {
    background-color: #5CBF60;
}

QPushButton[buttonType="disconnect"] {
    background-color: #F44336;
}
QPushButton[buttonType="disconnect"]:hover {
    background-color: #FF5346;
}

/* 온도 조절 버튼 작은 스타일 */
QPushButton[class="AdjustButton"] {
    font-size: 12px;
    font-weight: bold;
    padding: 4px 8px;
    min-width: 30px;
}

/* QComboBox 스타일 */
QComboBox {
    background-color: #2E2E2E;
    color: #FFFFFF;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 5px;
    font-size: 12px;
    font-weight: bold;
    min-width: 80px;
}
QComboBox::drop-down {
    border: none;
}
QComboBox::down-arrow {
    border: none;
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

.temp_widget { 
                    background-color: #37446B;
                    border: 1px solid #1D2439;
                    border-radius: 5px;
                    padding: 5px 10px;
                    }
                
                
#safety {
    border: none;
}
#caution {
    border: none;
}
#danger {
    border: none;
}
#safety_alert {
    border: 3px solid green;
}
#caution_alert {
    border: 3px solid yellow;
}
#danger_alert {
    border: 3px solid red;
}


"""


class DataSignal(QObject):
    update_data_signal = pyqtSignal(dict)
    # DetectionModule로 파라미터 업데이트를 전송하는 시그널
    parameter_update_signal = pyqtSignal(dict)
    anomaly_detected_signal = pyqtSignal(int)
    # 포트 변경 시그널 추가
    port_change_signal = pyqtSignal(dict)
    alarm_signal = pyqtSignal(str)
    
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

        self.current_file_path = os.path.abspath(__file__)
        self.current_dir = os.path.dirname(self.current_file_path)
        
        
        # 클래스 변수 대신 인스턴스 변수로 관리
        self.original_grid_size = self.ORIGINAL_GRID_SIZE
        self.interpolated_grid_size = self.DEFAULT_INTERPOLATED_GRID_SIZE
        self.cell_size = self.max_height // self.interpolated_grid_size
        self.font_size = self.cell_size // 3
        
        # 포트 및 연결 상태 관리
        self.current_port = ""
        self.current_baudrate = 57600
        self.is_connected = False
        
        # 센서 위치에 따른 토글 변수
        self.sensor_position = 'corner'
        
        # 이상 감지 관련
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
        self.weight_list_corner = [5, 5, 5, 5, 5, 5, 5, 5]

        # == 온도 출력 on off == 
        self.display_degree = True
        self.display_degree_over_avg = False
        
        # 가중치 UI 관련 변수들
        self.weight_grid_layout = None
        self.weight_spinboxes = []
        self.weight_layouts = []
        
        # 소리재생 플레이어 객체 유무
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        
        # 시각화 객체 
        self.animation = QVariantAnimation()
        
        
        self.init_ui()

    def update_anomaly_count(self, count):
        self.anomaly_count = count

    def init_ui(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle('temp_app_test 통합 관제 시스템')

        main_layout = QGridLayout(self)
        main_layout.setSpacing(15)

        self.left_panel = QWidget()
        self.left_panel.setObjectName("LeftPanel")
        self.left_panel_layout = QVBoxLayout(self.left_panel)
        self.left_panel_layout.setSpacing(5)

        # === 연결 설정 패널 ===
        connection_panel = self._create_connection_panel()
        self.left_panel_layout.addWidget(connection_panel)
        
        ## 상단 온도 통합 레이아웃
        temp_grid_widget = QWidget()        
        temp_grid_widget.setProperty('class','Panel')
        temp_grid_layout = QGridLayout(temp_grid_widget)
        
        self.sensor_temp_widget, self.sensor_temp_label = self._create_temp_grid("센서", "")
        self.max_temp_widget, self.max_temp_label = self._create_temp_grid("최고", "")
        self.avg_temp_widget, self.avg_temp_label = self._create_temp_grid("평균", "")
        
        temp_grid_layout.addWidget(self.sensor_temp_widget, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter )
        temp_grid_layout.addWidget(self.max_temp_widget, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter )
        temp_grid_layout.addWidget(self.avg_temp_widget, 0, 2, alignment=Qt.AlignmentFlag.AlignCenter )
        
        self.etc_label = QLabel("기타: N/A")
        self.etc_label.setProperty("class", "TitleLabel")
        
        #--------------------------------------------------

        fs_detect_widget = QWidget()
        fs_detect_widget.setProperty('class','Panel')
        fs_layout = QHBoxLayout(fs_detect_widget)
        
        fire_layout, self.fire_indicator = self._create_status_row("화재 감지")
        smoke_layout, self.smoke_indicator = self._create_status_row("연기 감지")
        
        fs_layout.addLayout(fire_layout)
        fs_layout.addLayout(smoke_layout)
        
        #--------------------------------------------------
        
        self.object_detection_label = QLabel("객체: N/A")
        self.object_detection_label.setProperty("class", "TitleLabel")
        
        #--------------------------------------------------
        
        self.suspect_fire_label = QLabel("의심 열원: N/A")
        self.suspect_fire_label.setProperty("class", "Panel")
        
        
        heat_source_widget = QWidget()
        heat_source_widget.setProperty("class", "Panel")
        heat_source_grid = QGridLayout(heat_source_widget)
        heat_source_grid.setSpacing(0)
        heat_source_grid.setContentsMargins(0, 0, 0, 0)
        # 모든 열에 균등한 확장 비율을 할당
        heat_source_grid.setColumnStretch(0, 1)
        heat_source_grid.setColumnStretch(1, 1)
        heat_source_grid.setColumnStretch(2, 1)


        self.safety_widget, self.safety_label = self._create_scd_widget("safety", '')
        self.caution_widget, self.caution_label = self._create_scd_widget("caution", '')
        self.danger_widget, self.danger_label = self._create_scd_widget("danger", '')

        heat_source_grid.addWidget(self.safety_widget, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        heat_source_grid.addWidget(self.caution_widget, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        heat_source_grid.addWidget(self.danger_widget, 0, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.humidity_label = QLabel("습도: 55.0%")
        self.humidity_label.setProperty("class", "InfoLabel")
        
        #### ---------------------로그 출력--------------------- ###
        
        self.log_widget = QWidget()
        self.log_widget.setProperty("class","Panel")
        self.log_layout = QGridLayout(self.log_widget)
        log_label = QLabel("최신 이상 감지")
        log_label.setProperty("class", "InfoLabel")
        
        log_clear_button = QPushButton("clear")

        self.log_text_panel = QTextBrowser()
        self.log_text_panel.setFixedHeight(30)
        self.log_text_panel.setStyleSheet(""" background-color:black """)
        
        log_clear_button.clicked.connect(self.log_text_panel.clear)
        
        log_open_button = QPushButton("자세히 보기")
        log_open_button.clicked.connect(self.open_log_file)

        self.log_layout.addWidget(log_label, 0, 0)
        self.log_layout.addWidget(log_clear_button, 0, 1)
        self.log_layout.addWidget(log_open_button, 0, 2)
        self.log_layout.addWidget(self.log_text_panel, 1, 0, 1, 3)
        
                
        ##### -------------------- 조작부 ---------------------- ###
        
        
        ## 설정부 통합 레이아웃
        self.config_layout = QVBoxLayout()
        
        
        
        
        # -- 온도 표시 위젯 --
        filter_grid_widget = QWidget()
        filter_grid_widget.setProperty("class","Panel")
        fg_layout = QVBoxLayout(filter_grid_widget)
        
        
        display_filter_layout = QHBoxLayout()
        
        display_label = QLabel("온도 필터: ")
        display_label.setProperty("class", "InfoLabel")
        display_degree = QCheckBox("온도 표시", self)
        display_degree.stateChanged.connect(self.display_degree_control)
        display_degree.setChecked(True)  # 기본값 설정
        display_degree_over_avg = QCheckBox("평균이상만", self)
        display_degree_over_avg.stateChanged.connect(self.display_degree_over_avg_control)
        display_degree_over_avg.setChecked(False)  # 기본값 설정
        
        display_filter_layout.addWidget(display_label)
        display_filter_layout.addWidget(display_degree)
        display_filter_layout.addWidget(display_degree_over_avg)
        
        # --- 그리드 크기 조절 위젯 ---
        grid_size_group_box = QWidget()
        # grid_size_group_box.setObjectName("TopPanel")
        grid_size_layout = QVBoxLayout(grid_size_group_box)
        grid_size_up_layout = QHBoxLayout()

        self.grid_label_prefix = QLabel("그리드 해상도:")
        self.grid_label_prefix.setProperty("class", "InfoLabel")
        # self.grid_label_prefix.setProperty("class", "TitleLabel")

        self.grid_size_spinbox = QSpinBox()
        self.grid_size_spinbox.setRange(self.ORIGINAL_GRID_SIZE, self.ORIGINAL_GRID_SIZE * 8)
        self.grid_size_spinbox.setSingleStep(self.ORIGINAL_GRID_SIZE)
        self.grid_size_spinbox.setValue(self.interpolated_grid_size)
        self.grid_size_spinbox.valueChanged.connect(self._update_grid_size_from_spinbox)
        self.grid_size_spinbox.setStyleSheet(f"background-color:black")
        self.grid_size_spinbox.setFont(QFont("Arial", 12))

        self.grid_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.grid_size_slider.setRange(self.ORIGINAL_GRID_SIZE, self.ORIGINAL_GRID_SIZE * 8)
        self.grid_size_slider.setSingleStep(self.ORIGINAL_GRID_SIZE)
        self.grid_size_slider.setValue(self.interpolated_grid_size)
        self.grid_size_slider.valueChanged.connect(self._update_grid_size_from_slider)
        
        grid_size_up_layout.addWidget(self.grid_label_prefix)
        grid_size_up_layout.addStretch(1)
        grid_size_up_layout.addWidget(self.grid_size_spinbox)

        grid_size_layout.addLayout(grid_size_up_layout)
        
        fg_layout.addLayout(display_filter_layout)
        fg_layout.addWidget(grid_size_group_box)
        
                        
        # --- 히트맵 온도 범위 조절 위젯 ---
        temp_range_widget = QWidget()
        temp_range_widget.setProperty("class","Panel")
        temp_range_layout = QGridLayout(temp_range_widget)
        
        temp_range_title = QLabel("히트맵 온도 범위")
        temp_range_title.setProperty("class", "InfoLabel")
        
        self.filter_label = QLabel("필터 온도: N/A")
        self.filter_label.setProperty("class", "InfoLabel")
            
        # 최저 온도 조절 UI
        min_temp_layout, self.min_temp_spinbox = self._create_temp_control_row("최저", self.min_temp)
        # 최고 온도 조절 UI
        max_temp_layout, self.max_temp_spinbox = self._create_temp_control_row("최고", self.max_temp)
        
        # 필터링 적용 온도 조절 UI
        filter_temp_layout, self.filter_temp_spinbox = self._create_temp_control_row("기준", self.filter_temp_add)
        
        # Signal 연결 - detector에 파라미터 업데이트 전송
        self.min_temp_spinbox.valueChanged.connect(lambda value: self._update_temp_range('min', value))
        self.max_temp_spinbox.valueChanged.connect(lambda value: self._update_temp_range('max', value))
        self.filter_temp_spinbox.valueChanged.connect(lambda value: self._update_filter_weight('filter_add', value))
        
        temp_range_layout.addWidget(temp_range_title, 0, 0, 1, 1)
        temp_range_layout.addWidget(self.filter_label, 1, 0, 2, 1)
        temp_range_layout.addLayout(min_temp_layout, 0, 1)
        temp_range_layout.addLayout(max_temp_layout, 0, 2)
        temp_range_layout.addLayout(filter_temp_layout, 1, 2)
        
        
        # === 센서 위치 설정 ===
        posi_weight_widget = QWidget()
        posi_weight_widget.setProperty("class","Panel")
        pw_layout = QVBoxLayout(posi_weight_widget)
        
        position_layout = QHBoxLayout()
        
        # 라디오 버튼 그룹 생성 (상호 배타적 선택 보장)
        self.posi_label = QLabel("센서 위치: ", self)
        self.posi_label.setProperty("class", "InfoLabel")
        self.position_button_group = QButtonGroup(self)
        self.posi_center_sensor = QRadioButton("중앙", self)
        self.posi_corner_sensor = QRadioButton("모서리", self)

        self.position_button_group.addButton(self.posi_center_sensor, 0)
        self.position_button_group.addButton(self.posi_corner_sensor, 1)
        
        # 기본 선택 (모서리)
        self.posi_corner_sensor.setChecked(True)
        
        # 시그널 연결
        self.posi_center_sensor.toggled.connect(self.position_toggled)
        self.posi_corner_sensor.toggled.connect(self.position_toggled)
        
        position_layout.addWidget(self.posi_label)
        position_layout.addWidget(self.posi_center_sensor)
        position_layout.addWidget(self.posi_corner_sensor)
        
        
        # 가중치 조절 UI 레이아웃 생성
        self.weight_grid_layout = QGridLayout()
        self._create_weight_controls()


        pw_layout.addLayout(position_layout)
        pw_layout.addLayout(self.weight_grid_layout)


        # config layout에 위젯들 추가                
        self.config_layout.addWidget(filter_grid_widget)
        self.config_layout.addWidget(temp_range_widget)
        self.config_layout.addWidget(posi_weight_widget)
        
        
        
        self.time_label = QLabel("시간: N/A")
        self.time_label.setFont(QFont("Arial", 9))

        # self.left_panel_layout에 위젯들 추가
        
        self.left_panel_layout.addWidget(temp_grid_widget)
        # self.left_panel_layout.addWidget(self.etc_label)
        # self.left_panel_layout.addLayout(fire_layout)
        # self.left_panel_layout.addLayout(smoke_layout)
        
        self.left_panel_layout.addWidget(fs_detect_widget)
        
        self.left_panel_layout.addWidget(self.suspect_fire_label)
        self.left_panel_layout.addWidget(heat_source_widget)
        self.left_panel_layout.addWidget(self.log_widget)
        self.left_panel_layout.addLayout(self.config_layout)

        self.left_panel_layout.addStretch(1)
        self.left_panel_layout.addWidget(self.time_label)

        main_layout.addWidget(self.left_panel, 0, 0, 2, 1)

        self.heatmap_layout = QGridLayout()
        self.heatmap_layout.setSpacing(0)
        self.heatmap_layout.setContentsMargins(0,0,0,0)
        main_layout.addLayout(self.heatmap_layout, 0, 1, 2, 1)

        # 초기 히트맵 생성
        self._create_heatmap_cells()

        main_layout.setColumnStretch(1, 1)
        main_layout.setRowStretch(0, 1)
    
        
    def _create_temp_grid(self, label_text, temp_text):
        temp_widget = QWidget()
        tempVlayout = QVBoxLayout(temp_widget)
        label_label = QLabel(label_text)
        label_label.setProperty('class','InfoLabel')
        temp_label = QLabel(temp_text)
        temp_label.setProperty('class','InfoLabel')
        tempVlayout.addWidget(label_label)      
        tempVlayout.addWidget(temp_label)
        
        temp_widget.setProperty("class", "temp_widget")
        # temp_widget.setStyleSheet(test1)
        temp_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        return temp_widget, temp_label
    
        
    def _create_scd_widget(self, label_label, label_text):
        
        scd_widget = QWidget()
        scdVlayout = QVBoxLayout(scd_widget)
        scd_label = QLabel(label_label)
        label_text = QLabel(label_text)
        scdVlayout.addWidget(scd_label)      
        scdVlayout.addWidget(label_text)
        
        scd_label.setProperty("class",'InfoLabel')
        label_text.setProperty("class",'InfoLabel')
        scd_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scd_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # scd_widget.setProperty("class", "temp_widget")
        # scd_widget.setStyleSheet(test2)
        
        
        return scd_widget, label_text
        
    def _create_connection_panel(self):
        """연결 설정 패널 생성"""
        connection_panel = QGroupBox("연결 설정")
        connection_panel.setObjectName("ConnectionPanel")
        connection_panel.setProperty("connected", "false")
        
        layout = QVBoxLayout(connection_panel)
        
        # 포트 및 보드레이트 설정 레이아웃
        settings_layout = QHBoxLayout()
        
        # 포트 선택
        port_label = QLabel("포트:")
        port_label.setProperty("class", "InfoLabel")
        
        self.port_combo = QComboBox()
        self.port_combo.addItems(["COM3", "COM4", "COM5", "COM6", "COM7", "COM8"])
        self.port_combo.setCurrentText("COM3")
        self.port_combo.setStyleSheet(""" background-color:black """)
        
        
        # 보드레이트 선택
        baudrate_label = QLabel("보드레이트:")
        baudrate_label.setProperty("class", "InfoLabel")
        
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["38400", "57600", "115200"])
        self.baudrate_combo.setCurrentText("57600")
        self.baudrate_combo.setStyleSheet(""" background-color:black """)
        
        settings_layout.addWidget(port_label)
        settings_layout.addWidget(self.port_combo)
        settings_layout.addWidget(baudrate_label)
        settings_layout.addWidget(self.baudrate_combo)
        settings_layout.addStretch()
        
        # 연결 상태 및 버튼 레이아웃
        connection_layout = QHBoxLayout()
        
        self.connection_status_label = QLabel("연결 안됨")
        self.connection_status_label.setProperty("class", "ConnectionLabel")
        
        self.connect_button = QPushButton("연결")
        self.connect_button.setProperty("buttonType", "connect")
        self.connect_button.clicked.connect(self.on_connect_clicked)
        
        self.disconnect_button = QPushButton("연결 해제")
        self.disconnect_button.setProperty("buttonType", "disconnect")
        self.disconnect_button.clicked.connect(self.on_disconnect_clicked)
        self.disconnect_button.setEnabled(False)
        
        connection_layout.addWidget(self.connection_status_label)
        connection_layout.addStretch()
        connection_layout.addWidget(self.connect_button)
        connection_layout.addWidget(self.disconnect_button)
        
        layout.addLayout(settings_layout)
        layout.addLayout(connection_layout)
        
        return connection_panel
    
    def on_connect_clicked(self):
        """연결 버튼 클릭 시 호출"""
        port = self.port_combo.currentText()
        baudrate = int(self.baudrate_combo.currentText())
        
        print(f"연결 요청: {port}, {baudrate}")
        
        # 포트 변경 시그널 발송
        port_info = {
            'port': port,
            'baudrate': baudrate
        }
        self.data_signal.port_change_signal.emit(port_info)
        
        # UI 상태 업데이트 (임시)
        self.connect_button.setEnabled(False)
        self.connection_status_label.setText(f"연결 중... ({port})")
    
    def on_disconnect_clicked(self):
        """연결 해제 버튼 클릭 시 호출"""
        print("연결 해제 요청")
        
        # 빈 포트 정보로 연결 해제 시그널 발송
        port_info = {
            'port': '',
            'baudrate': 0
        }
        self.data_signal.port_change_signal.emit(port_info)
    
    def update_connection_status(self, connected, port, error_msg=""):
        """연결 상태 업데이트 (외부에서 호출)"""
        self.is_connected = connected
        self.current_port = port if connected else ""
        if self.current_port == '':
            connected = False
        
        # UI 상태 업데이트
        connection_panel = self.findChild(QGroupBox, "ConnectionPanel")
        if connection_panel:
            connection_panel.setProperty("connected", "true" if connected else "false")
            connection_panel.style().unpolish(connection_panel)
            connection_panel.style().polish(connection_panel)
        
        if connected:
            self.connection_status_label.setText(f"연결됨: {port}")
            self.connect_button.setEnabled(False)
            self.disconnect_button.setEnabled(True)
            self.port_combo.setEnabled(False)
            self.baudrate_combo.setEnabled(False)
        else:
            if error_msg:
                self.connection_status_label.setText(f"연결 실패: {error_msg}")
            else:
                self.connection_status_label.setText("연결 안됨")
            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            self.port_combo.setEnabled(True)
            self.baudrate_combo.setEnabled(True)
        
        
    def _create_weight_controls(self):
        """센서 위치에 따른 가중치 조절 UI 생성"""
        # 기존 가중치 UI 제거
        self._clear_weight_controls()
        
        # 현재 센서 위치에 따른 가중치 리스트 및 개수 결정
        if self.sensor_position == 'center':
            weights_list = self.weight_list
            num_weights = 4
        elif self.sensor_position == 'corner':
            weights_list = self.weight_list_corner
            num_weights = 8
        
        # 새로운 가중치 UI 생성
        self.weight_label = QLabel("가중치")
        self.weight_label.setProperty("class", "InfoLabel")
        self.weight_grid_layout.addWidget(self.weight_label, 0, 0, 1, num_weights)
        self.weight_layouts = []
        self.weight_spinboxes = []
        
        for i in range(num_weights):
            weight_label = str(i + 1)
            current_weight = weights_list[i]
            
            # _create_temp_control_row 함수 호출
            layout, spinbox = self._create_temp_control_row(weight_label, current_weight)
            
            # 시그널 연결: 람다 함수에 i를 전달하여 어떤 가중치인지 구분
            spinbox.valueChanged.connect(lambda value, index=i: self._update_filter_weight(f'w{index + 1}', value))
            
            self.weight_layouts.append(layout)
            self.weight_spinboxes.append(spinbox)
            
            # 그리드 레이아웃에 추가
            row = i // 2
            col = i % 2
            self.weight_grid_layout.addLayout(layout, row, col+1)
    
    def _clear_weight_controls(self):
        """기존 가중치 UI 요소들 제거"""
        # 기존 레이아웃에서 모든 아이템 제거
        while self.weight_grid_layout.count():
            child = self.weight_grid_layout.takeAt(0)
            if child.layout():
                self._clear_layout(child.layout())
            elif child.widget():
                child.widget().deleteLater()
        
        # 리스트 초기화
        self.weight_layouts.clear()
        self.weight_spinboxes.clear()
    
    def _clear_layout(self, layout):
        """레이아웃 내의 모든 위젯 제거"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_layout(child.layout())
    
    def position_toggled(self):
        """센서 위치 라디오 버튼 토글 시 호출"""
        if self.posi_center_sensor.isChecked():
            self.sensor_position = 'center'
            print('Position changed to: center')
        elif self.posi_corner_sensor.isChecked():
            self.sensor_position = 'corner'
            print('Position changed to: corner')
        
        # 가중치 UI 다시 생성
        self._create_weight_controls()
        
        # detector에 센서 위치 업데이트 전송
        params = {
            'sensor_position': self.sensor_position,
        } 
        self.data_signal.parameter_update_signal.emit(params)
    
    def display_degree_control(self, state):
        """온도 표시 체크박스 제어"""
        self.display_degree = bool(state)
        
        # 현재 데이터로 히트맵 다시 업데이트 (온도 표시 여부 적용)
        if self.current_data_package:
            processed_values = self.current_data_package.get('processed_values', [])
            if processed_values:
                self.update_heatmap(processed_values)
        
    def display_degree_over_avg_control(self, state):
        """평균이상만 표시 체크박스 제어"""
        self.display_degree_over_avg = bool(state)
        print('display_degree_over_avg_control', state)
        
        # 현재 데이터로 히트맵 다시 업데이트 (온도 표시 여부 적용)
        if self.current_data_package:
            processed_values = self.current_data_package.get('processed_values', [])
            if processed_values:
                self.update_heatmap(processed_values)
        
    # === 온도 조절 UI 생성 헬퍼 함수 ===
    def _create_temp_control_row(self, label_text, initial_value):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setProperty("class", "InfoLabel")

        spin_box = QDoubleSpinBox()
        spin_box.setRange(-50.0, 100.0)
        spin_box.setSingleStep(0.1)
        spin_box.setDecimals(1)
        spin_box.setValue(initial_value)

        layout.addStretch(1)
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(spin_box)

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
        
        if self.sensor_position == 'center':
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
            
        elif self.sensor_position == 'corner':
            if name == 'w1':
                self.weight_list_corner[0] = value
                print(f'w1 {value} 갱신')
            elif name == 'w2':
                self.weight_list_corner[1] = value
                print(f'w2 {value} 갱신')
            elif name == 'w3':
                self.weight_list_corner[2] = value
                print(f'w3 {value} 갱신')
            elif name == 'w4':
                self.weight_list_corner[3] = value
                print(f'w4 {value} 갱신')
            elif name == 'w5':
                self.weight_list_corner[4] = value
                print(f'w5 {value} 갱신')
            elif name == 'w6':
                self.weight_list_corner[5] = value
                print(f'w6 {value} 갱신')
            elif name == 'w7':
                self.weight_list_corner[6] = value
                print(f'w7 {value} 갱신')
            elif name == 'w8':
                self.weight_list_corner[7] = value
                print(f'w8 {value} 갱신')

            # detector에 파라미터 업데이트 전송
            params = {
                'filter_temp_add': self.filter_temp_add,
                'weight_list_corner': self.weight_list_corner.copy()
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
        indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        layout.addWidget(indicator)
        layout.addStretch(1)
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
        
        # heat source dict
        heat_source_dict = data_package.get('heat_source_dict', {})        
        suspect_fire = data_package.get('suspect_fire', [])        
        
        total_log = data_package.get('total_log', 'N/A')        
        
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
            
            self.sensor_temp_label.setText(f"{sensor_degree:.1f}°C")
            self.max_temp_label.setText(f"{max_temp:.1f}°C")
            self.avg_temp_label.setText(f"{avg_temp:.1f}°C")
            self.avg_temp = avg_temp
            self.etc_label.setText(f"기타: {etc}")
            self.object_detection_label.setText(f"객체: {object_detection}")
            self.suspect_fire_label.setText(f"의심 열원: {suspect_fire}")
            self.filter_label.setText(f"필터 온도: {filter_temp:.1f}°C")
            
            self.log_text_panel.setText(total_log)
            self.log_text_panel.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            # heat source 정보 업데이트
            if heat_source_dict:
                if heat_source_dict.get('safety', 'N/A') is not None:
                    safety_text = len(heat_source_dict.get('safety', 'N/A'))
                else:
                    safety_text = 'None'
                
                self.safety_label.setText(str(safety_text))
                self.caution_label.setText(str(len(heat_source_dict.get('caution', 'N/A'))))
                self.danger_label.setText(str(len(heat_source_dict.get('danger', 'N/A'))))
                
                widgets = [self.safety_widget, self.caution_widget, self.danger_widget]
                keys = ['safety', 'caution', 'danger']

                for widget, key in zip(widgets, keys):
                    if len(heat_source_dict.get(key, 'N/A')) == 0 :
                        widget.setObjectName(key)
                        # print(f"{widget}.setObjectName('ID', '')")
                    else:
                        widget.setObjectName(key+'_alert')
                        
                        # print(f"{widget}.setObjectName('{key}')")
                
                if len(heat_source_dict.get('danger', 'N/A')):
                    self.alarm_call('danger')
                    # self.show_alert_popup(f"{'danger'}가 감지되었습니다!", f"시간: {current_time}\n시스템 로그를 확인하세요.")
                    self.alert_visual('danger')
                elif len(heat_source_dict.get('danger', 'N/A')) == 0 and len(heat_source_dict.get('caution', 'N/A')):
                    self.alarm_call('caution')
                    self.alert_visual('')
                else:
                    self.alert_visual('')
                    pass
                           
                widget.style().unpolish(widget)
                widget.style().polish(widget)  
                
                
                
            
        else:
            self.max_temp_label.setText(f"N/A")
            self.avg_temp_label.setText("N/A")

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
        # self.show_alert_popup(f"{anomaly_type}가 감지되었습니다!", f"시간: {current_time}\n시스템 로그를 확인하세요.")

    def update_indicator_status(self, indicator_widget, detected):
        new_state = "detected" if detected else "stable"
        if indicator_widget.property("state") != new_state:
            indicator_widget.setProperty("state", new_state)
            indicator_widget.style().unpolish(indicator_widget)
            indicator_widget.style().polish(indicator_widget)

    
    def alert_visual(self, str):
        if str == 'danger':
            # 애니메이션이 이미 존재하고 실행 중이면, 아무것도 하지 않고 종료
            if self.animation and self.animation.state() == QAbstractAnimation.State.Running:
                return
            
            # 애니메이션이 존재하지 않거나 멈춰있으면, 새로운 애니메이션을 시작
            else:
                # 기존 애니메이션이 있다면 연결 해제 및 제거
                if self.animation:
                    self.animation.valueChanged.disconnect(self.update_background_color)
                    self.animation = None

                # 새로운 애니메이션 객체 생성
                self.left_panel.setObjectName("leftPanel_alert")
                self.animation = QVariantAnimation(self.left_panel)
                self.animation.setDuration(1000)
                self.animation.setStartValue(QColor(0, 0, 0, 0)) # 투명
                self.animation.setEndValue(QColor(255, 0, 0, 255)) # 빨간색
                
                # 애니메이션 값이 변할 때마다 호출될 함수 연결
                self.animation.valueChanged.connect(self.update_background_color)
                
                # 무한 반복하도록 설정
                self.animation.setLoopCount(-1) 
                
                self.animation.start()
                
        # 'danger'가 아닐 경우, 애니메이션 정지 및 리셋
        else:
            self.left_panel.setObjectName("leftPanel")
            # 먼저 애니메이션 객체가 존재하는지 확인
            if self.animation:
                # 애니메이션이 실행 중이면 정지
                if self.animation.state() != self.animation.State.Stopped:
                    self.animation.stop()

                # 시그널 연결 해제 및 객체 리셋
                # disconnect를 호출하기 전에 연결 여부를 확인하는 명시적인 방법은 없으므로,
                # 애니메이션 객체를 리셋할 때 항상 disconnect를 호출하되,
                # 이전에 연결이 없었을 가능성을 염두에 두어야 합니다.
                try:
                    self.animation.valueChanged.disconnect(self.update_background_color)
                except TypeError:
                    pass # 연결되지 않은 경우 무시
                    
                self.animation = None

    # QWidget 클래스 내부에 추가할 함수
    def update_background_color(self, color):
        color_name = color.name(QColor.NameFormat.HexArgb)
        style_sheet = f"#leftPanel_alert {{ background-color: {color_name}; }}"
        self.left_panel.setStyleSheet(style_sheet)
        
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
                
                if self.display_degree_over_avg:
                    # value = value - self.avg_temp 
                    value = 0 if value - (self.avg_temp+2)  < 0 else value                        
                    # print(value)    
                # 셀 텍스트 업데이트 (온도 표시 여부에 따라)
                cell.setText(f"{value:.1f}" if self.display_degree else "")
                
                
                # 셀 스타일 업데이트
                cell.setStyleSheet(f"background-color: {color.name()}; color:black; font-weight: bold; font-size: {self.font_size}px; border: none;")

    def show_alert_popup(self, title, message):
        # return  # 팝업 비활성화
        alert = QMessageBox(self)
        alert.setIcon(QMessageBox.Icon.Warning)
        alert.setWindowTitle("! 경고 !")
        alert.setText(title)
        alert.setInformativeText(message)
        alert.setStandardButtons(QMessageBox.StandardButton.Ok)
        alert.show()

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
    
    
    def alarm_call(self, str):
        print('get', str)
        # 1. QMediaPlayer 및 QAudioOutput 객체 생성
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            return
        else:
            self.player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.player.setAudioOutput(self.audio_output)
            
            # 2. 재생할 파일 설정 (파일 경로는 실제 파일로 변경해야 합니다)
            file_path = f"{self.current_dir}/_asset/sound/{str}.mp3"  
            print(file_path)
            self.player.setSource(QUrl.fromLocalFile(file_path))

            # 3. 재생
            self.audio_output.setVolume(30)  # 볼륨 설정 (0~100)
            self.player.play()

# # 실제 센서 데이터를 생성하고 큐에 넣는 역할을 하는 스레드 (테스트용)
# class DataGeneratorThread(QThread):
#     def __init__(self, data_queue, data_signal_obj):
#         super().__init__()
#         self.data_queue = data_queue
#         self.signal = data_signal_obj
#         self.running = True
#         self.time_counter = 0

#     def run(self):
#         while self.running:
#             self.time_counter += 1
#             # 8x8 원본 데이터 생성 (평균 25도, 가끔 35도 이상 값 발생)
#             base_temps = np.random.normal(25.0, 2.0, OutputModule.ORIGINAL_GRID_SIZE**2)
#             if random.random() < 0.2:  # 20% 확률로 이상 고온 발생
#                 anomaly_index = random.randint(0, 63)
#                 base_temps[anomaly_index] = random.uniform(35.0, 45.0)

#             data_package = {
#                 'time': datetime.now().strftime("%H:%M:%S"),
#                 'values': base_temps.tolist(),
#                 'sensor_degree': np.mean(base_temps) + random.uniform(-0.5, 0.5),
#                 'etc': [round(random.uniform(20,80),1), round(random.uniform(20,80),1)],
#             }
#             # DetectionModule로 데이터 전송 (처리를 위해)
#             self.data_queue.put(data_package)
            
#             time.sleep(1)  # 1초 대기

#     def stop(self):
#         self.running = False
#         self.wait()

# # GUI 큐에서 데이터를 받아서 GUI에 전달하는 스레드
# class GUIUpdateThread(QThread):
#     def __init__(self, gui_queue, data_signal):
#         super().__init__()
#         self.gui_queue = gui_queue
#         self.data_signal = data_signal
#         self.running = True
        
#     def run(self):
#         while self.running:
#             if not self.gui_queue.empty():
#                 data_package = self.gui_queue.get()
#                 self.data_signal.update_data_signal.emit(data_package)
#             else:
#                 time.sleep(0.1)
                
#     def stop(self):
#         self.running = False
#         self.wait()