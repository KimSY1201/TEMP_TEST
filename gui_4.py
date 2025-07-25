import sys
import os
import subprocess
import numpy as np
import random
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QGridLayout, QMessageBox, QPushButton)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QFont

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
    font-size: 20px; /* 16px -> 20px */
    font-weight: bold;
    color: #FFFFFF; /* 흰색 */
    padding: 5px;
}

/* 일반 정보 레이블 스타일 */
QLabel[class="InfoLabel"] {
    font-size: 16px; /* 14px -> 16px */
    font-weight: bold;
}

/* 이상 감지 카운트 레이블 특별 스타일 */
#AnomalyCountLabel {
    font-size: 20px; /* 16px -> 20px */
    font-weight: bold;
    color: #F44336; /* 밝은 빨강 */
    padding: 5px;
}

/* 히트맵 셀 스타일 */
QLabel[class="HeatmapCell"] {
    font-weight: bold;
    font-size: 14px; /* 12px -> 14px */
    border: none;
}

/* 버튼 스타일 */
QPushButton {
    background-color: #555555;
    color: #FFFFFF;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 14px; /* 12px -> 14px */
    font-weight: bold;
}
QPushButton:hover {
    background-color: #6A6A6A; /* 마우스 올렸을 때 */
}
QPushButton:pressed {
    background-color: #4A4A4A; /* 눌렀을 때 */
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

class DataSignal(QObject):
    update_data_signal = pyqtSignal(dict)

class OutputModule(QWidget):
    ORIGINAL_GRID_SIZE = 8
    INTERPOLATED_GRID_SIZE = 15

    def __init__(self, data_signal_obj, available_rect):
        super().__init__()
        self.max_width = available_rect.width()
        self.max_height = available_rect.height()
        
        self.anomaly_count = 0
        self.log_filename = "detected_values.txt" 
        self.fire_alert_triggered = False
        self.smoke_alert_triggered = False
        self.data_signal = data_signal_obj
        self.data_signal.update_data_signal.connect(self.update_display)
        self.grid_cells = []
        self.init_ui()

    def init_ui(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle('통합 관제 시스템')
        
        # === [수정됨] 고정된 창 크기 설정 제거 ===
        # self.setGeometry(100, 100, 950, 785)

        main_layout = QGridLayout(self)
        main_layout.setSpacing(15)

        
        left_panel = QWidget()
        left_panel.setObjectName("LeftPanel")
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setSpacing(20)
        
        self.avg_temp_label = QLabel("평균 온도: N/A")
        self.avg_temp_label.setProperty("class", "TitleLabel")
        self.anomaly_count_label = QLabel(f"이상 감지: {self.anomaly_count} 회")
        self.anomaly_count_label.setObjectName("AnomalyCountLabel")
        
        fire_layout, self.fire_indicator = self._create_status_row("화재 감지")
        smoke_layout, self.smoke_indicator = self._create_status_row("연기 감지")

        self.humidity_label = QLabel("습도: 55.0%") 
        self.humidity_label.setProperty("class", "InfoLabel")
        
        self.log_button = QPushButton("이상 감지 정보 자세히 보기")
        self.log_button.clicked.connect(self.open_log_file)
        
        left_panel_layout.addWidget(self.avg_temp_label)
        left_panel_layout.addWidget(self.anomaly_count_label)
        left_panel_layout.addLayout(fire_layout)
        left_panel_layout.addLayout(smoke_layout)
        left_panel_layout.addWidget(self.humidity_label)
        left_panel_layout.addStretch(1)
        left_panel_layout.addWidget(self.log_button)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(0) # 셀 사이의 간격 제거
        for i in range(self.INTERPOLATED_GRID_SIZE):
            row_cells = []
            for j in range(self.INTERPOLATED_GRID_SIZE):
                cell = QLabel()
                # === [수정됨] 히트맵 셀의 고정 크기 제거 ===
                # cell.setFixedSize(25, 25) 
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setProperty("class", "HeatmapCell")
                cell.setStyleSheet("background-color: lightgray;")
                grid_layout.addWidget(cell, i, j)
                row_cells.append(cell)
            self.grid_cells.append(row_cells)

        self.time_label = QLabel("시간: N/A")
        self.time_label.setFont(QFont("Arial", 9))
        
        # main_layout.addWidget(top_panel, 0, 0, 1, 2)
        main_layout.addWidget(left_panel, 1, 0)
        main_layout.addLayout(grid_layout, 1, 1)
        main_layout.addWidget(self.time_label, 2, 1, Qt.AlignmentFlag.AlignRight)
        
        # 히트맵 영역(열 1, 행 1)이 남는 공간을 모두 차지하도록 설정
        main_layout.setColumnStretch(1, 1)
        main_layout.setRowStretch(1, 1)
    
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
        current_time = data_package.get('time', 'N/A')
        values = data_package.get('values', [])
        fire_detected = data_package.get('fire_detected', False)
        smoke_detected = data_package.get('smoke_detected', False)
        
        self.time_label.setText(f"시간: {current_time}")
        
        if values:
            avg_temp = np.mean(values)
            self.avg_temp_label.setText(f"평균 온도: {avg_temp:.1f}°C")
        else:
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
            self.update_heatmap(values)
    
    def handle_anomaly(self, anomaly_type, current_time):
        self.anomaly_count += 1
        self.anomaly_count_label.setText(f"이상 감지: {self.anomaly_count} 회")
        QApplication.processEvents()
        self.show_alert_popup(f"{anomaly_type}가 감지되었습니다!", f"시간: {current_time}\n시스템 로그를 확인하세요.")

    def update_indicator_status(self, indicator_widget, detected):
        new_state = "detected" if detected else "stable"
        if indicator_widget.property("state") != new_state:
            indicator_widget.setProperty("state", new_state)
            indicator_widget.style().unpolish(indicator_widget)
            indicator_widget.style().polish(indicator_widget)
            
    def update_heatmap(self, values):
        if len(values) != self.ORIGINAL_GRID_SIZE ** 2: return
        original_data = np.array(values).reshape((self.ORIGINAL_GRID_SIZE, self.ORIGINAL_GRID_SIZE))
        interpolated_grid = np.zeros((self.INTERPOLATED_GRID_SIZE, self.INTERPOLATED_GRID_SIZE))
        interpolated_grid[::2, ::2] = original_data
        for i in range(self.INTERPOLATED_GRID_SIZE):
            if i % 2 == 0:
                for j in range(1, self.INTERPOLATED_GRID_SIZE, 2):
                    interpolated_grid[i, j] = (interpolated_grid[i, j-1] + interpolated_grid[i, j+1]) / 2
        for j in range(self.INTERPOLATED_GRID_SIZE):
            for i in range(1, self.INTERPOLATED_GRID_SIZE, 2):
                interpolated_grid[i, j] = (interpolated_grid[i-1, j] + interpolated_grid[i+1, j]) / 2
        for i in range(self.INTERPOLATED_GRID_SIZE):
            for j in range(self.INTERPOLATED_GRID_SIZE):
                value = interpolated_grid[i, j]
                color = self.get_color_from_value(value)
                brightness = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
                text_color = "black" if brightness > 186 else "white"
                cell = self.grid_cells[i][j]
                cell.setText(f"{value:.1f}")
                # === [수정됨] 스타일시트 적용 시 폰트 속성 유지 ===
                cell.setStyleSheet(f"background-color: {color.name()}; color: {text_color}; font-weight: bold; font-size: 14px; border: none;")

    def show_alert_popup(self, title, message):
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
        """
        C# 코드의 로직을 따라 온도 값에 따른 RGB 색상 매핑을 수행합니다.
        입력: float (온도 값)
        출력: QColor 객체
        """
        min_temp = 19.0 # C#의 Int16 min_temp = 19; 에 해당. float 연산을 위해 .0 추가.
        max_temp = 32.0 # C#의 Int16 max_temp = 32; 에 해당. float 연산을 위해 .0 추가

        # C# 코드에서 value == "MOD" 일 때 value = "100"로 바꿨는데,
        # 파이썬 함수에서는 float 타입의 value를 직접 받으므로 이 부분은 제거합니다.
        # 만약 'MOD' 문자열이 들어올 가능성이 있다면, 해당 문자열 처리 로직은 이 함수 외부에서 해야 합니다.
        
        # C#의 decimal.Parse(value)에 해당. 이 함수는 이미 float 값(temp)을 받는다고 가정합니다.
        temp = float(value) # 입력 value가 float이 아닐 경우를 대비

        # nomalized_temp 계산
        # nomalized_temp = (float)((temp - min_temp) / (max_temp - min_temp));
        # 분모가 0이 되는 경우를 방지 (min_temp == max_temp)
        if (max_temp - min_temp) == 0:
            nomalized_temp = 0.0 # 혹은 적절한 기본값이나 오류 처리
        else:
            nomalized_temp = (temp - min_temp) / (max_temp - min_temp)

        # RGB 값 초기화
        red = 0.0
        green = 0.0
        blue = 0.0

        # C#의 5단계 색상 매핑 로직
        if nomalized_temp <= 0.2:
            red = 0
            green = float(255 * (5 * nomalized_temp))
            blue = 255
        elif nomalized_temp <= 0.4:
            red = 0
            green = 255
            blue = float(255 * (1 - (5 * (nomalized_temp - 0.2))))
        elif nomalized_temp <= 0.6:
            red = float(255 * (5 * (nomalized_temp - 0.4)))
            green = 255
            blue = 0
        elif nomalized_temp <= 0.8:
            red = 255
            green = float(255 * (1 - (5 * (nomalized_temp - 0.6))))
            blue = 0
        else: # nomalized_temp > 0.8
            red = 255
            green = 0
            blue = 0

        # RGB 값 0-255 범위로 클램프하고 정수형으로 변환
        r = int(max(0, min(255, red)))
        g = int(max(0, min(255, green)))
        b = int(max(0, min(255, blue)))

        return QColor(r, g, b)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    
    data_signal = DataSignal()
    output_gui = OutputModule(data_signal)
    
    # === [수정됨] 창을 일반 'show()' 대신 'showMaximized()'로 표시 ===
    output_gui.showMaximized()
    
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
            random_values = np.random.rand(OutputModule.ORIGINAL_GRID_SIZE**2) * 100
            fire_detected = self.time_counter % 5 == 0 
            smoke_detected = False
            data_package = {
                'time': f"00:00:{self.time_counter:02d}",
                'values': random_values.tolist(),
                'fire_detected': fire_detected,
                'smoke_detected': smoke_detected,
            }
            self.signal.update_data_signal.emit(data_package)

    data_gen = DataGenerator(data_signal)
    data_gen.start()
    sys.exit(app.exec())