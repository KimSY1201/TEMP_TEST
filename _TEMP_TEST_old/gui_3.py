import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QPalette

# PyQt와 통신하기 위한 시그널 객체
class DataSignal(QObject):
    update_heatmap_signal = pyqtSignal(dict)

class OutputModule(QWidget):
    ORIGINAL_GRID_SIZE = 8
    INTERPOLATED_GRID_SIZE = 15  # (ORIGINAL_GRID_SIZE * 2) - 1

    def __init__(self, data_signal_obj):
        super().__init__()
        self.data_signal = data_signal_obj
        self.data_signal.update_heatmap_signal.connect(self.update_display)
        self.grid_cells = []
        self.init_ui()

    def init_ui(self):
        # 1. GUI 크기를 약 1.5배 확대 (550x600 -> 755x785)
        self.setWindowTitle('Interpolated Heatmap Display (Viridis)')
        self.setGeometry(100, 100, 755, 785)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.time_label = QLabel("Current Time: N/A")
        
        self.time_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px;")
        main_layout.addWidget(self.time_label)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        for i in range(self.INTERPOLATED_GRID_SIZE):
            row_cells = []
            for j in range(self.INTERPOLATED_GRID_SIZE):
                cell_label = QLabel()
                
                # 1. 셀 크기를 약 1.5배 확대 (35x35 -> 50x50)
                cell_label.setFixedSize(50, 50)
                
                # 3. 텍스트 중앙 정렬 및 기본 스타일 설정
                cell_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell_label.setStyleSheet("background-color: lightgray; font-size: 10px;")

                grid_layout.addWidget(cell_label, i, j)
                row_cells.append(cell_label)
            self.grid_cells.append(row_cells)

        main_layout.addLayout(grid_layout)
        self.setLayout(main_layout)

    # ------------------------------------------------------------------
    # 새로운 색상 함수들
    # ------------------------------------------------------------------

    def get_color_viridis(self, value):
        """Viridis 색상 맵 (보라->파랑->초록->노랑)"""
        p = max(0, min(100, value)) / 100.0
        if p < 0.25:
            f = p / 0.25
            r, g, b = 68 * (1 - f) + 49 * f, 1 * (1 - f) + 104 * f, 84 * (1 - f) + 142 * f
        elif p < 0.5:
            f = (p - 0.25) / 0.25
            r, g, b = 49 * (1 - f) + 33 * f, 104 * (1 - f) + 145 * f, 142 * (1 - f) + 140 * f
        elif p < 0.75:
            f = (p - 0.5) / 0.25
            r, g, b = 33 * (1 - f) + 94 * f, 145 * (1 - f) + 181 * f, 140 * (1 - f) + 51 * f
        else:
            f = (p - 0.75) / 0.25
            r, g, b = 94 * (1 - f) + 253 * f, 181 * (1 - f) + 231 * f, 51 * (1 - f) + 37 * f
        return QColor(int(r), int(g), int(b))

    def get_color_plasma(self, value):
        """Plasma 색상 맵 (보라->분홍->주황->노랑)"""
        p = max(0, min(100, value)) / 100.0
        if p < 0.25: f = p / 0.25; r,g,b = 13*(1-f)+84*f, 8*(1-f)+40*f, 135*(1-f)+142*f
        elif p < 0.5: f = (p-0.25)/0.25; r,g,b = 84*(1-f)+158*f, 40*(1-f)+52*f, 142*(1-f)+106*f
        elif p < 0.75: f = (p-0.5)/0.25; r,g,b = 158*(1-f)+217*f, 52*(1-f)+83*f, 106*(1-f)+48*f
        else: f = (p-0.75)/0.25; r,g,b = 217*(1-f)+240*f, 83*(1-f)+159*f, 48*(1-f)+32*f
        return QColor(int(r), int(g), int(b))

    def get_color_jet(self, value):
        """Classic Jet 색상 맵 (파랑->청록->초록->노랑->빨강)"""
        p = max(0, min(100, value)) / 100.0
        if p < 0.125: r,g,b = 0, 0, 127 + int(128 * (p/0.125)) # 파랑 -> 연파랑
        elif p < 0.375: r,g,b = 0, int(255 * ((p - 0.125) / 0.25)), 255 # 연파랑 -> 청록
        elif p < 0.625: r,g,b = int(255 * ((p - 0.375) / 0.25)), 255, 255 - int(255 * ((p - 0.375) / 0.25)) # 청록 -> 노랑
        elif p < 0.875: r,g,b = 255, 255 - int(255 * ((p - 0.625) / 0.25)), 0 # 노랑 -> 빨강
        else: r,g,b = 255 - int(128 * ((p - 0.875) / 0.125)), 0, 0 # 빨강 -> 어두운 빨강
        return QColor(int(r), int(g), int(b))


    def update_display(self, data_package):
        current_time = data_package['time']
        values = data_package['values']

        self.time_label.setText(f"Current Time: {current_time}")

        if len(values) != self.ORIGINAL_GRID_SIZE ** 2:
            return

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
                
                # 활성화된 색상 함수 호출
                # color = self.get_color_viridis(value)
                # color = self.get_color_plasma(value) 
                color = self.get_color_jet(value)
                
                # 3. 가독성을 위해 배경색 밝기에 따라 텍스트 색상 동적 변경
                # (R*0.299 + G*0.587 + B*0.114) > 186 이면 밝은 색으로 간주하여 검은 텍스트 사용
                brightness = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
                text_color = "black" if brightness > 186 else "white"

                # 2. 모든 셀에 온도 값을 소수점 첫째 자리까지 표시
                self.grid_cells[i][j].setText(f"{value:.1f}")
                
                # 스타일시트 설정 (배경색, 폰트 스타일, 텍스트 색상)
                stylesheet = (
                    f"background-color: {color.name()};"
                    f"color: {text_color};"
                    "font-weight: bold;"
                    "font-size: 12px;"
                    "border: none;"
                )
                self.grid_cells[i][j].setStyleSheet(stylesheet)


# 아래는 테스트를 위한 가상 데이터 생성 및 실행 코드입니다.
# 실제 환경에서는 이 부분을 제거하고 OutputModule을 사용하시면 됩니다.
class DataGenerator:
    def __init__(self, signal_obj):
        self.signal = signal_obj
        self.timer = QTimer()
        self.timer.timeout.connect(self.generate_data)
        self.time_counter = 0

    def start(self):
        self.timer.start(1000)  # 1초마다 데이터 생성

    def generate_data(self):
        self.time_counter += 1
        # 8x8 격자에 맞는 64개의 랜덤 데이터 생성 (0-100)
        random_values = np.random.rand(OutputModule.ORIGINAL_GRID_SIZE**2) * 100
        data_package = {
            'time': f"00:00:{self.time_counter:02d}",
            'values': random_values.tolist()
        }
        self.signal.update_heatmap_signal.emit(data_package)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 데이터 통신을 위한 시그널 객체 생성
    data_signal = DataSignal()
    
    # OutputModule 인스턴스 생성
    output_gui = OutputModule(data_signal)
    output_gui.show()
    
    # 테스트를 위한 데이터 생성기 시작
    data_gen = DataGenerator(data_signal)
    data_gen.start()
    
    sys.exit(app.exec())