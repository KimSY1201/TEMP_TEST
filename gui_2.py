# 추가 수정 사항들
# 크가 더 키우기
# 셀마다 온도 값 다 넣기

 
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
        self.setWindowTitle('Interpolated Heatmap Display (Viridis)')
        self.setGeometry(100, 100, 550, 600)

        main_layout = QVBoxLayout()
        self.time_label = QLabel("Current Time: N/A")
        self.time_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(self.time_label)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        for i in range(self.INTERPOLATED_GRID_SIZE):
            row_cells = []
            for j in range(self.INTERPOLATED_GRID_SIZE):
                cell_label = QLabel()
                cell_label.setStyleSheet("background-color: lightgray;")
                cell_label.setFixedSize(35, 35)
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
        if p < 0.25: r,g,b = 0, int(255*(p/0.25)), 255
        elif p < 0.5: r,g,b = 0, 255, int(255*(1-(p-0.25)/0.25))
        elif p < 0.75: r,g,b = int(255*((p-0.5)/0.25)), 255, 0
        else: r,g,b = 255, int(255*(1-(p-0.75)/0.25)), 0
        return QColor(int(r), int(g), int(b))


    # 1. 중복된 update_display 메서드를 하나로 합침
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
                
                # 2. 올바른 색상 함수 이름으로 호출
                # color = self.get_color_viridis(value)
                # color = self.get_color_plasma(value) 
                color = self.get_color_jet(value)
                
                stylesheet = f"background-color: {color.name()}; border: none;"

                if i % 2 == 0 and j % 2 == 0:
                    self.grid_cells[i][j].setText(f"{int(value)}")
                    # 텍스트 색상을 흰색으로 하여 가독성 확보
                    stylesheet += "font-weight: bold; color: white;" 
                else:
                    self.grid_cells[i][j].setText("") 
                
                self.grid_cells[i][j].setStyleSheet(stylesheet)
