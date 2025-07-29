import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QPalette

# PyQt와 통신하기 위한 시그널 객체
class DataSignal(QObject):
    update_heatmap_signal = pyqtSignal(dict)

class OutputModule(QWidget):
    def __init__(self, data_signal_obj):
        super().__init__()
        self.data_signal = data_signal_obj
        # 시그널 연결 확인을 위한 print 추가
        print("OutputModule: Connecting update_heatmap_signal to update_display.")
        self.data_signal.update_heatmap_signal.connect(self.update_display)
        self.grid_cells = [] # 히트맵 셀들을 저장할 리스트
        self.init_ui()
        
        print("OutputModule: UI initialized.")

    def init_ui(self):
        self.setWindowTitle('Heatmap Display')
        self.setGeometry(100, 100, 400, 450) # 창 위치 및 크기

        main_layout = QVBoxLayout()
        
        # 현재 시각 표시 라벨
        self.time_label = QLabel("Current Time: N/A")
        # self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(self.time_label)

        # 8x8 히트맵 그리드
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10) # 셀 간 간격

        for i in range(8):
            row_cells = []
            for j in range(8):
                cell_label = QLabel("0") # 초기 값
                # cell_label.setAlignment(Qt.AlignCenter)
                cell_label.setStyleSheet("background-color: lightgray; border: 0px solid black;")
                cell_label.setFixedSize(45, 45) # 셀 크기
                grid_layout.addWidget(cell_label, i, j)
                row_cells.append(cell_label)
            self.grid_cells.append(row_cells)
        
        main_layout.addLayout(grid_layout)
        self.setLayout(main_layout)

    
    def update_display(self, data_package):
        # update_display 호출 시점 확인을 위한 print 추가
        print(f"OutputModule: update_display called for time {data_package['time']}")
        current_time = data_package['time']
        values = data_package['values']

        self.time_label.setText(f"Current Time: {current_time}")

        if len(values) == 64:
            for i in range(8):
                for j in range(8):
                    index = i * 8 + j
                    value = values[index]
                    self.grid_cells[i][j].setText(str(value))
                    color = self.get_color_from_value(value)
                    self.grid_cells[i][j].setStyleSheet(
                        f"background-color: {color.name()}; border: 0px solid black; font-weight: bold;"
                    )
            # GUI 업데이트 완료 print 추가
            print("OutputModule: Heatmap updated successfully.")
        else:
            print(f"OutputModule: Expected 64 values, got {len(values)}. Cannot update heatmap.")