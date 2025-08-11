
import sys
import os
import subprocess
import numpy as np
import random
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QGridLayout, QMessageBox, QPushButton, QRadioButton,
                             QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QButtonGroup)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint, QThread
from PyQt6.QtGui import QColor, QFont, QScreen, QMouseEvent
import queue
import time



class module1(QWidget):
    def __init__(self, count, count2):
        self.count = count
        self.count2 = count2
        self.init_ui()
        
    def init_ui(self):
        super.__init__()
        self.setObjectName("MainWindow")
        self.setWindowTitle('temp_app_test 통합 관제 시스템')

        self.mainLayout = QGridLayout(self)
        some_text = QLabel()
        some_text.setText('test123')
        self.mainLayout.addWidget(some_text)


        