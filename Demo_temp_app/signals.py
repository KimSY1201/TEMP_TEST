from PyQt6.QtCore import QObject, pyqtSignal

class DataSignal(QObject):
    """main.py와 gui.py 간의 통신을 위한 PyQt 시그널 묶음"""
    update_data_signal      = pyqtSignal(dict)  # Detector → GUI 데이터
    parameter_update_signal = pyqtSignal(dict)  # (옵션) GUI → Detector 파라미터
    anomaly_detected_signal = pyqtSignal(int)   # (옵션) 이상감지 카운트
    port_change_signal      = pyqtSignal(dict)  # GUI에서 포트 변경 전송

    def __init__(self):
        super().__init__()