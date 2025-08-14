import sys
import os
from queue import Queue
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# 필요한 모듈들을 가져옵니다.
from detector import DetectionModule
from gui import OutputModule, GLOBAL_STYLESHEET
from Receiver import ReceiverWorker
from signals import DataSignal

# ==========================
# 자동 연결 설정 (TCP 전용)
# ==========================
AUTO_CONNECT = True
DEFAULT_PORT = 8080  # TCP 통신을 위한 기본 포트 번호
DEFAULT_BAUDRATE = 57600 # TCP에서는 사용되지 않음

# ==========================
# 포트 → sensor_id 매핑 규칙 (TCP 전용)
# ==========================
TCP_PORT_SENSOR_MAP = {
    8080: 1,
    8081: 2,
    8082: 3,
}
# 시리얼 관련 맵은 제거되었습니다.
# SERIAL_PORT_SENSOR_MAP = { ... }


def resolve_sensor_id(port, default_id=1):
    """포트 번호로 sensor_id 결정 (TCP 전용으로 단순화)."""
    # port가 정수(int)인 경우만 처리합니다.
    if isinstance(port, int):
        return TCP_PORT_SENSOR_MAP.get(port, default_id)
    # 시리얼 포트 관련 로직은 제거되었습니다.
    return default_id


class ApplicationManager:
    """애플리케이션 전체를 관리"""

    def __init__(self):
        self.output_queue = Queue()
        self.detection_queue = Queue()
        self.data_signal_obj = DataSignal()
        self.receiver = None
        self.detection_module = None
        self.output_module_gui = None
        self.current_sensor_id = 1
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.check_queues)

    def initialize_gui(self, available_rect):
        self.output_module_gui = OutputModule(self.data_signal_obj, available_rect)
        self.data_signal_obj.port_change_signal.connect(self.on_port_changed)
        self.output_module_gui.show()

    def _start_detection_module(self, sensor_id: int):
        if self.detection_module and self.detection_module.is_alive():
            try:
                self.detection_module.stop()
                self.detection_module.join(timeout=2)
            except Exception:
                pass
            self.detection_module = None

        print(f"[Detector] start with sensor_id={sensor_id}")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 데이터 저장 폴더가 없으면 생성
        data_dir = os.path.join(current_dir, "_data")
        os.makedirs(data_dir, exist_ok=True)
        
        self.detection_module = DetectionModule(
            self.data_signal_obj,
            self.detection_queue,
            self.output_queue,
            threshold=5.0,
            filename=os.path.join(data_dir, "detected_values.txt"),
            sensor_id=sensor_id,
        )
        self.detection_module.start()

    def initialize_detection_module(self):
        self._start_detection_module(self.current_sensor_id)
        if not self.timer.isActive():
            self.timer.start()

    def initialize_receiver(self, port, baudrate=57600):
        if self.receiver is not None:
            self.stop_receiver()
        try:
            self.receiver = ReceiverWorker(self.detection_queue, port=port, baudrate=baudrate)
            self.receiver.start()
            if self.output_module_gui:
                self.output_module_gui.update_connection_status(True, port)
            print(f"[Receiver] started on {port}")
            return True
        except Exception as e:
            if self.output_module_gui:
                self.output_module_gui.update_connection_status(False, port, str(e))
            print(f"[Receiver] start failed: {e}")
            return False

    def on_port_changed(self, port_info):
        """GUI로부터 포트 변경 신호를 받았을 때 호출되는 콜백 함수"""
        port = port_info.get('port', DEFAULT_PORT)
        baudrate = port_info.get('baudrate', DEFAULT_BAUDRATE)

        # 연결 해제 신호
        if port == 0:
            self.stop_receiver()
            print("[Port] closed")
            return

        new_sensor_id = resolve_sensor_id(port, default_id=self.current_sensor_id)
        sensor_changed = (new_sensor_id != self.current_sensor_id)

        ok = self.initialize_receiver(port, baudrate)

        if ok and sensor_changed:
            self.current_sensor_id = new_sensor_id
            self._start_detection_module(self.current_sensor_id)

        if ok:
            print(f"[Port] connected: {port} (sensor_id={self.current_sensor_id})")
        else:
            print(f"[Port] connect failed: {port}")

    def stop_receiver(self):
        if self.receiver and self.receiver.is_alive():
            try:
                self.receiver.stop()
                self.receiver.join(timeout=2)
            except Exception:
                pass
        self.receiver = None
        if self.output_module_gui:
            self.output_module_gui.update_connection_status(False, "")

    def check_queues(self):
        if not self.output_queue.empty():
            data_package = self.output_queue.get()
            self.data_signal_obj.update_data_signal.emit(data_package)

    def shutdown(self):
        print("Shutting down...")
        if self.timer.isActive():
            self.timer.stop()
        self.stop_receiver()
        if self.detection_module and self.detection_module.is_alive():
            try:
                self.detection_module.stop()
                self.detection_module.join(timeout=2)
            except Exception:
                pass
        print("Shutdown complete.")


def main():
    app = QApplication(sys.argv)
    available_rect = app.primaryScreen().availableGeometry()
    app.setStyleSheet(GLOBAL_STYLESHEET)

    mgr = ApplicationManager()
    mgr.initialize_gui(available_rect)

    if AUTO_CONNECT:
        mgr.current_sensor_id = resolve_sensor_id(DEFAULT_PORT, default_id=1)
        mgr.initialize_detection_module()
        mgr.initialize_receiver(DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE)
    else:
        mgr.initialize_detection_module()

    app.aboutToQuit.connect(mgr.shutdown)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()