import sys
from queue import Queue
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# 각 모듈 파일에서 필요한 클래스들을 import 합니다.
# 파일 이름은 실제 프로젝트 구조에 맞게 유지해야 합니다. (gui_4.py, detector.py, Receiver.py)
from detector import DetectionModule
from gui import OutputModule, DataSignal, GLOBAL_STYLESHEET # gui_4에서 스타일시트도 가져옵니다.
from Receiver import ReceiverModule


def check_queues(q: Queue, signal_obj: DataSignal):
    """
    큐를 주기적으로 확인하여 데이터가 있으면 GUI 업데이트 시그널을 발생시키는 함수
    """ 
    if not q.empty():
        # 큐에서 데이터 패키지를 가져옴
        data_package = q.get()
        # 올바른 시그널 이름으로 emit 호출 (가장 중요한 수정사항)
        signal_obj.update_data_signal.emit(data_package)


def on_app_exit(receiver: ReceiverModule, detection_module: DetectionModule):
    """
    애플리케이션 종료 시 실행될 함수. 모든 스레드를 안전하게 종료합니다.
    """
    print("Application closing. Stopping modules...")
    receiver.stop()
    detection_module.stop()
    receiver.join()
    detection_module.join()
    print("Modules stopped.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    primary_screen = app.primaryScreen()
    available_rect = primary_screen.availableGeometry()
        
    # GUI에 전역 스타일시트를 적용합니다.
    app.setStyleSheet(GLOBAL_STYLESHEET)

    # 모듈 간 데이터 통신을 위한 큐 생성
    output_queue = Queue()      # Receiver -> GUI 로 전달될 데이터를 담는 큐
    detection_queue = Queue()   # Receiver -> Detector 로 전달될 데이터를 담는 큐

    # PyQt 시그널 객체 생성 (GUI 업데이트용)
    data_signal_obj = DataSignal()

    # 1. 리시버 모듈 인스턴스 생성 및 시작
    # COM 포트와 보드레이트는 실제 환경에 맞게 조절하세요.
    # receiver = ReceiverModule(output_queue, detection_queue, port='COM4', baudrate=38400)
    receiver = ReceiverModule(detection_queue, port='COM3', baudrate=57600)
    # receiver = ReceiverModule(detection_queue, port='COM3', baudrate=38400)
    receiver.start()

    # 2. 출력 모듈 (GUI) 인스턴스 생성 및 표시
    output_module_gui = OutputModule(data_signal_obj,available_rect)
    output_module_gui.show()

    # 3. 감지 모듈 인스턴스 생성 및 시작
    detection_module = DetectionModule(data_signal_obj, detection_queue, output_queue, threshold=5.0, filename="detected_values.txt")
    detection_module.start()

    # GUI 업데이트를 위한 타이머 설정
    timer = QTimer()
    timer.setInterval(100)  # 100ms마다 큐 확인
    # 'check_queues' 함수를 타이머의 timeout 시그널에 연결
    timer.timeout.connect(lambda: check_queues(output_queue, data_signal_obj))
    timer.start()

    # 애플리케이션 종료 시 스레드들을 안전하게 종료하기 위한 람다 함수 연결
    app.aboutToQuit.connect(lambda: on_app_exit(receiver, detection_module))

    sys.exit(app.exec())