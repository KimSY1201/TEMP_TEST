import sys
from queue import Queue
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# 리팩토링된 모듈들 import
from detector import DetectionModule
from gui import OutputModule, DataSignal, GLOBAL_STYLESHEET
from receiver import ReceiverModule


def check_detection_queue(detection_output_queue: Queue, signal_obj: DataSignal):
    """
    DetectionModule의 출력 큐를 주기적으로 확인하여 GUI 업데이트 시그널을 발생시키는 함수
    이제 DetectionModule에서 처리된 결과만 GUI로 전달됩니다.
    """ 
    if not detection_output_queue.empty():
        # DetectionModule에서 처리된 데이터 패키지를 가져옴
        processed_data_package = detection_output_queue.get()
        # GUI 업데이트 시그널 발생
        signal_obj.update_data_signal.emit(processed_data_package)


def on_app_exit(receiver: ReceiverModule, detection_module: DetectionModule):
    """
    애플리케이션 종료 시 실행될 함수. 모든 스레드를 안전하게 종료합니다.
    """
    print("Application closing. Stopping modules...")
    
    # ReceiverModule 종료
    receiver.stop()
    receiver.join()
    
    # DetectionModule 종료
    detection_module.stop()
    
    print("All modules stopped safely.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    primary_screen = app.primaryScreen()
    available_rect = primary_screen.availableGeometry()
        
    # GUI에 전역 스타일시트를 적용합니다.
    app.setStyleSheet(GLOBAL_STYLESHEET)

    # 모듈 간 데이터 통신을 위한 큐 생성
    # 데이터 흐름: Receiver -> DetectionModule -> GUI
    receiver_to_detector_queue = Queue()  # Receiver -> DetectionModule
    detector_to_gui_queue = Queue()       # DetectionModule -> GUI

    # PyQt 시그널 객체 생성 (GUI 업데이트용)
    data_signal_obj = DataSignal()

    # 1. 리시버 모듈 인스턴스 생성 및 시작
    # 이제 DetectionModule로만 데이터를 전달합니다 (GUI로 직접 전달하지 않음)
    print("Starting ReceiverModule...")
    receiver = ReceiverModule(
        output_queue=receiver_to_detector_queue,  # DetectionModule로만 전달
        detection_queue=None,  # 더 이상 사용하지 않음
        port='COM3', 
        baudrate=57600
    )
    receiver.start()

    # 2. 감지 모듈 인스턴스 생성 및 시작
    # DetectionModule이 Receiver로부터 데이터를 받아 처리한 후 GUI로 전달
    print("Starting DetectionModule...")
    detection_module = DetectionModule(
        input_queue=receiver_to_detector_queue,   # Receiver로부터 데이터 받음
        output_queue=detector_to_gui_queue,       # GUI로 데이터 전달
        threshold=5.0, 
        filename="detected_values.txt"
    )
    detection_module.start()

    # 3. 출력 모듈 (GUI) 인스턴스 생성 및 표시
    print("Starting GUI...")
    output_module_gui = OutputModule(data_signal_obj, available_rect)
    output_module_gui.showMaximized()  # 전체화면으로 표시

    # 4. GUI 업데이트를 위한 타이머 설정
    # DetectionModule의 출력 큐만 모니터링합니다
    timer = QTimer()
    timer.setInterval(50)  # 50ms마다 큐 확인 (더 빠른 반응성)
    timer.timeout.connect(lambda: check_detection_queue(detector_to_gui_queue, data_signal_obj))
    timer.start()

    # 5. 애플리케이션 종료 시 스레드들을 안전하게 종료하기 위한 연결
    app.aboutToQuit.connect(lambda: on_app_exit(receiver, detection_module))

    print("System started successfully. Data flow: Receiver -> DetectionModule -> GUI")
    print("Press Ctrl+C or close the window to exit.")

    # 애플리케이션 실행
    sys.exit(app.exec())