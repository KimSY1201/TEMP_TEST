import sys
from queue import Queue
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
import os

# 각 모듈 파일에서 필요한 클래스들을 import 합니다.
from detector import DetectionModule
from gui import OutputModule, DataSignal, GLOBAL_STYLESHEET
from Receiver import ReceiverModule


class ApplicationManager:
    """애플리케이션 전체를 관리하는 클래스"""
    
    def __init__(self):
        # 모듈 간 데이터 통신을 위한 큐 생성
        self.output_queue = Queue()      # Detector -> GUI 로 전달될 데이터를 담는 큐
        self.detection_queue = Queue()   # Receiver -> Detector 로 전달될 데이터를 담는 큐

        # PyQt 시그널 객체 생성 (GUI 업데이트용)
        self.data_signal_obj = DataSignal()
        
        # 모듈 인스턴스들
        self.receiver = None
        self.detection_module = None
        self.output_module_gui = None
        
        # GUI 업데이트 타이머
        self.timer = QTimer()
        self.timer.setInterval(100)  # 100ms마다 큐 확인
        self.timer.timeout.connect(self.check_queues)
        
    def initialize_gui(self, available_rect):
        """GUI 초기화 (가장 먼저 실행)"""
        print("GUI 모듈 초기화 중...")
        
        # GUI 인스턴스 생성 및 표시
        self.output_module_gui = OutputModule(self.data_signal_obj, available_rect)
        
        # GUI에서 포트 변경 시그널 연결
        self.data_signal_obj.port_change_signal.connect(self.on_port_changed)
        
        self.output_module_gui.show()
        print("GUI 모듈 초기화 완료")
        
    def initialize_detection_module(self):
        """감지 모듈 초기화"""
        print("감지 모듈 초기화 중...")
        
        # 감지 모듈 인스턴스 생성 및 시작
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        self.detection_module = DetectionModule(
            self.data_signal_obj, 
            self.detection_queue, 
            self.output_queue, 
            threshold=5.0, 
            filename=f"{current_dir}/_data/detected_values.txt"
        )
        self.detection_module.start()
        
        # GUI 업데이트 타이머 시작
        self.timer.start()
        
        print("감지 모듈 초기화 완료")
        
    def initialize_receiver(self, port, baudrate=57600):
        """리시버 모듈 초기화"""
        print(f"리시버 모듈 초기화 중... (포트: {port}, 보드레이트: {baudrate})")
        
        if self.receiver is not None:
            # 기존 리시버가 있다면 중지
            self.stop_receiver()
            
        try:
            # 새 리시버 인스턴스 생성 및 시작
            self.receiver = ReceiverModule(self.detection_queue, port=port, baudrate=baudrate)
            self.receiver.start()
            
            # GUI에 연결 상태 업데이트
            if self.output_module_gui:
                self.output_module_gui.update_connection_status(True, port)
                
            print(f"리시버 모듈 초기화 완료 (포트: {port})")
            return True
            
        except Exception as e:
            print(f"리시버 모듈 초기화 실패: {e}")
            
            # GUI에 연결 실패 상태 업데이트
            if self.output_module_gui:
                self.output_module_gui.update_connection_status(False, port, str(e))
            return False
    
    def on_port_changed(self, port_info):
        """GUI에서 포트 변경 요청 시 호출"""
        port = port_info.get('port', 'COM3')
        baudrate = port_info.get('baudrate', 57600)
        
        if (port, baudrate) == (0, 0):
            if self.receiver.ser and self.receiver.ser.is_open:
                self.receiver.ser.close()
                print("포트 연결 종료")
                self.initialize_receiver(port, baudrate)
                return
        
        print(f"포트 변경 요청: {port} (보드레이트: {baudrate})")
        
        # 리시버 재초기화
        success = self.initialize_receiver(port, baudrate)
        
        if success:
            print(f"포트 변경 완료: {port}")
        else:
            print(f"포트 변경 실패: {port}")
    
    def stop_receiver(self):
        """리시버 모듈 중지"""
        if self.receiver and self.receiver.is_alive():
            print("리시버 모듈 중지 중...")
            self.receiver.stop()
            self.receiver.join(timeout=2)
            
            if self.receiver.is_alive():
                print("경고: 리시버 모듈이 정상적으로 종료되지 않았습니다.")
            else:
                print("리시버 모듈 중지 완료")
                
            # GUI에 연결 해제 상태 업데이트
            if self.output_module_gui:
                self.output_module_gui.update_connection_status(False, "")
    
    def check_queues(self):
        """큐를 주기적으로 확인하여 데이터가 있으면 GUI 업데이트 시그널을 발생시키는 함수""" 
        if not self.output_queue.empty():
            # 큐에서 데이터 패키지를 가져옴
            data_package = self.output_queue.get()
            # GUI 업데이트 시그널 발송
            self.data_signal_obj.update_data_signal.emit(data_package)
    
    def shutdown(self):
        """애플리케이션 종료 시 실행될 함수. 모든 스레드를 안전하게 종료합니다."""
        print("애플리케이션 종료 중...")
        
        # 타이머 중지
        if self.timer.isActive():
            self.timer.stop()
        
        # 리시버 모듈 중지
        self.stop_receiver()
        
        # 감지 모듈 중지
        if self.detection_module and self.detection_module.is_alive():
            print("감지 모듈 중지 중...")
            self.detection_module.stop()
            self.detection_module.join(timeout=2)
            
            if self.detection_module.is_alive():
                print("경고: 감지 모듈이 정상적으로 종료되지 않았습니다.")
            else:
                print("감지 모듈 중지 완료")
        
        print("애플리케이션 종료 완료")


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    primary_screen = app.primaryScreen()
    available_rect = primary_screen.availableGeometry()
        
    # GUI에 전역 스타일시트를 적용합니다.
    app.setStyleSheet(GLOBAL_STYLESHEET)

    # 애플리케이션 매니저 생성
    app_manager = ApplicationManager()
    
    # 1. GUI 먼저 초기화 (가장 우선)
    app_manager.initialize_gui(available_rect)
    
    # 2. 감지 모듈 초기화
    app_manager.initialize_detection_module()
    
    # 3. GUI에서 기본 포트 설정으로 리시버 초기화 시도
    # (사용자가 GUI에서 포트를 선택할 때까지는 연결하지 않음)
    
    # 애플리케이션 종료 시 안전한 종료를 위한 연결
    app.aboutToQuit.connect(app_manager.shutdown)

    # 애플리케이션 실행
    sys.exit(app.exec())


if __name__ == '__main__':
    main()