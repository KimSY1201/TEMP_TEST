import sys
from queue import Queue
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
import os

# 기존 모듈 파일에서 필요한 클래스들을 import 합니다.
from detector import DetectionModule
from gui import OutputModule, DataSignal, GLOBAL_STYLESHEET
from Receiver import ReceiverModule

# 새로 추가된 송수신 모듈
from transmission import TransmissionManager, TransmissionSignal


class ApplicationManager:
    """애플리케이션 전체를 관리하는 클래스"""
    
    def __init__(self):
        # 기존 모듈 간 데이터 통신을 위한 큐들
        self.output_queue = Queue()      # Detector -> GUI 로 전달될 데이터를 담는 큐
        self.detection_queue = Queue()   # Receiver -> Detector 로 전달될 데이터를 담는 큐

        # PyQt 시그널 객체들
        self.data_signal_obj = DataSignal()
        self.transmission_signal = TransmissionSignal()  # 새로 추가
        
        # 기존 모듈 인스턴스들
        self.receiver = None
        self.detection_module = None
        self.output_module_gui = None
        
        # 새로 추가된 송수신 모듈
        self.transmission_manager = TransmissionManager(self.transmission_signal)
        
        # GUI 업데이트 타이머
        self.timer = QTimer()
        self.timer.setInterval(100)  # 100ms마다 큐 확인
        self.timer.timeout.connect(self.check_queues)
        
        # 송수신 모듈 시그널 연결
        self.setup_transmission_signals()
        
    def setup_transmission_signals(self):
        """송수신 모듈 시그널 연결 설정"""
        # 전송 상태 업데이트
        self.transmission_signal.status_update.connect(self.on_transmission_status_update)
        
        # 연결 상태 변경
        self.transmission_signal.connection_status.connect(self.on_transmission_connection_change)
        
        # 데이터 전송 완료
        self.transmission_signal.data_transmitted.connect(self.on_data_transmitted)
        
        # 오류 발생
        self.transmission_signal.error_occurred.connect(self.on_transmission_error)
        
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
        
    def initialize_transmission_module(self, sender_host='localhost', sender_port=8080, receiver_port=8081):
        """송수신 모듈 초기화"""
        print("송수신 모듈 초기화 중...")
        
        try:
            # 송수신 모듈 설정
            self.transmission_manager.configure_sender(sender_host, sender_port)
            self.transmission_manager.configure_receiver('0.0.0.0', receiver_port)
            
            # 송수신 모듈 시작
            sender_ok, receiver_ok = self.transmission_manager.start_all()
            
            if sender_ok:
                print(f"TCP 송신 모듈 시작됨: {sender_host}:{sender_port}")
            else:
                print("TCP 송신 모듈 시작 실패")
                
            if receiver_ok:
                print(f"TCP 수신 모듈 시작됨: 포트 {receiver_port}")
            else:
                print("TCP 수신 모듈 시작 실패")
            
            return sender_ok or receiver_ok
            
        except Exception as e:
            print(f"송수신 모듈 초기화 실패: {e}")
            return False
        
    def initialize_receiver(self, port, baudrate=57600):
        
        """리시버 모듈 초기화 (기존 코드 유지)"""
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
        """GUI에서 포트 변경 요청 시 호출 (기존 코드 유지)"""
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
        """리시버 모듈 중지 (기존 코드 유지)"""
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
        """큐를 주기적으로 확인하여 데이터가 있으면 처리하는 함수 (기존 + 새 기능)"""
        # 기존: Detector -> GUI 데이터 처리
        if not self.output_queue.empty():
            # 큐에서 데이터 패키지를 가져옴
            data_package = self.output_queue.get()
            # GUI 업데이트 시그널 발송
            self.data_signal_obj.update_data_signal.emit(data_package)
            
            # 새로 추가: Detector 데이터를 TCP/IP로 전송
            self.send_detector_data_to_tcp(data_package)
        
        # 새로 추가: TCP 수신 데이터 처리
        self.check_received_tcp_data()
    
    def send_detector_data_to_tcp(self, data_package):
        """Detector 모듈의 데이터를 TCP/IP로 전송"""
        try:
            # 송수신 매니저를 통해 데이터 전송
            success = self.transmission_manager.send_detector_data(data_package)
            
            # if success:
            #     print(f"Detector 데이터 TCP 전송 완료: {type(data_package)}")
            # else:
            #     print("Detector 데이터 TCP 전송 실패 (큐 가득참)")
                
        except Exception as e:
            print(f"Detector 데이터 TCP 전송 오류: {e}")
    
    def check_received_tcp_data(self):
        """TCP로 수신된 데이터 확인 및 처리"""
        received_data = self.transmission_manager.get_received_data()
        
        if received_data:
            print(f"TCP 데이터 수신됨: {received_data['source']} -> {len(str(received_data['data']))} bytes")
            
            # 필요시 수신된 데이터를 다른 모듈로 전달
            # 예: self.some_module.process_received_data(received_data)
    
    # 송수신 모듈 이벤트 핸들러들
    def on_transmission_status_update(self, status):
        """송수신 상태 업데이트 처리"""
        # 송신 모듈 상태
        if 'sender' in status:
            sender_status = status['sender']
            if sender_status['running']:
                sent_count = sender_status['stats'].get('total_sent', 0)
                error_count = sender_status['stats'].get('errors', 0)
                # print(f"송신 상태: 전송={sent_count}, 오류={error_count}, 연결={'OK' if sender_status['connected'] else 'FAIL'}")
        
        # 수신 모듈 상태
        if 'receiver' in status:
            receiver_status = status['receiver']
            if receiver_status['running']:
                received_count = receiver_status['stats'].get('total_received', 0)
                client_count = receiver_status['stats'].get('client_count', 0)
                # print(f"수신 상태: 수신={received_count}, 클라이언트={client_count}")
        
        # 큐 상태
        if 'queue_sizes' in status:
            queue_info = status['queue_sizes']
            send_queue_size = queue_info.get('send_queue', 0)
            receive_queue_size = queue_info.get('receive_queue', 0)
            if send_queue_size > 0 or receive_queue_size > 0:
                print(f"큐 상태: 송신대기={send_queue_size}, 수신대기={receive_queue_size}")
    
    def on_transmission_connection_change(self, connection_info):
        """송수신 연결 상태 변경 처리"""
        conn_type = connection_info.get('type', 'unknown')
        message = connection_info.get('message', '')
        
        if conn_type == 'tcp_sender':
            connected = connection_info.get('connected', False)
            status_text = "연결됨" if connected else "연결 해제"
            print(f"TCP 송신: {status_text} - {message}")
            
        elif conn_type == 'tcp_receiver':
            if 'client_connected' in connection_info:
                client_connected = connection_info['client_connected']
                client_info = connection_info.get('client_info', '')
                status_text = "연결됨" if client_connected else "해제됨"
                print(f"TCP 수신 클라이언트 {client_info}: {status_text}")
            else:
                print(f"TCP 수신 서버: {message}")
    
    def on_data_transmitted(self, transmission_info):
        """데이터 전송 완료 처리"""
        direction = transmission_info.get('direction', '')
        size = transmission_info.get('size', 0)
        timestamp = transmission_info.get('timestamp', '')
        
        if direction == 'sent':
            data_type = transmission_info.get('data_type', '')
            # print(f"[{timestamp[:19]}] 송신 완료: {size} bytes ({data_type})")
            
        elif direction == 'received':
            client = transmission_info.get('client', '')
            # print(f"[{timestamp[:19]}] 수신 완료: {size} bytes from {client}")
            
        elif direction == 'received_from_client':
            client = transmission_info.get('client', '')
            # print(f"[{timestamp[:19]}] 클라이언트 수신: {size} bytes from {client}")
    
    def on_transmission_error(self, error_message):
        """송수신 오류 처리"""
        print(f"송수신 오류: {error_message}")
    
    def configure_transmission(self, sender_host='localhost', sender_port=8080, receiver_port=8081):
        """송수신 모듈 설정 변경"""
        print(f"송수신 설정 변경: 송신={sender_host}:{sender_port}, 수신=포트{receiver_port}")
        
        # 기존 송수신 모듈 중지
        self.transmission_manager.stop_all()
        
        # 새 설정으로 재시작
        return self.initialize_transmission_module(sender_host, sender_port, receiver_port)
    
    def get_transmission_statistics(self):
        """송수신 통계 정보 반환"""
        stats = {
            'sender': {
                'running': False,
                'connected': False,
                'total_sent': 0,
                'total_bytes': 0,
                'errors': 0,
                'last_transmission': None
            },
            'receiver': {
                'running': False,
                'total_received': 0,
                'total_bytes': 0,
                'client_count': 0,
                'total_connections': 0
            }
        }
        
        if self.transmission_manager.sender:
            sender_stats = self.transmission_manager.sender.stats
            stats['sender'].update({
                'running': self.transmission_manager.sender.running,
                'connected': self.transmission_manager.sender.connected,
                'total_sent': sender_stats.get('total_sent', 0),
                'total_bytes': sender_stats.get('total_bytes', 0),
                'errors': sender_stats.get('errors', 0),
                'last_transmission': sender_stats.get('last_transmission')
            })
        
        if self.transmission_manager.receiver:
            receiver_stats = self.transmission_manager.receiver.stats
            stats['receiver'].update({
                'running': self.transmission_manager.receiver.running,
                'total_received': receiver_stats.get('total_received', 0),
                'total_bytes': receiver_stats.get('total_bytes', 0),
                'client_count': receiver_stats.get('client_count', 0),
                'total_connections': receiver_stats.get('total_connections', 0)
            })
        
        return stats
    
    def shutdown(self):
        """애플리케이션 종료 시 실행될 함수. 모든 스레드를 안전하게 종료합니다."""
        print("애플리케이션 종료 중...")
        
        # 타이머 중지
        if self.timer.isActive():
            self.timer.stop()
        
        # 송수신 모듈 중지
        print("송수신 모듈 중지 중...")
        self.transmission_manager.stop_all()
        print("송수신 모듈 중지 완료")
        
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
    
    # 3. 송수신 모듈 초기화 (새로 추가)
    # 설정: TCP 송신은 localhost:8080, TCP 수신은 포트 8080
    transmission_success = app_manager.initialize_transmission_module(
        # sender_host='localhost',  # 목적지 서버 IP (필요시 변경)
        # sender_host='116.126.127.76',  # 나
        sender_host='116.126.127.75',  # 옆
        sender_port=8080,         # 목적지 서버 포트
        receiver_port=8081        # 이 애플리케이션의 수신 포트
    )
    
    if transmission_success:
        print("송수신 모듈이 성공적으로 시작되었습니다.")
        print("- Detector 데이터는 자동으로 TCP/IP로 전송됩니다.")
        print("- TCP 포트 8081에서 데이터 수신 대기 중입니다.")
    else:
        print("송수신 모듈 시작에 일부 실패했지만 애플리케이션을 계속 실행합니다.")
    
    # 4. GUI에서 기본 포트 설정으로 리시버 초기화 시도
    # (사용자가 GUI에서 포트를 선택할 때까지는 연결하지 않음)
    
    # 애플리케이션 종료 시 안전한 종료를 위한 연결
    app.aboutToQuit.connect(app_manager.shutdown)

    # 실행 중 통계 정보 출력 (선택사항)
    def print_statistics():
        """주기적으로 통계 정보 출력"""
        stats = app_manager.get_transmission_statistics()
        if stats['sender']['total_sent'] > 0 or stats['receiver']['total_received'] > 0:
            print(f"\n=== 송수신 통계 ===")
            print(f"송신: {stats['sender']['total_sent']}개, {stats['sender']['total_bytes']}bytes, 오류: {stats['sender']['errors']}")
            print(f"수신: {stats['receiver']['total_received']}개, {stats['receiver']['total_bytes']}bytes, 클라이언트: {stats['receiver']['client_count']}")
            print("==================\n")
    
    # 통계 출력 타이머 (선택사항 - 필요없으면 주석 처리)
    stats_timer = QTimer()
    stats_timer.timeout.connect(print_statistics)
    stats_timer.start(10000)  # 10초마다 통계 출력

    print("\n=== 애플리케이션 시작됨 ===")
    print("기능:")
    print("- RS-232 데이터 수신 및 감지")
    print("- 감지된 데이터의 TCP/IP 자동 전송")
    print("- TCP/IP 데이터 수신 대기")
    print("- 실시간 GUI 표시")
    print("=====================================\n")

    # 애플리케이션 실행
    sys.exit(app.exec())


if __name__ == '__main__':
    main()