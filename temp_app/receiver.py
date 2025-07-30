import serial
import time
from datetime import datetime
import threading
import csv
import os

class ReceiverModule(threading.Thread):
    """
    리팩토링된 ReceiverModule: DetectionModule로만 데이터를 전달
    8x8 배열 데이터를 효율적으로 전송
    """
    
    def __init__(self, output_queue, detection_queue=None, port='COM4', baudrate=38400):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.output_queue = output_queue  # DetectionModule로 데이터 전달용 큐
        # detection_queue는 더 이상 사용하지 않음 (하위 호환성을 위해 매개변수 유지)
        
        self.ser = None
        self.running = True
        
        # 데이터 수신 버퍼
        self.current_values_buffer = []
        self.expecting_data = False
        
        # 통계 정보
        self.total_packets_received = 0
        self.total_packets_processed = 0
        self.last_packet_time = None

    def run(self):
        """메인 실행 루프"""
        try:
            # 시리얼 포트 초기화
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"ReceiverModule: {self.port} 포트가 {self.baudrate} 보드레이트로 열렸습니다.")
            print(f"ReceiverModule: DetectionModule로만 데이터를 전달합니다.")
            
        except serial.SerialException as e:
            print(f"ReceiverModule: 시리얼 포트 열기 오류: {e}")
            self.running = False
            return
        except Exception as e:
            print(f"ReceiverModule: 시리얼 포트 초기화 중 오류 발생: {e}")
            self.running = False
            return

        # 메인 데이터 수신 루프
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    raw_data = self._read_serial_data()
                    if raw_data is not None:
                        self._process_received_data(raw_data)
                else:
                    time.sleep(0.01)  # CPU 과부하 방지

            except serial.SerialTimeoutException:
                pass  # 타임아웃은 정상적인 상황
            except serial.SerialException as e:
                print(f"ReceiverModule: 시리얼 통신 오류 발생: {e}")
                self.running = False
            except Exception as e:
                print(f"ReceiverModule: 예상치 못한 오류 발생: {e} (타입: {type(e).__name__})")
                self.running = False

        # 종료 시 정리
        self._cleanup()

    def _read_serial_data(self):
        """시리얼 데이터 읽기"""
        try:
            raw_data = self.ser.readline().decode('utf-8', errors='replace').strip()
            return raw_data
        except UnicodeDecodeError as ude:
            print(f"ReceiverModule: 디코딩 오류 발생: {ude}")
            return None

    def _process_received_data(self, raw_data):
        """수신된 데이터 처리"""
        if raw_data == 'MOD':
            # 새 데이터 패키지 시작
            self._start_new_data_package()
        elif self.expecting_data:
            # 숫자 데이터 처리
            self._process_numeric_data(raw_data)

    def _start_new_data_package(self):
        """새 데이터 패키지 시작 처리"""
        self.current_values_buffer = []
        self.expecting_data = True
        self.total_packets_received += 1

    def _process_numeric_data(self, raw_data):
        """숫자 데이터 처리"""
        try:
            value = float(raw_data)
            self.current_values_buffer.append(value)

            # 67개 값 수신 완료 시 패키지 처리
            if len(self.current_values_buffer) == 67:
                self._complete_data_package()
                
        except ValueError:
            print(f"ReceiverModule: 숫자가 아닌 값 수신: '{raw_data}'. 버퍼 초기화.")
            self._reset_buffer()

    def _complete_data_package(self):
        """데이터 패키지 완성 및 전달"""
        # 데이터 분리: 센서 온도(1) + 8x8 배열(64) + 기타(2)
        sensor_degree = self.current_values_buffer[0]
        thermal_values = self.current_values_buffer[1:65]  # 8x8 = 64개 값
        etc_values = self.current_values_buffer[65:67]     # 추가 데이터 2개
        
        # 현재 시간
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # DetectionModule로 전달할 데이터 패키지 구성
        data_package = {
            'sensor_degree': sensor_degree,
            'time': current_time,
            'values': thermal_values,  # 8x8 배열 (64개 값)
            'etc': etc_values,
            'packet_id': self.total_packets_processed + 1,
            'source': 'receiver'
        }
        
        # DetectionModule로 전달
        self.output_queue.put(data_package)
        
        # 통계 업데이트
        self.total_packets_processed += 1
        self.last_packet_time = current_time
        
        # CSV 로그 저장 (선택적)
        self._save_to_csv(data_package)
        
        # 버퍼 초기화
        self._reset_buffer()
        
        # 주기적으로 처리 상태 출력
        if self.total_packets_processed % 100 == 0:
            print(f"ReceiverModule: {self.total_packets_processed}개 패키지 처리 완료")

    def _reset_buffer(self):
        """수신 버퍼 초기화"""
        self.current_values_buffer = []
        self.expecting_data = False

    def _save_to_csv(self, data_package):
        """CSV 파일에 데이터 저장 (선택적 기능)"""
        try:
            csv_filename = './thermal_data.csv'
            write_header = not os.path.exists(csv_filename)
            
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                
                # 헤더 작성 (파일이 새로 생성될 때만)
                if write_header:
                    header = ['timestamp', 'packet_id', 'sensor_degree']
                    header.extend([f'thermal_{i:02d}' for i in range(64)])  # thermal_00 ~ thermal_63
                    header.extend(['etc_1', 'etc_2'])
                    csv_writer.writerow(header)
                
                # 데이터 행 작성
                row_data = [
                    data_package['time'],
                    data_package['packet_id'],
                    data_package['sensor_degree']
                ]
                row_data.extend(data_package['values'])  # 64개 온도 값
                row_data.extend(data_package['etc'])     # 추가 데이터 2개
                
                csv_writer.writerow(row_data)
                
        except Exception as e:
            print(f"ReceiverModule: CSV 저장 오류: {e}")

    def _cleanup(self):
        """종료 시 정리 작업"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"ReceiverModule: {self.port} 포트가 닫혔습니다.")
        
        print(f"ReceiverModule: 총 {self.total_packets_received}개 패키지 수신, {self.total_packets_processed}개 처리 완료")

    def stop(self):
        """스레드 종료 요청"""
        print("ReceiverModule: 종료 요청됨")
        self.running = False

    def get_status(self):
        """현재 상태 정보 반환"""
        return {
            'running': self.running,
            'port': self.port,
            'baudrate': self.baudrate,
            'total_received': self.total_packets_received,
            'total_processed': self.total_packets_processed,
            'last_packet_time': self.last_packet_time,
            'buffer_size': len(self.current_values_buffer),
            'expecting_data': self.expecting_data
        }