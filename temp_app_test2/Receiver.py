import serial
import time
from datetime import datetime
import threading
import random # 테스트용 임의 값 생성을 위해 추가
import csv
import os


class ReceiverModule(threading.Thread):
    """
    시리얼 포트에서 센서 데이터를 수신하는 모듈
    리팩토링 변경사항:
    - 동적 포트 전환 지원
    - 향상된 에러 처리 및 복구 기능
    - 연결 상태 모니터링
    """
    
    def __init__(self, detection_queue, port='COM3', baudrate=57600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.detection_queue = detection_queue
        self.ser = None
        self.running = True # 스레드 실행/종료 제어 플래그
        self.connected = False # 연결 상태 플래그
        
        # 데이터 수신 관련 변수
        self.current_values_buffer = [] # 64개 값을 모을 임시 버퍼
        self.expecting_data = False     # 'MOD'를 받은 후 데이터를 기대하는 상태 플래그
        
        # 통계 및 모니터링
        self.total_packages_received = 0
        self.last_receive_time = None
        self.connection_retry_count = 0
        self.max_retry_attempts = 3
        
        # CSV 로깅 경로
        self.csv_path = None
        self._setup_csv_path()
        
    def _setup_csv_path(self):
        """CSV 저장 경로 설정"""
        try:
            cwd = os.getcwd().replace('\\', '/')
            data_dir = f"{cwd}/_data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            self.csv_path = f"{data_dir}/degree_test.csv"
        except Exception as e:
            print(f"ReceiverModule: CSV 경로 설정 실패: {e}")
            self.csv_path = None

    def connect_to_port(self):
        """포트에 연결 시도"""
        if self.ser and self.ser.is_open:
            self.disconnect_from_port()
            
        try:
            print(f"ReceiverModule: {self.port} 포트 연결 시도 (보드레이트: {self.baudrate})")
            # timeout을 적절히 설정합니다. 너무 짧으면 불완전한 데이터만 읽을 수 있습니다.
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1) 
            self.connected = True
            self.connection_retry_count = 0
            print(f"ReceiverModule: {self.port} 포트 연결 성공")
            return True
            
        except serial.SerialException as e:
            print(f"ReceiverModule: 시리얼 포트 연결 실패: {e}")
            self.connected = False
            self.connection_retry_count += 1
            return False
        except Exception as e:
            print(f"ReceiverModule: 포트 연결 중 예상치 못한 오류: {e}")
            self.connected = False
            self.connection_retry_count += 1
            return False

    def disconnect_from_port(self):
        """포트 연결 해제"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
                print(f"ReceiverModule: {self.port} 포트 연결 해제됨")
            except Exception as e:
                print(f"ReceiverModule: 포트 연결 해제 중 오류: {e}")
        
        self.connected = False
        self.ser = None

    def change_port(self, new_port, new_baudrate=None):
        """포트 및 보드레이트 변경"""
        if new_port == "" or new_port is None:
            # 빈 포트는 연결 해제를 의미
            self.disconnect_from_port()
            self.port = ""
            return True
            
        # 기존 연결 해제
        if self.connected:
            self.disconnect_from_port()
        
        # 새 설정 적용
        self.port = new_port
        if new_baudrate:
            self.baudrate = new_baudrate
            
        print(f"ReceiverModule: 포트 변경됨 - {self.port}, 보드레이트: {self.baudrate}")
        
        # 새 포트로 연결 시도
        return self.connect_to_port()

    def save_to_csv(self, sensor_degree, values):
        """데이터를 CSV 파일에 저장"""
        if not self.csv_path:
            return
            
        try:
            # 파일 존재 여부에 따라 쓰기 모드 결정
            write_type = 'w' if not os.path.exists(self.csv_path) else 'a'
            
            # with open(self.csv_path, write_type, newline='') as csvfile:
            #     csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     result = [datetime.now(), sensor_degree]
            #     result.extend(values)
            #     csv_writer.writerow(result)
                
        except Exception as e:
            print(f"ReceiverModule: CSV 저장 실패: {e}")

    def process_complete_package(self, sensor_degree, received_values, etc_values):
        """완성된 데이터 패키지 처리"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        data_package = {
            'sensor_degree': sensor_degree, 
            'time': current_time, 
            'values': received_values, 
            'etc': etc_values
        }
        
        # DetectionModule로 데이터 전송
        self.detection_queue.put(data_package)
        
        # CSV 파일에 저장
        self.save_to_csv(sensor_degree, received_values)
        
        # 통계 업데이트
        self.total_packages_received += 1
        self.last_receive_time = datetime.now()
        
        # 주기적으로 통계 출력 (100개 패키지마다)
        if self.total_packages_received % 100 == 0:
            print(f"ReceiverModule: 총 {self.total_packages_received}개 패키지 수신 완료")

    def parse_data_38400(self, raw_data):
        """38400 보드레이트용 데이터 파싱"""
        if raw_data == 'MOD':
            # 'MOD'를 받으면 새 패키지 시작으로 간주하고 버퍼 초기화
            self.current_values_buffer = []
            self.expecting_data = True
        elif self.expecting_data:
            try:
                value = float(raw_data)
                self.current_values_buffer.append(value)

                if len(self.current_values_buffer) == 65:
                    # 센서 주변 온도값 1 + 64개값
                    sensor_degree = self.current_values_buffer[0]
                    received_values = self.current_values_buffer[1:65]
                    etc_values = [0, 0] 
                    
                    # 버퍼 초기화
                    self.current_values_buffer = []
                    self.expecting_data = False
                    
                    # 완성된 패키지 처리
                    self.process_complete_package(sensor_degree, received_values, etc_values)
                    
            except ValueError:
                print(f"ReceiverModule: 경고: 'MOD' 후 숫자가 아닌 값 수신: '{raw_data}'. 버퍼 초기화.")
                self.current_values_buffer = []
                self.expecting_data = False

    def parse_data_57600(self, raw_data):
        """57600 보드레이트용 데이터 파싱"""
        if raw_data == 'MOD':
            # 'MOD'를 받으면 새 패키지 시작으로 간주하고 버퍼 초기화
            self.current_values_buffer = []
            self.expecting_data = True
        elif self.expecting_data:
            try:
                value = float(raw_data)
                self.current_values_buffer.append(value)

                if len(self.current_values_buffer) == 67:
                    # 센서 주변 온도값 1 + 64개값 + 추가 2개값
                    sensor_degree = self.current_values_buffer[0]
                    received_values = self.current_values_buffer[1:65]
                    etc_values = self.current_values_buffer[65:67] 
                    
                    # 버퍼 초기화
                    self.current_values_buffer = []
                    self.expecting_data = False
                    
                    # 완성된 패키지 처리
                    self.process_complete_package(sensor_degree, received_values, etc_values)
                    
            except ValueError:
                print(f"ReceiverModule: 경고: 'MOD' 후 숫자가 아닌 값 수신: '{raw_data}'. 버퍼 초기화.")
                self.current_values_buffer = []
                self.expecting_data = False

    def run(self):
        """스레드 메인 루프"""
        print("ReceiverModule: 스레드 시작됨")
        
        # 초기 연결 시도
        if self.port:
            self.connect_to_port()

        while self.running:
            try:
                # 연결되지 않은 상태에서는 대기
                if not self.connected or not self.ser or not self.ser.is_open:
                    time.sleep(0.5)
                    continue

                # COM 포트에 데이터가 들어올 때까지 대기
                if self.ser.in_waiting > 0:
                    try:
                        # 한 줄을 읽고 UTF-8로 디코딩 후 양쪽 공백 제거
                        raw_data = self.ser.readline().decode('utf-8', errors='replace').strip()
                        
                        if not raw_data:  # 빈 데이터는 무시
                            continue
                            
                    except UnicodeDecodeError as ude:
                        print(f"ReceiverModule: 디코딩 오류: {ude}")
                        time.sleep(0.01)
                        continue
                    except Exception as e:
                        print(f"ReceiverModule: 데이터 읽기 오류: {e}")
                        time.sleep(0.01)
                        continue
                    
                    # 보드레이트에 따른 데이터 파싱
                    if self.baudrate == 38400:
                        self.parse_data_38400(raw_data)
                    else:  # 57600 및 기타 보드레이트
                        self.parse_data_57600(raw_data)
                        
                else:
                    # 데이터가 들어오지 않으면 잠시 대기 (CPU 과부하 방지)
                    time.sleep(0.01)

            except serial.SerialTimeoutException:
                # 타임아웃은 정상적인 상황이므로 로그 출력하지 않음
                pass
                
            except serial.SerialException as e:
                print(f"ReceiverModule: 시리얼 통신 오류: {e}")
                self.connected = False
                
                # 재연결 시도
                if self.connection_retry_count < self.max_retry_attempts:
                    print(f"ReceiverModule: 재연결 시도 ({self.connection_retry_count + 1}/{self.max_retry_attempts})")
                    time.sleep(2)  # 2초 대기 후 재연결 시도
                    if self.port:
                        self.connect_to_port()
                else:
                    print(f"ReceiverModule: 최대 재연결 시도 횟수 초과. 대기 상태로 전환.")
                    time.sleep(5)  # 5초 대기 후 다시 시도
                    self.connection_retry_count = 0  # 재시도 카운터 리셋
                    
            except Exception as e:
                print(f"ReceiverModule: 예상치 못한 오류: {e} (타입: {type(e).__name__})")
                time.sleep(1)  # 1초 대기 후 계속

        # 스레드 종료 시 연결 해제
        self.disconnect_from_port()
        print("ReceiverModule: 스레드 종료됨")

    def stop(self):
        """스레드 중지"""
        print("ReceiverModule: 종료 요청됨")
        self.running = False
        
    def get_status(self):
        """현재 상태 반환"""
        return {
            'connected': self.connected,
            'port': self.port,
            'baudrate': self.baudrate,
            'total_packages': self.total_packages_received,
            'last_receive': self.last_receive_time,
            'retry_count': self.connection_retry_count
        }
        
    def is_connected(self):
        """연결 상태 확인"""
        return self.connected and self.ser and self.ser.is_open