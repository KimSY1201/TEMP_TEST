import serial
import time
from datetime import datetime
import threading
import random # 테스트용 임의 값 생성을 위해 추가
import csv
import os
# import numpy as np # ReceiverModule에서는 직접적으로 사용되지 않으므로 제거 가능


class ReceiverModule(threading.Thread):
    def __init__(self, detection_queue, port='COM4', baudrate=38400):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.detection_queue = detection_queue
        self.ser = None
        self.running = True # 스레드 실행/종료 제어 플래그
        
        self.current_values_buffer = [] # 64개 값을 모을 임시 버퍼
        self.expecting_data = False     # 'MOD'를 받은 후 데이터를 기대하는 상태 플래그

    def run(self):
        try:
            # timeout을 적절히 설정합니다. 너무 짧으면 불완전한 데이터만 읽을 수 있습니다.
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1) 
            print(f"ReceiverModule: {self.port} 포트가 {self.baudrate} 보레이트로 열렸습니다.")
        except serial.SerialException as e:
            print(f"ReceiverModule: 시리얼 포트 열기 오류: {e}")
            self.running = False # 오류 발생 시 스레드 종료
            return
        except Exception as e: # 예상치 못한 예외 처리
            print(f"ReceiverModule: 시리얼 포트 초기화 중 오류 발생: {e}")
            self.running = False
            return  

        while self.running:
            try:
                # COM 포트에 데이터가 들어올 때까지 대기
                if self.ser.in_waiting > 0:
                    try:
                        # 한 줄을 읽고 UTF-8로 디코딩 후 양쪽 공백 제거
                        # 디코딩 오류 발생 시 처리 (이전과 동일)
                        # 'errors=' 인자를 추가하여 디코딩 오류 발생 시 처리 방식 명시
                        raw_data = self.ser.readline().decode('utf-8', errors='replace').strip()
                        # print(f"ReceiverModule: 수신된 원본 데이터: '{raw_data}'") 
                    except UnicodeDecodeError as ude: # readline() 자체에서 발생할 수 있는 디코딩 오류
                        print(f"ReceiverModule: 디코딩 오류 발생 (UnicodeDecodeError): {ude}. 데이터: {self.ser.read_all().hex()}")
                        time.sleep(0.01) # 짧게 대기 후 다음 시도
                        continue # 오류 발생 데이터는 버리고 다음 데이터 시도
                    
                    # ------------------------------------------------------------------
                    # 데이터 파싱 로직 (주요 변경점)
                    # ------------------------------------------------------------------
                    if raw_data == 'MOD':
                        # 'MOD'를 받으면 새 패키지 시작으로 간주하고 버퍼 초기화
                        self.current_values_buffer = []
                        self.expecting_data = True
                        # print("ReceiverModule: 'MOD' 수신. 새 데이터 패키지 시작.")
                    elif self.expecting_data:
                        # 'MOD'를 받은 상태에서 숫자 데이터를 기대
                        try:
                            value = float(raw_data) # 수신된 문자열을 실수로 변환
                            self.current_values_buffer.append(value)
                            # print(f"ReceiverModule: 값 수신 - 현재 버퍼 크기: {len(self.current_values_buffer)}")

                            if len(self.current_values_buffer) == 67:
                                # 센서 주변 온도값 1 + 64개값 + 불명 2개값
                                # 64개의 값이 모두 모이면 패키지 완성
                                sensor_degree = self.current_values_buffer[0]
                                received_values = self.current_values_buffer[1:65] # 현재 버퍼 복사
                                etc_values = self.current_values_buffer[65:] 
                                self.current_values_buffer = [] # 버퍼 초기화 (다음 MOD를 위해)
                                self.expecting_data = False     # 다음 'MOD'를 기다림
                                
                                # 완성된 패키지를 큐에 전달
                                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                data_package = {'sensor_degree': sensor_degree, 'time': current_time, 'values': received_values, 'etc': etc_values}
                                # self.output_queue.put(data_package)
                                # 디텍터로 보낸다음 디턱터에서 다시 보냄.
                                self.detection_queue.put(data_package)
                                # print(f"ReceiverModule: 64개 값 패키지 완성 및 큐에 추가. 첫 5개 값: {received_values[:5]}")
                                
                                cwd = os.getcwd().replace('\\', '/')
                                PATH = f"{cwd}/_data/degree_test.csv"
                                # print('here')
                                if not os.path.exists(PATH):
                                    write_type = 'w'
                                else:
                                    write_type = 'a'
                                
                                with open(PATH, write_type , newline='') as csvfile:
                                        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                        result = [datetime.now(), sensor_degree]
                                        result.extend(received_values)
                                        # print(result)
                                        csv_writer.writerow(result)
                                        csv_writer.writerow([datetime.now()])
                                        csv_writer.writerow([sensor_degree])
                                        for row in received_values:
                                            csv_writer.writerow([row])
                                    
                                    
                        except ValueError:
                            # 숫자로 변환할 수 없는 데이터가 들어오면 무시하고 버퍼 리셋
                            print(f"ReceiverModule: 경고: 'MOD' 후 숫자가 아닌 값 수신: '{raw_data}'. 버퍼 초기화.")
                            self.current_values_buffer = []
                            self.expecting_data = False
                    else:
                        # 'MOD'를 기다리는 상태에서 예상치 못한 데이터가 들어오면 무시
                        # print(f"ReceiverModule: 경고: 'MOD' 대기 중 예상치 못한 데이터 수신: '{raw_data}'")
                        pass # 불필요한 로그는 숨길 수 있습니다.
                else:
                    # 데이터가 들어오지 않으면 잠시 대기 (CPU 과부하 방지)
                    time.sleep(0.01)

            except serial.SerialTimeoutException:
                # print("ReceiverModule: 시리얼 읽기 타임아웃.") # 너무 자주 출력될 수 있으니 주석 처리
                pass # 타임아웃은 흔한 일이라 따로 처리하지 않을 수 있습니다.
            except serial.SerialException as e:
                print(f"ReceiverModule: 시리얼 통신 오류 발생: {e}")
                self.running = False # 치명적인 통신 오류 시 스레드 종료
            except Exception as e:
                # 어떤 종류의 오류인지 정확히 알 수 있도록 타입도 함께 출력
                print(f"ReceiverModule에서 예상치 못한 오류 발생: {e} (타입: {type(e).__name__})")
                self.running = False # 다른 일반적인 오류 시 스레드 종료

        # 스레드 종료 시 시리얼 포트 닫기
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"ReceiverModule: {self.port} 포트가 닫혔습니다.")

    def stop(self):
        print(f"ReceiverModule 종료 요청됨.")
        self.running = False