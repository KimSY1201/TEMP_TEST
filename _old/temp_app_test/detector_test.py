import numpy as np
import pandas as pd
from datetime import datetime
import queue
from collections import deque
import threading
import time
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler
# from gui import OutputModule

""" 
25/08/04
수정점
안전 상태와 위험 상태를 구분하기.
안전 상태 : 1.열원이 없거나
            2. 열원이 있더라도 변화가 없거나.
            3. 열원이 감소

위험 상태 : 안전 상태에서 
            경계: 열원 추가 발생 (버퍼 사용 시작)
            10초간 열원 변화 추적 및 판단
            화재 인식: 열원이 추가적으로 더 생길 시(확산 시)
            * 일단 더 생기는 경우 무조건 화재로 간주.
              더 생겼지만 화재가 아닌 경우는 추후 피드백으로 보완하기

리팩토링 사항:
- GUI의 열원 후처리 로직을 DetectionModule로 이동
- 온도 필터링, 가중치 적용, 보간 처리를 detector에서 수행
- 화재/연기 감지 로직 통합
"""

class DetectionModule(threading.Thread):
    def __init__(self, data_signal_obj, detection_queue, output_queue, base_temperature=25.0, threshold=5.0, filename="detected_values.txt"):
        super().__init__()
        self.output_queue = output_queue
        self.detection_queue = detection_queue
        self.base_temperature = base_temperature
        self.threshold = threshold
        self.filename = filename
        self.running = True
        self.grid_size = 8
        self.data_signal = data_signal_obj
        
        # === 누적 이상 감지 횟수를 저장할 변수 ===
        self.anomaly_total_count = 0
        self.data_buffer = deque(maxlen=20)  # 최대 20프레임 보관
        
        # === 열원 처리 관련 파라미터 (GUI에서 이동) ===
        self.min_temp = 19.0
        self.max_temp = 32.0
        self.avg_temp = 0
        self.filter_temp_add = 5
        self.filter_temp = 0
        self.weight_list = [3.8, 3.9, 4.0, 5.1]
        self.interpolated_grid_size = 8  # 기본 보간 그리드 크기
        
        # === 화재/연기 감지 관련 변수 ===
        self.fire_detected = False
        self.smoke_detected = False
        self.fire_threshold_temp = 35.0  # 화재 감지 임계온도
        self.smoke_threshold_temp = 30.0  # 연기 감지 임계온도
        self.detection_count_threshold = 3  # 연속 감지 횟수 임계값
        
        # 5초간의 데이터 버퍼로 실제 화재인지 아닌지 판별
        # 추가적인 열원이 생겼을 때, 버퍼 저장 시작. 
        # 추가적인 열원이 계속 유지될 경우 화재로 간주.
        self.high_temp_counter = None
        self.safety_high_temp_counter = None
        self.fire_detection_buffer = deque(maxlen=15)
        self.buffer_counter = 0
        self.high_temp_counter_init = None
        
        # data_signal_obj.parameter_update_signal.connect(self.update_detection_parameters)
        
        # 모델 로드 시도
        self.model = {}
        self.model_path = {'rfm':'./smv_rfm_model.joblib', 'lgbm': ''}
        self.load_model()
        self.encoder = joblib.load('./smv_label_en.joblib')
        self.scaler = joblib.load('./smv_mmx_sc.joblib')
        
    
    def load_model(self):
        """머신러닝 모델 로드"""
        try:
            for i in self.model_path.keys():
                if self.model_path[i]:  # 경로가 있는 경우만 로드
                    self.model[i] = joblib.load(self.model_path[i])
                    print(f"{i} Model loaded successfully from {self.model_path[i]}")
            self.is_model_loaded = True
        except FileNotFoundError as e:
            print(f"Model file not found: {e}")
            self.is_model_loaded = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_model_loaded = False
    
    def model_predict(self, data_package, model_name):
        """머신러닝 모델 예측"""
        temp_dict = {
            'sensor_degree': [data_package['sensor_degree']],
            'mean': [np.mean(data_package['values'])]
        }

        for i in range(len(data_package['values'])):
            key = f'thermal_{i:02d}'
            value = [data_package['values'][i]]
            temp_dict.update({key: value})
                    
        ddf = pd.DataFrame(temp_dict)
        ddf.iloc[:,:] = self.scaler.transform(ddf.iloc[:,:])
        model = self.model[model_name]
        object_detect = self.encoder.inverse_transform(model.predict(ddf.iloc[:,:]))
        return object_detect
    
    def update_detection_parameters(self, params):
        """GUI에서 전달받은 파라미터 업데이트"""
        if 'min_temp' in params:
            self.min_temp = params['min_temp']
        if 'max_temp' in params:
            self.max_temp = params['max_temp']
        if 'filter_temp_add' in params:
            self.filter_temp_add = params['filter_temp_add']
        if 'weight_list' in params:
            self.weight_list = params['weight_list']
        if 'interpolated_grid_size' in params:
            self.interpolated_grid_size = params['interpolated_grid_size']
    
    def apply_heat_source_filtering(self, values):
        """
        열원 필터링 및 가중치 적용 (GUI에서 이동)
        구역별로 다른 가중치를 적용하여 중앙부 열원을 강조
        """
        values = np.array(values, np.float32).reshape(8, 8)
        filtered_values = values.copy()
        
        # 평균 온도 계산 및 필터 온도 설정
        self.avg_temp = np.mean(values)
        self.filter_temp = self.avg_temp + self.filter_temp_add
        
        for i in range(8):  # 행 (row)
            for j in range(8):  # 열 (column)
                current_temp = values[i][j]
                
                if current_temp >= self.filter_temp:
                    # 구역 0: 중앙 2x2 구역 (최고 가중치)
                    if (3 <= i <= 4) and (3 <= j <= 4):
                        filtered_values[i][j] = current_temp * self.weight_list[0]
                    # 구역 1: 중앙 4x4에서 구역 0을 제외한 구역
                    elif (2 <= i <= 5) and (2 <= j <= 5):
                        filtered_values[i][j] = current_temp * self.weight_list[1]
                    # 구역 2: 중앙 6x6에서 구역 1,0을 제외한 구역
                    elif (1 <= i <= 6) and (1 <= j <= 6):
                        filtered_values[i][j] = current_temp * self.weight_list[2]
                    # 구역 3: 가장 바깥쪽 테두리
                    else:
                        filtered_values[i][j] = current_temp * self.weight_list[3]
        
        return filtered_values
    
    def interpolate_temperature_data(self, filtered_values):
        """
        온도 데이터 보간 처리 (GUI에서 이동)
        8x8 데이터를 지정된 크기로 보간
        """
        grid_multiplier = self.interpolated_grid_size // 8
        if grid_multiplier <= 1:
            return filtered_values
        
        # 단순 반복을 통한 보간 (nearest neighbor 방식)
        interpolated_values = filtered_values.repeat(grid_multiplier, axis=0).repeat(grid_multiplier, axis=1)
        return interpolated_values
    
    def detect_fire_and_smoke(self, values):
        """
        화재 및 연기 감지 로직 (GUI에서 이동 및 개선)
        """
        # max_temp = np.max(values)
        # avg_temp = np.mean(values)
        fire_detected = False
        smoke_detected = False
        self.high_temp_counter = np.sum(values > self.filter_temp)
        
        if self.high_temp_counter > 0 and self.buffer_counter == 0: 
            self.buffer_counter = 15
            if self.high_temp_counter != self.safety_high_temp_counter:
                self.high_temp_counter_init = np.sum(values > self.filter_temp)
        
        if self.buffer_counter > 0 :
            self.fire_detection_buffer.append(values)
            self.buffer_counter -= 1
        
               
        # 화재 감지 조건
        
        if len(self.fire_detection_buffer) == 15:
            print('full buffer, proved')
            tested_list = []
            for i in self.fire_detection_buffer[-5:]:
                tested_list.append(np.sum(i > self.filter_temp))
            if (sum(tested_list) / 5) > self.high_temp_counter_init :
                fire_detected = True
            else:
                self.safety_high_temp_counter = self.high_temp_counter_init
                
        
        # 최근 N회 중 임계값 이상 감지되면 최종 감지
        # fire_detected = sum(self.fire_detection_buffer) >= self.detection_count_threshold
        
        # 연기 감지 조건
        # smoke_condition = (
        #     max_temp > self.smoke_threshold_temp and 
        #     medium_temp_count >= 5 and  # 5개 이상의 셀이 연기 임계온도 초과
        #     avg_temp > self.base_temperature + 3  # 평균 온도가 기준보다 3도 이상 높음
        # )
        # self.smoke_detection_buffer.append(smoke_condition)
        # smoke_detected = sum(self.smoke_detection_buffer) >= self.detection_count_threshold
        
        
        return fire_detected, smoke_detected
    
    def calculate_detection_stats(self, original_values, processed_values):
        """
        감지 통계 계산
        """
        return {
            'max_temp': np.max(original_values),
            'avg_temp': np.mean(original_values),
            'processed_max_temp': np.max(processed_values),
            'processed_avg_temp': np.mean(processed_values),
            'fire_detected': self.fire_detected,
            'smoke_detected': self.smoke_detected,
            'high_temp_count': np.sum(original_values > self.filter_temp),
            'anomaly_regions': self.identify_anomaly_regions(original_values)
        }
    
    def identify_anomaly_regions(self, values):
        """
        이상 온도 영역 식별
        """
        values_2d = np.array(values).reshape(8, 8)
        anomaly_positions = []
        
        for i in range(8):
            for j in range(8):
                if values_2d[i, j] > self.fire_threshold_temp:
                    anomaly_positions.append({
                        'position': (i, j),
                        'temperature': values_2d[i, j],
                        'severity': 'high' if values_2d[i, j] > self.fire_threshold_temp + 5 else 'medium'
                    })
        
        return anomaly_positions
    
    def run(self):
        while self.running:
            if not self.detection_queue.empty():
                data_package = self.detection_queue.get()
                
                # data_package 유효성 확인
                if 'values' in data_package and 'time' in data_package:
                    current_time = data_package['time']
                    values = data_package['values']

                    if len(values) == 64:
                        # 1. 원본 온도 처리
                        self.process_values(current_time, values)
                        
                        # 2. 머신러닝 객체 감지
                        object_detection = []
                        if self.is_model_loaded and 'rfm' in self.model:
                            object_detection = self.model_predict(data_package, 'rfm')
                        
                        # 3. 열원 후처리 수행
                        filtered_values = self.apply_heat_source_filtering(values)
                        interpolated_values = self.interpolate_temperature_data(filtered_values)
                        
                        # 4. 화재/연기 감지
                        self.fire_detected, self.smoke_detected = self.detect_fire_and_smoke(np.array(values))
                        
                        # 5. 감지 통계 계산
                        detection_stats = self.calculate_detection_stats(values, interpolated_values)
                        
                        # 6. 처리된 데이터 패키지 구성
                        data_package.update({
                            'object_detection': object_detection,
                            'processed_values': interpolated_values.flatten().tolist(),
                            'interpolated_grid_size': self.interpolated_grid_size,
                            'detection_stats': detection_stats,
                            'fire_detected': self.fire_detected,
                            'smoke_detected': self.smoke_detected,
                            'anomaly_count': self.anomaly_total_count,
                            'processing_params': {
                                'min_temp': self.min_temp,
                                'max_temp': self.max_temp,
                                'filter_temp': self.filter_temp,
                                'avg_temp': self.avg_temp
                            }
                        })
                        
                        # 7. GUI로 전달
                        self.process_data_package(data_package)
                        
                    else:
                        print(f"DetectionModule: Expected 64 values, got {len(values)}. Skipping processing.")
                else:
                    print(f"DetectionModule: Received invalid data_package. Keys 'values' or 'time' missing.")
            else:
                time.sleep(0.1)  # 큐가 비어있으면 잠시 대기

    def process_values(self, current_time, values):
        """기존 이상 감지 처리 (파일 로깅)"""
        values_array = np.array(values)
        average = np.mean(values_array)
        detection_limit = self.base_temperature + self.threshold
        
        detected_high_values = []
        for i, val in enumerate(values_array):
            if val > detection_limit:
                detected_high_values.append(f"Index {i}: {val:.2f} (감지 기준 {detection_limit:.1f}°C 초과)")
        
        if detected_high_values:
            self.anomaly_total_count += 1
            
            # 파일에 누적 횟수도 함께 기록
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(f"--- Detected at {current_time} ---\n")
                f.write(f"누적 이상 감지 횟수: {self.anomaly_total_count}회\n")
                f.write(f"Overall Average: {average:.2f}°C\n")
                for item in detected_high_values:
                    f.write(f"- {item}\n")
                f.write("\n")
                
            print(f"Detected high values at {current_time}. Total detections: {self.anomaly_total_count}. Saved to {self.filename}")
        
    def process_data_package(self, data_package):
        """
        처리된 데이터 패키지를 GUI로 전달
        """
        current_time = data_package['time']
        values = data_package['values']
        sensor_degree = data_package.get('sensor_degree', 0.0)
        etc = data_package.get('etc', [])

        # 데이터 버퍼에 추가
        self.data_buffer.append({
            'time': current_time,
            'values_array': np.array(values).reshape((self.grid_size, self.grid_size)),
            'timestamp': datetime.now()
        })
        
        # GUI로 전달할 최종 데이터 패키지
        gui_data_package = {
            'time': current_time,
            'values': values,  # 원본 64개 값
            'processed_values': data_package.get('processed_values', values),  # 처리된 값들
            'sensor_degree': sensor_degree,
            'etc': etc,
            'anomaly_count': self.anomaly_total_count,
            'object_detection': data_package.get('object_detection', []),
            'fire_detected': data_package.get('fire_detected', False),
            'smoke_detected': data_package.get('smoke_detected', False),
            'detection_stats': data_package.get('detection_stats', {}),
            'interpolated_grid_size': data_package.get('interpolated_grid_size', 8),
            'processing_params': data_package.get('processing_params', {})
        }
        
        # GUI 큐로 전달
        self.output_queue.put(gui_data_package)
        
    def stop(self):
        self.running = False