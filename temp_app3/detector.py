import os
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

""" 25/08/04 리팩토링 사항:
- GUI의 열원 후처리 로직을 DetectionModule로 이동
- 온도 필터링, 가중치 적용, 보간 처리를 detector에서 수행
- 화재/연기 감지 로직 통합
""" 
""" 250805 
추가 수정사항:
    - 중앙 센서와 별개로, 모서리 센서의 경우 별도의 보정값 필요함.
    - 1~8 단계로 센서값을 부여할 필요
    - 5.5 열 까지는 적당히 측정되지만, 이후부터는 아슬아슬하게 감지됨
    - 7,7열의 경우 높이 170cm 쯤에서 활성화됨.
    - 센서의 정반대, 가장 먼 곳의 3칸은 측정되지 않음.
    
    화재 감지 알고리즘 추가사항
    - 확산되지 않는 화재 -> 일단 열외
    - deque 크기 확장. 1초당 3~4개 이므로 10초정도 고려?
    - 열원 숫자에 따라 감지 레벨 변화
        - 화재 감지 이전에
        - 주의 / 경고 / 감지 3단계로 나누기
        - 열원 발생시 주의
            - 1deque로 판단 후 안전 처리
        - 열원 추가 발생시 경고
            - 2deque로 판단 후 안전 처리
        - 감지는 실제 화재의 형상으로 열원이 계속해서 증가할 경우.
        
    열원의 지속적인 관리 필요 
    -> safety 변수에 개수 뿐만이 아니라 열원 좌표도 추가하여 관리, dict로?
        안전 열원이라고 판단시 리스트에서 제외
        이동하여 새로운 열원이 될 경우 다시 감시 시작.
        
    다만 총합 열원 크기가 4개 이상일 경우 화재 경고
"""
""" 250807 필터 온도 + 33도 기준 설정

"""

class DetectionModule(threading.Thread):
    def __init__(self, data_signal_obj, detection_queue, output_queue, threshold=5.0, filename="./temp_app/_data/detected_values.txt"):
        super().__init__()
        self.output_queue = output_queue
        self.detection_queue = detection_queue
        self.threshold = threshold
        self.filename = filename
        self.running = True
        self.grid_size = 8
        self.data_signal = data_signal_obj
        
        self.sensor_position = 'corner'
        
        
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
        self.weight_list_corner = [5, 5, 5, 5, 5, 5, 5, 5]
        self.interpolated_grid_size = 8  # 기본 보간 그리드 크기
        
        # === 화재/연기 감지 관련 변수 ===
        self.fire_detected = False
        self.smoke_detected = False
        self.fire_threshold_temp = 35.0  # 화재 감지 임계온도
        self.smoke_threshold_temp = 30.0  # 연기 감지 임계온도
        self.detection_count_threshold = 3  # 연속 감지 횟수 임계값
        
        self.suspect_heatsource_coordinate = tuple()
        self.suspect_fire_coordinate = tuple()
        
        self.total_log = ''
        # 5초간의 데이터 버퍼로 실제 화재인지 아닌지 판별
        # 추가적인 열원이 생겼을 때, 버퍼 저장 시작. 
        # 추가적인 열원이 계속 유지될 경우 화재로 간주.
        self.last_high_temp_counter = 0 
        self.high_temp_counter = 0
        self.safety_status = ''
        
        # 1초당 3~4개. 약 10초인 30개로 설정
        self.deque_size = 30
        self.suspect_heatsource_buffer = deque(maxlen=self.deque_size)
        self.suspect_fire_buffer = deque(maxlen=self.deque_size)
        self.fire_detection_buffer = deque(maxlen=self.deque_size)
        self.fire_detection_buffer2 = deque(maxlen=self.deque_size*2)
        self.buffer_counter = 0
        # self.high_temp_counter_init = 0
        
        data_signal_obj.parameter_update_signal.connect(self.update_detection_parameters)
        
        # 모델 로드 시도
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        # self.model = {}
        # self.model_path = {'rfm':f'{current_dir}/_model/smv_rfm_model.joblib', 'lgbm': ''}
        # self.load_model()
        # self.encoder = joblib.load(f'{current_dir}/_model/smv_label_en.joblib')
        # self.scaler = joblib.load(f'{current_dir}/_model/smv_mmx_sc.joblib')
        
    
    def load_model(self):
        """머신러닝 모델 로드"""
        self.is_model_loaded = True
        return
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
        return False
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
        if 'position' in params:
            self.sensor_position = params['sensor_position']
        if 'min_temp' in params:
            self.min_temp = params['min_temp']
        if 'max_temp' in params:
            self.max_temp = params['max_temp']
        if 'filter_temp_add' in params:
            self.filter_temp_add = params['filter_temp_add']
        if 'weight_list' in params:
            self.weight_list = params['weight_list']
        if 'weight_list_corner' in params:
            self.weight_list_corner = params['weight_list_corner']
        if 'interpolated_grid_size' in params:
            self.interpolated_grid_size = params['interpolated_grid_size']
    
    def apply_heat_source_filtering_center(self, values):
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
    
    def apply_heat_source_filtering_corner(self, values):
        """
        열원 필터링 및 가중치 적용 (GUI에서 이동)
        코너 설치 상태이므로 대각선으로 넓어지면서 8개의 가중치 부여.
        """
        values = np.array(values, np.float32).reshape(8, 8)
        filtered_values = values.copy()
        
        # 평균 온도 계산 및 필터 온도 설정 (기존 로직과 동일)
        self.avg_temp = np.mean(values)
        self.filter_temp = self.avg_temp + self.filter_temp_add
        
        for i in range(8): # 행 (row)
            for j in range(8): # 열 (column)
                current_temp = values[i][j]
                
                if current_temp >= self.filter_temp:
                    # 우측 상단 (0, 7)으로부터의 맨해튼 거리(대각선 거리) 계산
                    # distance = (i - 0) + (7 - j)
                    distance = i + (7 - j)
                    
                    # 거리에 따라 8개의 가중치를 부여하는 로직
                    # 거리가 멀어질수록 가중치 인덱스도 커집니다.
                    if distance == 0:
                        filtered_values[i][j] = current_temp * self.weight_list_corner[0]
                    elif distance == 1:
                        filtered_values[i][j] = current_temp * self.weight_list_corner[1]
                    elif distance == 2:
                        filtered_values[i][j] = current_temp * self.weight_list_corner[2]
                    elif distance == 3:
                        filtered_values[i][j] = current_temp * self.weight_list_corner[3]
                    elif distance == 4:
                        filtered_values[i][j] = current_temp * self.weight_list_corner[4]
                    elif distance == 5:
                        filtered_values[i][j] = current_temp * self.weight_list_corner[5]
                    elif distance == 6:
                        filtered_values[i][j] = current_temp * self.weight_list_corner[6]
                    else: # distance가 7 이상일 때 (가장 먼 구역)
                        filtered_values[i][j] = current_temp * self.weight_list_corner[7]
        
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
    
    def detect_suspect_fire(self, values):
        """ 
        의심 열원 감지 로직
        감지 영역 경계선에 있는 경우 열 감지량이 나누어져 열원으로 인식되지 않음
        이 점을 보완하기 위해서 80% 확률로 평균온도보다 2도이상을 유지한다면
        * 추가로 필터온도 or 33도를 넘어설 경우 의심열원에서 배제. -> 화재 감지 함수에서 잡아낼 수 있음.
        의심 열원으로 인지하고 별도의 카운터로 기록
        기본적으로는 2칸 인식. 1칸인데 기준 온도 미달은 화재가 아니라고 생각함.
        
        추가 개선점?
        가장 외곽에서 발생시 1칸으로도 인식해야하나?
        
        """
        # if np.sum((values < min(self.filter_temp, 33) and (values > self.avg_temp + 2))) > 2:
        # suspect_fire_coordinate = np.where(values < min(self.filter_temp, 33))
        
        suspect_fire_coordinate = np.where((values < min(self.filter_temp, 33)) & (values > self.avg_temp + 2))
        # print(suspect_fire_coordinate)
        self.suspect_fire_buffer.append(suspect_fire_coordinate)
        
        # print(self.suspect_fire_buffer)
        if len(self.suspect_fire_buffer) == self.deque_size:
                
            flattened_list = [item for t in self.suspect_fire_buffer for item in t[0]]
            value_counts = pd.Series(flattened_list).value_counts()
            # print(value_counts)
            # 80% 이상의 튜플에 등장하는 값을 찾습니다.
            threshold = len(self.suspect_fire_buffer) * 0.6
            suspect_fire_coordinate60 = value_counts[value_counts >= threshold].index.tolist()

            # print(f"전체 튜플 개수: {len(self.suspect_fire_buffer)}")
            # print(f"60% 이상(기준: {threshold}회) 포함된 숫자: {suspect_fire_coordinate60}")

            return suspect_fire_coordinate60

        
        
    
    
    def detect_fire_and_smoke(self, values):
        """
        화재 및 연기 감지 로직
            
        화재 감지 알고리즘 추가사항
        - 확산되지 않는 화재 -> 일단 열외
        - deque 크기 확장. 1초당 3~4개 이므로 10초인 30
        - 열원 숫자에 따라 감지 레벨 변화
            - 열원감지에 더 보수적으로. 지속적으로 관측되는 열원에 관해서만 감소/유지/확산 관측
                - is_heatsource()를 통해서 지속적으로 관측되는 열원의 좌표 얻음 
            - is_heatsource()를 통해서 확정된 열원들만 열원카운팅함.
                
            - 주의 / 경고 / 감지 3단계로 나누기
            - 열원 발생시 주의
                - 1deque로 판단 후 안전 처리
            - 열원 추가 발생시 경고
                - 2deque로 판단 후 안전 처리
            - 감지는 실제 화재의 형상으로 열원이 계속해서 증가할 경우.
            
        열원의 지속적인 관리 필요 
        -> safety 변수에 개수 뿐만이 아니라 열원 좌표도 추가하여 관리, dict로?
            안전 열원이라고 판단시 리스트에서 제외
            이동하여 새로운 열원이 될 경우 다시 감시 시작.
            
        다만 총합 열원 크기가 4개 이상일 경우 화재 경고
        
        """
        fire_detected = False
        smoke_detected = False
        confirmed_heatsource_coordinate = self.is_heatsource(values)
        if len(confirmed_heatsource_coordinate) > 0 :
            print('confirmed_heatsource_coordinate', confirmed_heatsource_coordinate)
        self.high_temp_counter = len(confirmed_heatsource_coordinate)
        
        # self.high_temp_counter = np.sum(values > min(self.filter_temp, 33))
        # print( 'self.last_high_temp_counter', self.last_high_temp_counter)
        # print( 'self.high_temp_counter', self.high_temp_counter)
        
        #   열원 존재 조건                  버퍼가 끝났다는 조건                기존 열원에 변화가 있다는 조건
        if (self.high_temp_counter > 0 and self.buffer_counter == 0) and self.high_temp_counter != self.last_high_temp_counter: 
            print('열원 개수 변화 감지', self.last_high_temp_counter, self.high_temp_counter)
            self.buffer_counter = self.deque_size
        
        # 위의 3개 조건 충족시 버퍼카운터 리필.
            
            # if self.high_temp_counter != len(self.safety_high_temp_dict):
                
            #     # 이전과 현재가 다름으로 조건 들어옴. 추가조건이 필요한가? 열원이 늘었따는 결과만남음.
            #     # values를 확정열원으로 필터링하여 저장
            #     self.high_temp_counter_init = np.sum(values[confirmed_heatsource_coordinate] > min(self.filter_temp, 33))
            #     self.high_temp_counter_init = np.sum(values[confirmed_heatsource_coordinate] > min(self.filter_temp, 33))
            #     # print(np.where(values > self.filter_temp))
        
        ## 버퍼카운터가 남아있다면(위의 3가지 조건 충족 시) 화재감지 버퍼에 저장시작.
        ## 모든 데이터를 받는 게 아니라 확정열원의 데이터만 받기.
        if self.buffer_counter > 0 :
            self.fire_detection_buffer.append(values)
            self.buffer_counter -= 1
        
        # 확정 열원 개수를 받고, 버퍼에 데이터를 수집한 뒤 이 열원이 scd중 어디로 가야할지 판별하는 로직
        # 화재 감지 버퍼가 가득 차면 평가 시작
        # if len(self.fire_detection_buffer) == self.deque_size:
            # 감소 / 유지 / 확산을 판별해야함
            # 추가되는 열원도 확정열원이어야함.
            # 유지는 safety로, 확산이면 caution으로 보내야함
            
            
        
        # print('현재/이전', self.high_temp_counter, self.last_high_temp_counter)
        if len(self.fire_detection_buffer) == self.deque_size:
            print(f'full buffer, proved {len(self.fire_detection_buffer)} == {self.deque_size}')

            is_fire = self.is_fire(self.fire_detection_buffer, confirmed_heatsource_coordinate)
            if is_fire :
                print('caution 진입', self.high_temp_counter)
                self.safety_status = 'caution'
            else:
                print("safety 진입")
                self.safety_status = 'safety'
            self.fire_detection_buffer.clear()

        
        # 주의 단계 2배 데크로 테스트
        if self.safety_status == 'caution':
            self.fire_detection_buffer2.append(values)
            # print(self.fire_detection_buffer2)
            # 버퍼2가 가득 찬 경우 and 카운터가 바뀌었을때.  => 열원이 늘었을 때만으로 한정.
        if len(self.fire_detection_buffer2) == self.deque_size*2 and self.safety_status == 'caution': # and self.high_temp_counter > len(self.safety_high_temp_dict):
            is_fire = self.is_fire(self.fire_detection_buffer2, confirmed_heatsource_coordinate)
            if is_fire :
                print("danger 진입")
                fire_detected = is_fire
                self.safety_status = 'danger'
            else:
                print("safety 진입")
                self.safety_status = 'safety'
            self.fire_detection_buffer2.clear()
        

        return fire_detected, smoke_detected
    
    
    def is_heatsource(self, values):
        """ 
        감지한 열이 지속적으로 관측되는 열원인지 아닌지를 판별.
        deque를 받아서 열이 관측된 셀이 얼마나 지속되는지 확인
        
        """
        
        #  온도 조건에 만족하는 좌표 수집
        suspect_heatsource_coordinate = np.where((values > min(self.filter_temp, 33)))
        # print('suspect_heatsource_coordinate', suspect_heatsource_coordinate)
        self.suspect_heatsource_buffer.append(suspect_heatsource_coordinate)
        
        # print(self.suspect_heatsource_buffer)
        # 1차 검증 크기만큼 데이터가 수집됬을 경우. 30이면 10초
        if len(self.suspect_heatsource_buffer) == self.deque_size:
                
            flattened_list = [item for t in self.suspect_heatsource_buffer for item in t[0]]
            value_counts = pd.Series(flattened_list).value_counts()
            # print(value_counts)
            # 모든 튜플에 100%로 등장하는 값을 찾습니다.
            threshold = len(self.suspect_heatsource_buffer) * 1.0
            # print('suspect_heatsource_buffer', len(self.suspect_heatsource_buffer))
            confirmed_heatsource_coordinate = value_counts[value_counts >= threshold].index.tolist()
            
            # print(f"전체 튜플 개수: {len(self.suspect_heatsource_buffer)}")
            # print(f"100% 이상(기준: {threshold}회) 포함된 숫자: {confirmed_heatsource_coordinate}")

            return confirmed_heatsource_coordinate
        else:
            return tuple()
        
        
        
    def is_fire(self, deque, confirmed_heatsource_coordinate):
        """ 
        deque를 받아 안의 데이터로 열원이 확산되었는지, 아닌지를 판별함.
        기본 deque와 확장 deque간 다른 메커니즘 사용 
        열원이 heat_source_limit 개 이상이면 무조건.
        """
        # print(len(deque))
        # print(deque)
        heat_source_limit = 15
        
        if np.sum(deque[-1] > min(self.filter_temp, 33)) > heat_source_limit:
            print(f'heat source over {heat_source_limit}')
            return True
        frontlist = list()
        backlist = list()
        if deque:
            for i in range((deque.maxlen//2)):
                if len(deque) <= 1: break
                # # print('f',len(deque), len(frontlist))
                # frontlist.append(np.sum(deque.popleft() > min(self.filter_temp, 33)))
                # # print('b',len(deque), len(backlist))
                # backlist.append(np.sum(deque.pop() > min(self.filter_temp, 33)))
                # 안전열원으로 필터링 후 값이 존재하는 좌표의 개수만 전/후 리스트에 추가
                frontlist.append(np.sum(deque.popleft()[confirmed_heatsource_coordinate] > 0))
                backlist.append(np.sum(deque.pop()[confirmed_heatsource_coordinate] > 0))
                
            # print('f',len(deque), (frontlist))
            # print('b',len(deque), (backlist))                
            front_avg = sum(frontlist) / (deque.maxlen//2)
            back_avg = sum(backlist) / (deque.maxlen//2)
            print('fb', front_avg, back_avg)
            if front_avg >= back_avg : 
                return False
            else:
                return True
    
    
    
    
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
            'smoke_detected': self.smoke_detected
            # 'high_temp_count': np.sum(original_values > self.filter_temp),
            # 'anomaly_regions': self.identify_anomaly_regions(original_values)
        }
    
    def run(self):
        while self.running:
            if not self.detection_queue.empty():
                self.last_high_temp_counter = self.high_temp_counter
                data_package = self.detection_queue.get()
                
                # data_package 유효성 확인
                if 'values' in data_package and 'time' in data_package:
                    current_time = data_package['time']
                    values = data_package['values']

                    if len(values) == 64:
                       
                        # 2. 머신러닝 객체 감지
                        object_detection = []
                        if self.is_model_loaded and 'rfm' in self.model:
                            object_detection = self.model_predict(data_package, 'rfm')
                        
                        # 3. 열원 후처리 수행
                        if self.sensor_position == 'corner':
                            filtered_values = self.apply_heat_source_filtering_corner(values)
                        else:
                            filtered_values = self.apply_heat_source_filtering_center(values)
                        interpolated_values = self.interpolate_temperature_data(filtered_values)
                        
                        # 4. 화재/연기 감지
                        self.fire_detected, self.smoke_detected = self.detect_fire_and_smoke(np.array(values))
                        # 의심 열원 감지
                        # if np.sum((values < min(self.filter_temp, 33) and (values > self.avg_temp + 2))) > 2:
                        self.suspect_fire_coordinate = self.detect_suspect_fire(np.array(values))
                        
                        # 5. 감지 통계 계산
                        detection_stats = self.calculate_detection_stats(values, interpolated_values)
                        
                        # 이상 기록 저장
        
                        if self.safety_status in ['danger', 'caution']:
                            self.log_anomaly(current_time, values)
                        else:
                            self.total_log = ''
                        
                        # 6. 처리된 데이터 패키지 구성
                        data_package.update({
                            'object_detection': object_detection,
                            'processed_values': interpolated_values.flatten().tolist(),
                            'interpolated_grid_size': self.interpolated_grid_size,
                            'detection_stats': detection_stats,
                            'fire_detected': self.fire_detected,
                            'smoke_detected': self.smoke_detected,
                            'anomaly_count': self.anomaly_total_count,
                            'safety_status': self.safety_status,
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

    def log_anomaly(self, current_time, values):
        """기존 이상 감지 처리 (파일 로깅)"""
        values_array = np.array(values)
        average = np.mean(values_array)
        max = np.max(values_array)

        if self.safety_status == 'danger':
            strength = 'danger'
            # print(strength)
        else:
            strength = 'caution'
            # print(2, strength)
        # 파일에 누적 횟수도 함께 기록
        self.total_log = f"{current_time[2:]} avg: {average:.2f}°C max: {max:.2f}°C 강도:{strength}\n" 
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(self.total_log)
            
            
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
            'processing_params': data_package.get('processing_params', {}),
            'suspect_fire': self.suspect_fire_coordinate,
            'safety_status': self.safety_status,
            'total_log': self.total_log
        }
        
        # GUI 큐로 전달
        self.output_queue.put(gui_data_package)
        
    def stop(self):
        self.running = False