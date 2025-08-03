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

class DetectionModule(threading.Thread):
    def __init__(self, detection_queue, output_queue, base_temperature=25.0, threshold=5.0, filename="detected_values.txt"):
        super().__init__()
        self.output_queue = output_queue
        self.detection_queue = detection_queue
        self.base_temperature = base_temperature
        self.threshold = threshold
        self.filename = filename
        self.running = True
        self.grid_size = 8
        # === [수정됨] 누적 이상 감지 횟수를 저장할 변수 추가 ===
        self.anomaly_total_count = 0
        self.data_buffer = deque(maxlen=20)  # 최대 20프레임 보관
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
                self.model[i] = joblib.load(self.model_path[i])
                
                
                self.is_model_loaded = True
                print(f"{i} Model loaded successfully from {self.model_path[i]}")
        except FileNotFoundError:
            print(f"{i} Model file not found at {self.model_path[i]}")
            self.is_model_loaded = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_model_loaded = False
    
    def model_predict(self, data_package, model_name):
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
        # print('오브젝트 감지',object_detect)
        return object_detect
    
    def run(self):
        while self.running:
            if not self.detection_queue.empty():
                data_package = self.detection_queue.get()
                # data_package에 'values'와 'time' 키가 있는지 확인
                if 'values' in data_package and 'time' in data_package:
                    current_time = data_package['time']
                    values = data_package['values']

                    if len(values) == 64:
                        self.process_values(current_time, values)
                        object_detection = self.model_predict(data_package, 'rfm')
                        data_package['object_detection'] = object_detection
                        self.process_data_package(data_package, object_detection)
                    else:
                        print(f"DetectionModule: Expected 64 values, got {len(values)}. Skipping processing.")
                else:
                    print(f"DetectionModule: Received invalid data_package. Keys 'values' or 'time' missing.")
            else:
                time.sleep(0.1) # 큐가 비어있으면 잠시 대기

    def process_values(self, current_time, values):
        return
        values_array = np.array(values)
        average = np.mean(values_array)
        detection_limit = self.base_temperature + self.threshold
        
        detected_high_values = []
        for i, val in enumerate(values_array):
            if val > detection_limit:
                detected_high_values.append(f"Index {i}: {val:.2f} (감지 기준 {detection_limit:.1f}°C 초과)")
        
        if detected_high_values:
            # === [수정됨] 이상 감지 시 누적 횟수 1 증가 ===
            self.anomaly_total_count += 1
            
            # === [수정됨] 파일에 누적 횟수도 함께 기록 ===
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(f"--- Detected at {current_time} ---\n")
                f.write(f"누적 이상 감지 횟수: {self.anomaly_total_count}회\n") # 누적 횟수 기록
                f.write(f"Overall Average: {average:.2f}°C\n")
                for item in detected_high_values:
                    f.write(f"- {item}\n")
                f.write("\n")
                
            # 콘솔 출력에도 누적 횟수 표시
            print(f"Detected high values at {current_time}. Total detections: {self.anomaly_total_count}. Saved to {self.filename}")
        else:
            # 정상 상태일 때의 메시지는 너무 자주 출력될 수 있으므로 주석 처리하거나 필요시 사용
            print(f"No values detected above threshold at {current_time}.")
            pass
        
    def process_data_package(self, data_package, object_detection):
        """
        Receiver로부터 받은 데이터 패키지를 처리하고 GUI로 전달
        """
        current_time = data_package['time']
        values = data_package['values']
        sensor_degree = data_package.get('sensor_degree', 0.0)
        etc = data_package.get('etc', [])

        if len(values) != 64:
            print(f"DetectionModule: Expected 64 values, got {len(values)}. Skipping processing.")
            return

        # 8x8 배열로 변환
        values_array = np.array(values).reshape((self.grid_size, self.grid_size))
        
        # 데이터 버퍼에 추가
        self.data_buffer.append({
            'time': current_time,
            'values_array': values_array.copy(),
            'timestamp': datetime.now()
        })
                
        # GUI로 전달할 데이터 패키지 구성
        gui_data_package = {
            'time': current_time,
            'values': values,  # 원본 64개 값 (8x8)
            'sensor_degree': sensor_degree,
            'etc': etc,
            'anomaly_count': self.anomaly_total_count,
            'detection_stats': {
                'fire_detected' : False,
                'smoke_detected' : False,
                'max_temp': ['max_temp'],
                'avg_temp': ['avg_temp'],
                'object_detect' : object_detection,
            },
        }
        
        # GUI 큐로 전달
        self.output_queue.put(gui_data_package)
        
    def stop(self):
        self.running = False