import numpy as np
from datetime import datetime
import threading
import time
import queue
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label, center_of_mass
import cv2

class DetectionModule(threading.Thread):
    def __init__(self, input_queue, output_queue, base_temperature=25.0, threshold=2.0, filename="detected_values.txt"):
        super().__init__()
        self.input_queue = input_queue    # Receiver로부터 데이터를 받는 큐
        self.output_queue = output_queue  # GUI로 데이터를 전달하는 큐
        self.base_temperature = base_temperature
        self.threshold = threshold
        self.filename = filename
        self.running = True
        
        # 누적 이상 감지 횟수
        self.anomaly_total_count = 0
        
        # 가우시안 필터 설정
        self.gaussian_sigma = 0.8
        self.grid_size = 8  # 원본 8x8 그리드
        
        # 열원 감지 설정
        self.min_hotspot_size = 1  # 최소 열원 크기 (픽셀)
        self.max_hotspot_size = 16  # 최대 열원 크기 (픽셀)

    def run(self):
        while self.running:
            try:
                # input_queue에서 데이터 받기 (타임아웃 설정)
                data_package = self.input_queue.get(timeout=0.1)
                
                if 'values' in data_package and 'time' in data_package:
                    self.process_data_package(data_package)
                else:
                    print(f"DetectionModule: Invalid data_package received. Missing 'values' or 'time'.")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"DetectionModule: Error processing data: {e}")

    def process_data_package(self, data_package):
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
        
        # 이상 감지 수행
        detection_result = self.detect_anomalies(values_array, current_time)
        
        # 열원 감지 수행
        hotspot_data = self.detect_hotspots(values_array)
        
        # GUI로 전달할 데이터 패키지 구성
        gui_data_package = {
            'time': current_time,
            'values': values,  # 원본 64개 값 (8x8)
            'sensor_degree': sensor_degree,
            'etc': etc,
            'fire_detected': detection_result['fire_detected'],
            'smoke_detected': detection_result['smoke_detected'],
            'anomaly_count': self.anomaly_total_count,
            'hotspots': hotspot_data,  # 감지된 열원 정보
            'detection_stats': {
                'max_temp': detection_result['max_temp'],
                'avg_temp': detection_result['avg_temp'],
                'detection_limit': detection_result['detection_limit']
            }
        }
        
        # GUI 큐로 전달
        self.output_queue.put(gui_data_package)

    def detect_anomalies(self, values_array, current_time):
        """
        온도 이상 감지 로직
        """
        # 가우시안 필터 적용 (노이즈 감소)
        filtered_array = gaussian_filter(values_array, sigma=self.gaussian_sigma)
        
        # 통계 계산
        average = np.mean(filtered_array)
        max_temp = np.max(filtered_array)
        detection_limit = self.base_temperature + self.threshold
        
        # 이상 온도 감지
        anomaly_mask = filtered_array > detection_limit
        detected_indices = np.where(anomaly_mask)
        
        fire_detected = False
        smoke_detected = False
        
        if len(detected_indices[0]) > 0:
            # 이상 감지 발생
            self.anomaly_total_count += 1
            
            # 화재/연기 감지 로직 (온도 기준)
            if max_temp > self.base_temperature + self.threshold * 2:  # 더 높은 온도는 화재
                fire_detected = True
            elif max_temp > detection_limit:  # 임계값 초과는 연기
                smoke_detected = True
            
            # 로그 파일에 기록
            self.log_detection(current_time, filtered_array, detected_indices, average, detection_limit)
            
            print(f"DetectionModule: Anomaly detected at {current_time}. Total: {self.anomaly_total_count}")

        return {
            'fire_detected': fire_detected,
            'smoke_detected': smoke_detected,
            'max_temp': max_temp,
            'avg_temp': average,
            'detection_limit': detection_limit,
            'anomaly_indices': detected_indices
        }

    def detect_hotspots(self, values_array):
        """
        열원의 위치와 형태를 감지
        """
        # 가우시안 필터 적용
        filtered_array = gaussian_filter(values_array, sigma=self.gaussian_sigma)
        
        # 임계값을 넘는 영역 찾기
        threshold_temp = self.base_temperature + self.threshold * 0.7  # 좀 더 낮은 임계값 사용
        binary_mask = filtered_array > threshold_temp
        
        # 연결된 구성요소 찾기 (열원 클러스터링)
        labeled_array, num_features = label(binary_mask)
        
        hotspots = []
        
        for i in range(1, num_features + 1):
            # 각 열원 영역 분석
            hotspot_mask = (labeled_array == i)
            hotspot_size = np.sum(hotspot_mask)
            
            # 크기 필터링
            if self.min_hotspot_size <= hotspot_size <= self.max_hotspot_size:
                # 중심점 계산
                center_y, center_x = center_of_mass(hotspot_mask)
                
                # 해당 영역의 최고 온도
                max_temp_in_hotspot = np.max(filtered_array[hotspot_mask])
                avg_temp_in_hotspot = np.mean(filtered_array[hotspot_mask])
                
                # 열원 경계 찾기
                hotspot_coords = np.where(hotspot_mask)
                min_row, max_row = np.min(hotspot_coords[0]), np.max(hotspot_coords[0])
                min_col, max_col = np.min(hotspot_coords[1]), np.max(hotspot_coords[1])
                
                hotspot_info = {
                    'id': i,
                    'center': (float(center_x), float(center_y)),  # (x, y) 좌표
                    'size': int(hotspot_size),
                    'max_temp': float(max_temp_in_hotspot),
                    'avg_temp': float(avg_temp_in_hotspot),
                    'bbox': {  # 바운딩 박스
                        'min_row': int(min_row),
                        'max_row': int(max_row),
                        'min_col': int(min_col),
                        'max_col': int(max_col)
                    },
                    'coordinates': [(int(x), int(y)) for x, y in zip(hotspot_coords[1], hotspot_coords[0])]  # 실제 픽셀 좌표들
                }
                
                hotspots.append(hotspot_info)
        
        return hotspots

    def log_detection(self, current_time, filtered_array, detected_indices, average, detection_limit):
        """
        이상 감지 결과를 파일에 로그
        """
        detected_high_values = []
        for i, j in zip(detected_indices[0], detected_indices[1]):
            temp = filtered_array[i, j]
            detected_high_values.append(f"Position ({i},{j}): {temp:.2f}°C (기준 {detection_limit:.1f}°C 초과)")
        
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(f"--- Detected at {current_time} ---\n")
            f.write(f"누적 이상 감지 횟수: {self.anomaly_total_count}회\n")
            f.write(f"Overall Average: {average:.2f}°C\n")
            f.write(f"Detection Limit: {detection_limit:.1f}°C\n")
            for item in detected_high_values:
                f.write(f"- {item}\n")
            f.write("\n")

    def stop(self):
        """
        스레드 안전하게 종료
        """
        self.running = False
        self.join()  # 스레드 종료 대기