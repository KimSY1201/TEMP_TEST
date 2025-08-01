import numpy as np
from datetime import datetime
import threading
import time
import queue
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label, center_of_mass
from collections import deque
import cv2

class HotspotTracker:
    """열원 추적을 위한 클래스"""
    
    def __init__(self, max_history=10, similarity_threshold=1.5, persistence_threshold=3):
        self.max_history = max_history  # 최대 보관할 이력 개수
        self.similarity_threshold = similarity_threshold  # 유사성 판단 거리 임계값
        self.persistence_threshold = persistence_threshold  # 열원으로 인정하는 최소 지속 횟수
        self.hotspot_history = deque(maxlen=max_history)  # 열원 이력 버퍼
        self.tracked_hotspots = {}  # 추적 중인 열원들 {id: tracker_info}
        self.next_id = 1  # 다음 할당할 열원 ID
        
    def update(self, current_hotspots):
        """현재 감지된 열원들과 이력을 비교하여 추적 업데이트"""
        # 현재 프레임을 이력에 추가
        self.hotspot_history.append(current_hotspots)
        
        # 기존 추적 중인 열원들과 매칭
        matched_hotspots = []
        unmatched_current = list(current_hotspots)
        
        # 기존 추적 중인 열원들과 현재 열원들 매칭
        for tracker_id, tracker_info in list(self.tracked_hotspots.items()):
            best_match = None
            min_distance = float('inf')
            
            # 현재 열원들 중에서 가장 가까운 것 찾기
            for i, current_hotspot in enumerate(unmatched_current):
                distance = self._calculate_distance(tracker_info['last_position'], current_hotspot['center'])
                if distance < min_distance and distance < self.similarity_threshold:
                    min_distance = distance
                    best_match = (i, current_hotspot)
            
            if best_match:
                # 매칭된 경우
                match_index, matched_hotspot = best_match
                self._update_tracker(tracker_id, matched_hotspot)
                matched_hotspots.append(self._create_tracked_hotspot(tracker_id, matched_hotspot))
                unmatched_current.pop(match_index)
            else:
                # 매칭되지 않은 경우 - 카운터 감소
                tracker_info['missed_count'] += 1
                if tracker_info['missed_count'] > 3:  # 3프레임 연속 놓치면 제거
                    del self.tracked_hotspots[tracker_id]
        
        # 매칭되지 않은 새로운 열원들을 새 추적기로 등록
        for new_hotspot in unmatched_current:
            new_id = self.next_id
            self.next_id += 1
            self.tracked_hotspots[new_id] = {
                'id': new_id,
                'first_detected': datetime.now(),
                'last_position': new_hotspot['center'],
                'detection_count': 1,
                'missed_count': 0,
                'temperature_history': [new_hotspot['max_temp']],
                'size_history': [new_hotspot['size']],
                'is_confirmed': False
            }
        
        # 지속성이 확인된 열원들만 반환
        confirmed_hotspots = []
        for tracker_id, tracker_info in self.tracked_hotspots.items():
            if tracker_info['detection_count'] >= self.persistence_threshold:
                tracker_info['is_confirmed'] = True
                # 현재 프레임에서 매칭된 열원이 있으면 그 정보 사용
                current_hotspot = None
                for hotspot in matched_hotspots:
                    if hotspot['tracker_id'] == tracker_id:
                        current_hotspot = hotspot
                        break
                
                if current_hotspot:
                    confirmed_hotspots.append(current_hotspot)
        
        return confirmed_hotspots
    
    def _calculate_distance(self, pos1, pos2):
        """두 위치 간의 유클리드 거리 계산"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _update_tracker(self, tracker_id, matched_hotspot):
        """추적기 정보 업데이트"""
        tracker_info = self.tracked_hotspots[tracker_id]
        tracker_info['last_position'] = matched_hotspot['center']
        tracker_info['detection_count'] += 1
        tracker_info['missed_count'] = 0
        tracker_info['temperature_history'].append(matched_hotspot['max_temp'])
        tracker_info['size_history'].append(matched_hotspot['size'])
        
        # 이력 길이 제한
        if len(tracker_info['temperature_history']) > self.max_history:
            tracker_info['temperature_history'].pop(0)
        if len(tracker_info['size_history']) > self.max_history:
            tracker_info['size_history'].pop(0)
    
    def _create_tracked_hotspot(self, tracker_id, current_hotspot):
        """추적 정보가 포함된 열원 데이터 생성"""
        tracker_info = self.tracked_hotspots[tracker_id]
        
        # 온도와 크기의 평균값 계산
        avg_temp = np.mean(tracker_info['temperature_history'])
        avg_size = np.mean(tracker_info['size_history'])
        
        tracked_hotspot = current_hotspot.copy()
        tracked_hotspot.update({
            'tracker_id': tracker_id,
            'detection_count': tracker_info['detection_count'],
            'is_confirmed': tracker_info['is_confirmed'],
            'avg_temp_history': float(avg_temp),
            'avg_size_history': float(avg_size),
            'temperature_trend': self._calculate_trend(tracker_info['temperature_history']),
            'first_detected': tracker_info['first_detected'].strftime("%H:%M:%S")
        })
        
        return tracked_hotspot
    
    def _calculate_trend(self, values):
        """값들의 트렌드 계산 (증가/감소/안정)"""
        if len(values) < 3:
            return "insufficient_data"
        
        recent_avg = np.mean(values[-3:])
        older_avg = np.mean(values[:-3]) if len(values) > 3 else values[0]
        
        diff = recent_avg - older_avg
        if diff > 0.5:
            return "increasing"
        elif diff < -0.5:
            return "decreasing"
        else:
            return "stable"

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
        
        # 열원 추적기 초기화
        self.hotspot_tracker = HotspotTracker(
            max_history=10,           # 최대 10프레임 이력 보관
            similarity_threshold=1.5, # 1.5 픽셀 내에서 같은 열원으로 인정
            persistence_threshold=3   # 3번 이상 감지되면 확실한 열원으로 인정
        )
        
        # 데이터 버퍼 (이전 프레임들 저장)
        self.data_buffer = deque(maxlen=20)  # 최대 20프레임 보관
        
        print("DetectionModule initialized with hotspot tracking")

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
        
        # 데이터 버퍼에 추가
        self.data_buffer.append({
            'time': current_time,
            'values_array': values_array.copy(),
            'timestamp': datetime.now()
        })
        
        # 이상 감지 수행
        detection_result = self.detect_anomalies(values_array, current_time)
        
        # 원시 열원 감지 수행
        raw_hotspots = self.detect_hotspots(values_array)
        
        # 열원 추적 및 지속성 확인
        confirmed_hotspots = self.hotspot_tracker.update(raw_hotspots)
        
        # 추가적인 유사성 검증
        validated_hotspots = self.validate_hotspots_with_history(confirmed_hotspots)
        
        print(f"DetectionModule: Raw hotspots: {len(raw_hotspots)}, "
              f"Confirmed: {len(confirmed_hotspots)}, "
              f"Validated: {len(validated_hotspots)}")
        
        # GUI로 전달할 데이터 패키지 구성
        gui_data_package = {
            'time': current_time,
            'values': values,  # 원본 64개 값 (8x8)
            'sensor_degree': sensor_degree,
            'etc': etc,
            'fire_detected': detection_result['fire_detected'],
            'smoke_detected': detection_result['smoke_detected'],
            'anomaly_count': self.anomaly_total_count,
            'hotspots': validated_hotspots,  # 검증된 열원 정보
            'detection_stats': {
                'max_temp': detection_result['max_temp'],
                'avg_temp': detection_result['avg_temp'],
                'detection_limit': detection_result['detection_limit'],
                'raw_hotspot_count': len(raw_hotspots),
                'confirmed_hotspot_count': len(confirmed_hotspots)
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
        열원의 위치와 형태를 감지 (원시 감지)
        """
        # 가우시안 필터 적용
        filtered_array = gaussian_filter(values_array, sigma=self.gaussian_sigma)
        
        # 임계값을 넘는 영역 찾기
        threshold_temp = values_array.mean() + 0.6    # 동적 임계값 사용
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

    def validate_hotspots_with_history(self, confirmed_hotspots):
        """
        버퍼에 저장된 이전 데이터와 비교하여 열원 유사성 검증
        """
        if len(self.data_buffer) < 3:  # 최소 3프레임 필요
            return confirmed_hotspots
        
        validated_hotspots = []
        
        for hotspot in confirmed_hotspots:
            similarity_score = self.calculate_hotspot_similarity(hotspot)
            
            # 유사성 점수가 임계값 이상이면 유효한 열원으로 인정
            if similarity_score > 0.6:  # 60% 이상 유사하면 유효
                hotspot['similarity_score'] = float(similarity_score)
                hotspot['validation_status'] = 'validated'
                validated_hotspots.append(hotspot)
            else:
                hotspot['similarity_score'] = float(similarity_score)
                hotspot['validation_status'] = 'rejected'
                print(f"Hotspot {hotspot.get('tracker_id', 'unknown')} rejected due to low similarity: {similarity_score:.3f}")
        
        return validated_hotspots

    def calculate_hotspot_similarity(self, current_hotspot):
        """
        현재 열원과 이전 프레임들의 유사성 계산
        """
        if len(self.data_buffer) < 2:
            return 0.0
        
        current_center = current_hotspot['center']
        current_temp = current_hotspot['max_temp']
        current_size = current_hotspot['size']
        
        similarity_scores = []
        
        # 최근 5프레임과 비교
        recent_frames = list(self.data_buffer)[-5:]
        
        for frame_data in recent_frames:
            frame_array = frame_data['values_array']
            frame_hotspots = self.detect_hotspots(frame_array)
            
            # 현재 열원과 가장 가까운 이전 열원 찾기
            best_match_score = 0.0
            
            for prev_hotspot in frame_hotspots:
                # 위치 유사성
                distance = np.sqrt((current_center[0] - prev_hotspot['center'][0])**2 + 
                                 (current_center[1] - prev_hotspot['center'][1])**2)
                position_similarity = max(0, 1 - distance / 3.0)  # 3픽셀 이내에서 유사성 계산
                
                # 온도 유사성
                temp_diff = abs(current_temp - prev_hotspot['max_temp'])
                temp_similarity = max(0, 1 - temp_diff / 5.0)  # 5도 이내에서 유사성 계산
                
                # 크기 유사성
                size_diff = abs(current_size - prev_hotspot['size'])
                size_similarity = max(0, 1 - size_diff / max(current_size, prev_hotspot['size']))
                
                # 전체 유사성 점수 (가중 평균)
                total_similarity = (position_similarity * 0.5 + 
                                  temp_similarity * 0.3 + 
                                  size_similarity * 0.2)
                
                best_match_score = max(best_match_score, total_similarity)
            
            if best_match_score > 0:
                similarity_scores.append(best_match_score)
        
        # 평균 유사성 점수 반환
        return np.mean(similarity_scores) if similarity_scores else 0.0

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