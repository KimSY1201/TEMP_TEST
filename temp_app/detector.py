import numpy as np
import pandas as pd
from datetime import datetime
import threading
import time
import queue
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label, center_of_mass
from collections import deque
import cv2
import joblib
from sklearn.preprocessing import StandardScaler

class HotspotTracker:
    """열원 추적을 위한 클래스"""
    
    def __init__(self, max_history=10, similarity_threshold=1.5, persistence_threshold=10):
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

class MOGBackgroundSubtractor:
    """MOG2 기반 배경 차분을 위한 클래스"""
    
    def __init__(self, history=500, var_threshold=16, detect_shadows=False):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.learning_rate = 0.01  # 배경 학습률
        self.min_area = 1  # 최소 전경 영역 크기
        self.max_area = 20  # 최대 전경 영역 크기
        self.initialized = False
        self.frame_count = 0
        
    def apply(self, temperature_array):
        """온도 배열에 MOG2 적용"""
        # 온도 데이터를 0-255 범위로 정규화
        temp_normalized = self._normalize_temperature(temperature_array)
        
        # 8비트 그레이스케일로 변환
        temp_uint8 = temp_normalized.astype(np.uint8)
        
        # MOG2 적용
        learning_rate = self.learning_rate if self.initialized else -1
        foreground_mask = self.mog2.apply(temp_uint8, learningRate=learning_rate)
        
        self.frame_count += 1
        if self.frame_count > 10:  # 10프레임 후 초기화 완료
            self.initialized = True
        
        return foreground_mask
    
    def _normalize_temperature(self, temp_array):
        """온도 배열을 0-255 범위로 정규화"""
        min_temp = np.min(temp_array)
        max_temp = np.max(temp_array)
        
        if max_temp - min_temp == 0:
            return np.zeros_like(temp_array)
        
        normalized = (temp_array - min_temp) / (max_temp - min_temp) * 255
        return normalized
    
    def extract_hotspots(self, foreground_mask, temperature_array):
        """전경 마스크에서 열원 추출"""
        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # 연결된 구성요소 찾기
        labeled_array, num_features = label(cleaned_mask > 0)
        
        hotspots = []
        
        for i in range(1, num_features + 1):
            # 각 전경 영역 분석
            region_mask = (labeled_array == i)
            region_size = np.sum(region_mask)
            
            # 크기 필터링
            if self.min_area <= region_size <= self.max_area:
                # 중심점 계산
                center_y, center_x = center_of_mass(region_mask)
                
                # 해당 영역의 온도 정보
                region_temps = temperature_array[region_mask]
                max_temp = np.max(region_temps)
                avg_temp = np.mean(region_temps)
                
                # 영역 경계
                region_coords = np.where(region_mask)
                min_row, max_row = np.min(region_coords[0]), np.max(region_coords[0])
                min_col, max_col = np.min(region_coords[1]), np.max(region_coords[1])
                
                hotspot_info = {
                    'id': i,
                    'center': (float(center_x), float(center_y)),
                    'size': int(region_size),
                    'max_temp': float(max_temp),
                    'avg_temp': float(avg_temp),
                    'bbox': {
                        'min_row': int(min_row),
                        'max_row': int(max_row),
                        'min_col': int(min_col),
                        'max_col': int(max_col)
                    },
                    'coordinates': [(int(x), int(y)) for x, y in zip(region_coords[1], region_coords[0])],
                    'detection_method': 'MOG2'
                }
                
                hotspots.append(hotspot_info)
        
        return hotspots

class CrossValidator:
    """온도 기반 방식과 MOG2 방식의 교차검증을 위한 클래스"""
    
    def __init__(self, overlap_threshold=0.5, consensus_threshold=0.7):
        self.overlap_threshold = overlap_threshold  # 겹침 판정 임계값
        self.consensus_threshold = consensus_threshold  # 합의 점수 임계값
        
    def cross_validate(self, temp_hotspots, mog_hotspots):
        """두 방식의 결과를 교차검증"""
        validated_hotspots = []
        validation_stats = {
            'temp_only': 0,
            'mog_only': 0,
            'both_detected': 0,
            'consensus_validated': 0
        }
        
        # 온도 기반 열원들 검증
        for temp_hotspot in temp_hotspots:
            temp_hotspot['detection_method'] = 'lgbm_model'  # 수정된 부분
            consensus_score = self._calculate_consensus(temp_hotspot, mog_hotspots)
            
            if consensus_score >= self.consensus_threshold:
                temp_hotspot['consensus_score'] = consensus_score
                temp_hotspot['validation_status'] = 'consensus_validated'
                temp_hotspot['cross_validation'] = True
                validated_hotspots.append(temp_hotspot)
                validation_stats['consensus_validated'] += 1
                validation_stats['both_detected'] += 1
            else:
                temp_hotspot['consensus_score'] = consensus_score
                temp_hotspot['validation_status'] = 'temp_only'
                temp_hotspot['cross_validation'] = False
                # 온도 기반만 감지된 경우도 포함 (더 보수적 접근)
                if consensus_score > 0.3:  # 약간의 겹침이라도 있으면 포함
                    validated_hotspots.append(temp_hotspot)
                validation_stats['temp_only'] += 1
        
        # MOG2 기반 열원들 중 온도 기반으로 감지되지 않은 것들 검증
        for mog_hotspot in mog_hotspots:
            if not self._already_matched(mog_hotspot, temp_hotspots):
                mog_hotspot['consensus_score'] = 0.0
                mog_hotspot['validation_status'] = 'mog_only'
                mog_hotspot['cross_validation'] = False
                # MOG2만 감지한 경우는 조건부 포함
                if self._is_significant_mog_detection(mog_hotspot):
                    validated_hotspots.append(mog_hotspot)
                validation_stats['mog_only'] += 1
        
        return validated_hotspots, validation_stats
    
    def _calculate_consensus(self, temp_hotspot, mog_hotspots):
        """온도 기반 열원과 MOG2 열원들 간의 합의 점수 계산"""
        if not mog_hotspots:
            return 0.0
        
        best_overlap = 0.0
        
        for mog_hotspot in mog_hotspots:
            overlap_score = self._calculate_overlap(temp_hotspot, mog_hotspot)
            best_overlap = max(best_overlap, overlap_score)
        
        return best_overlap
    
    def _calculate_overlap(self, hotspot1, hotspot2):
        """두 열원 간의 겹침 점수 계산"""
        # 위치 기반 겹침
        center1 = hotspot1['center']
        center2 = hotspot2['center']
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # 바운딩 박스 기반 겹침
        bbox1 = hotspot1['bbox']
        bbox2 = hotspot2['bbox']
        
        # IoU (Intersection over Union) 계산
        x1_inter = max(bbox1['min_col'], bbox2['min_col'])
        y1_inter = max(bbox1['min_row'], bbox2['min_row'])
        x2_inter = min(bbox1['max_col'], bbox2['max_col'])
        y2_inter = min(bbox1['max_row'], bbox2['max_row'])
        
        if x2_inter > x1_inter and y2_inter > y1_inter:
            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            area1 = (bbox1['max_col'] - bbox1['min_col']) * (bbox1['max_row'] - bbox1['min_row'])
            area2 = (bbox2['max_col'] - bbox2['min_col']) * (bbox2['max_row'] - bbox2['min_row'])
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0
        else:
            iou = 0
        
        # 거리 기반 점수
        distance_score = max(0, 1 - distance / 3.0)  # 3픽셀 내에서 유사성 계산
        
        # 종합 점수 (IoU와 거리 점수의 가중 평균)
        overlap_score = (iou * 0.6 + distance_score * 0.4)
        
        return overlap_score
    
    def _already_matched(self, mog_hotspot, temp_hotspots):
        """MOG2 열원이 이미 온도 기반 열원과 매칭되었는지 확인"""
        for temp_hotspot in temp_hotspots:
            if self._calculate_overlap(mog_hotspot, temp_hotspot) >= self.overlap_threshold:
                return True
        return False
    
    def _is_significant_mog_detection(self, mog_hotspot):
        """MOG2 단독 감지가 의미있는지 판단"""
        # 크기나 지속성 등을 고려한 판단 로직
        return mog_hotspot['size'] >= 2  # 최소 2픽셀 이상

class LGBMHotspotDetector:
    """LightGBM 모델을 사용한 열원 감지 클래스"""
    
    def __init__(self, model_path="./lgbm_human_detection_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()  # 데이터 정규화용
        self.is_model_loaded = False
        self.confidence_threshold = 0.5  # 예측 확률 임계값
        
        # 모델 로드 시도
        self.load_model()
        
        # 8x8을 24x24로 업샘플링할 때 사용할 매개변수
        self.upsampling_method = 'bicubic'  # 'nearest', 'bilinear', 'bicubic' 중 선택
        
    def load_model(self):
        """LightGBM 모델 로드"""
        try:
            self.model = joblib.load(self.model_path)
            self.is_model_loaded = True
            print(f"LGBMHotspotDetector: Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"LGBMHotspotDetector: Model file not found at {self.model_path}")
            self.is_model_loaded = False
        except Exception as e:
            print(f"LGBMHotspotDetector: Error loading model: {e}")
            self.is_model_loaded = False
    
    def filtered_data(self, values_8x8):
        """8x8 배열을 24x24로 업샘플링"""
        frame_8x8 = values_8x8
        frame_24x24 = cv2.resize(frame_8x8, (24,24), interpolation=cv2.INTER_CUBIC)
        
        # 이제 모든 특성을 24x24 이미지(frame_24x24)에서 계산합니다.
        features = {}
    
        # --- 전역(Global) 통계 특성 ---
        features['mean_temp'] = np.mean(frame_24x24)
        features['std_temp'] = np.std(frame_24x24)
        features['max_temp'] = np.max(frame_24x24)
        features['min_temp'] = np.min(frame_24x24)
        features['q95_temp'] = np.quantile(frame_24x24, 0.95)
        features['max_minus_mean'] = features['max_temp'] - features['mean_temp']
        
        # --- 공간(Zonal) 특성 (24x24 기준에 맞게 인덱스 조정) ---
        # 중앙 12x12 구역 정의 (24*0.25=6, 24*0.75=18)
        center_zone = frame_24x24[6:18, 6:18]
        
        features['center_mean_temp'] = np.mean(center_zone)
        features['center_std_temp'] = np.std(center_zone)
        features['center_vs_global_mean_diff'] = features['center_mean_temp'] - features['mean_temp']
        
        # 최대 온도의 위치 정보 (24x24 이미지 기준)
        max_pos = np.unravel_index(np.argmax(frame_24x24), frame_24x24.shape)
        features['max_temp_y_pos'] = max_pos[0]
        features['max_temp_x_pos'] = max_pos[1]
        
        feature_names_in_order = [
            'mean_temp', 'std_temp', 'max_temp', 'min_temp', 'q95_temp', 'max_minus_mean',
            'center_mean_temp', 'center_std_temp', 'center_vs_global_mean_diff',
            'max_temp_y_pos', 'max_temp_x_pos'
        ]

        # 딕셔너리의 순서를 보장하거나 (Python 3.7+),
        # DataFrame을 생성할 때 columns 인수에 명시적으로 순서 지정
        filted_data = pd.DataFrame([features], columns=feature_names_in_order)
        # filted_data = pd.DataFrame([features])
        
        return [filted_data.iloc[0, :]]
    
    def detect_hotspots(self, values_array):
        """LightGBM 모델을 사용하여 열원 감지"""
        if not self.is_model_loaded:
            print("LGBMHotspotDetector: Model not loaded, falling back to threshold-based detection")
            return self._fallback_detection(values_array)
        
        try:
            # 8x8을 24x24로 업샘플링 후 정제
            upscaled_array = self.filtered_data(values_array)
            # print(upscaled_array)
            # 모델 예측 (확률 반환)
            prediction_proba = self.model.predict_proba(upscaled_array)[0]
            prediction_class = self.model.predict(upscaled_array)[0]
            
            # 열원이 있다고 예측된 경우 (클래스 1)
            if prediction_class == 1 and prediction_proba[1] >= self.confidence_threshold:
                hotspots = self._extract_hotspot_from_prediction(
                    values_array, upscaled_array, prediction_proba[1]
                )
            
                print(f"LGBMHotspotDetector: Hotspot detected with confidence {prediction_proba[1]:.3f}")
                return hotspots
            else:
                print(f"LGBMHotspotDetector: No hotspot detected (confidence: {prediction_proba[1]:.3f})")
                return []
                
        except Exception as e:
            print(f"LGBMHotspotDetector: Error in model prediction: {e}")
            return self._fallback_detection(values_array)
    
    def _extract_hotspot_from_prediction(self, original_8x8, upscaled_24x24, confidence):
        """모델 예측 결과로부터 열원 정보 추출"""
        # 원본 8x8 배열에서 가장 높은 온도 지점들을 찾아 열원으로 간주
        mean_temp = np.mean(original_8x8)
        std_temp = np.std(original_8x8)
        threshold = mean_temp + std_temp * 0.5  # 동적 임계값
        
        # 임계값을 넘는 영역 찾기
        hotspot_mask = original_8x8 > threshold
        
        if not np.any(hotspot_mask):
            # 임계값을 넘는 지점이 없으면 최고 온도 지점을 열원으로 간주
            max_temp_idx = np.unravel_index(np.argmax(original_8x8), original_8x8.shape)
            hotspot_mask = np.zeros_like(original_8x8, dtype=bool)
            hotspot_mask[max_temp_idx] = True
        
        # 연결된 구성요소 분석
        labeled_array, num_features = label(hotspot_mask)
        
        hotspots = []
        
        for i in range(1, num_features + 1):
            region_mask = (labeled_array == i)
            region_size = np.sum(region_mask)
            
            # 중심점 계산
            center_y, center_x = center_of_mass(region_mask)
            
            # 해당 영역의 온도 정보
            region_temps = original_8x8[region_mask]
            max_temp = np.max(region_temps)
            avg_temp = np.mean(region_temps)
            
            # 영역 경계
            region_coords = np.where(region_mask)
            min_row, max_row = np.min(region_coords[0]), np.max(region_coords[0])
            min_col, max_col = np.min(region_coords[1]), np.max(region_coords[1])
            
            hotspot_info = {
                'id': i,
                'center': (float(center_x), float(center_y)),
                'size': int(region_size),
                'max_temp': float(max_temp),
                'avg_temp': float(avg_temp),
                'bbox': {
                    'min_row': int(min_row),
                    'max_row': int(max_row),
                    'min_col': int(min_col),
                    'max_col': int(max_col)
                },
                'coordinates': [(int(x), int(y)) for x, y in zip(region_coords[1], region_coords[0])],
                'detection_method': 'lgbm_model',
                'model_confidence': float(confidence),
                'model_threshold': float(threshold)
            }
            
            hotspots.append(hotspot_info)
        
        return hotspots
    
    def _fallback_detection(self, values_array):
        """모델 로드 실패 시 사용할 기본 감지 방법"""
        # 기존 온도 기반 방식과 유사한 fallback
        mean_temp = np.mean(values_array)
        std_temp = np.std(values_array)
        threshold = mean_temp + std_temp * 1.0  # 1 표준편차 이상
        
        hotspot_mask = values_array > threshold
        
        if not np.any(hotspot_mask):
            return []
        
        labeled_array, num_features = label(hotspot_mask)
        hotspots = []
        
        for i in range(1, num_features + 1):
            region_mask = (labeled_array == i)
            region_size = np.sum(region_mask)
            
            if region_size >= 1:  # 최소 1픽셀 이상
                center_y, center_x = center_of_mass(region_mask)
                region_temps = values_array[region_mask]
                max_temp = np.max(region_temps)
                avg_temp = np.mean(region_temps)
                
                region_coords = np.where(region_mask)
                min_row, max_row = np.min(region_coords[0]), np.max(region_coords[0])
                min_col, max_col = np.min(region_coords[1]), np.max(region_coords[1])
                
                hotspot_info = {
                    'id': i,
                    'center': (float(center_x), float(center_y)),
                    'size': int(region_size),
                    'max_temp': float(max_temp),
                    'avg_temp': float(avg_temp),
                    'bbox': {
                        'min_row': int(min_row),
                        'max_row': int(max_row),
                        'min_col': int(min_col),
                        'max_col': int(max_col)
                    },
                    'coordinates': [(int(x), int(y)) for x, y in zip(region_coords[1], region_coords[0])],
                    'detection_method': 'fallback_threshold',
                    'model_confidence': 0.0,
                    'model_threshold': float(threshold)
                }
                
                hotspots.append(hotspot_info)
        
        return hotspots

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
        
        # LightGBM 열원 감지기 초기화 (새로 추가)
        self.lgbm_detector = LGBMHotspotDetector("lgbm_human_detection_model.joblib")
        
        # 열원 추적기 초기화
        self.hotspot_tracker = HotspotTracker(
            max_history=10,           # 최대 10프레임 이력 보관
            similarity_threshold=1.5, # 1.5 픽셀 내에서 같은 열원으로 인정
            persistence_threshold=3   # 3번 이상 감지되면 확실한 열원으로 인정
        )
        
        # MOG2 배경 차분기 초기화
        self.mog_subtractor = MOGBackgroundSubtractor(
            history=500,
            var_threshold=12,
            detect_shadows=False
        )
        
        # 교차검증기 초기화
        self.cross_validator = CrossValidator(
            overlap_threshold=0.5,     # 50% 이상 겹치면 같은 열원으로 인정
            consensus_threshold=0.7    # 70% 이상 합의점수면 검증된 열원으로 인정
        )
        
        # 데이터 버퍼 (이전 프레임들 저장)
        self.data_buffer = deque(maxlen=20)  # 최대 20프레임 보관
        
        print("DetectionModule initialized with LightGBM + MOG2 hybrid detection system")

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
        
        # 1. 기본 온도 기반 이상 감지 수행
        detection_result = self.detect_anomalies(values_array, current_time)
        
        # 2. LightGBM 기반 열원 감지 (기존 온도 기반 방식 대체)
        lgbm_based_hotspots = self.detect_hotspots_lgbm(values_array)
        
        # 3. MOG2 기반 열원 감지 (항상 실행)
        mog_based_hotspots = self.detect_hotspots_mog2(values_array)
        
        # 4. 교차검증 수행 (LightGBM + MOG2)
        cross_validated_hotspots, validation_stats = self.cross_validator.cross_validate(
            lgbm_based_hotspots, mog_based_hotspots
        )
        
        # 5. 열원 추적 및 지속성 확인 (교차검증된 열원들에 대해)
        confirmed_hotspots = []
        if cross_validated_hotspots:
            confirmed_hotspots = self.hotspot_tracker.update(cross_validated_hotspots)
        
        # 6. 추가적인 유사성 검증
        validated_hotspots = self.validate_hotspots_with_history(confirmed_hotspots)
        
        # 7. 화재/연기 감지 업데이트 (LightGBM 결과 반영)
        updated_detection_result = self.update_fire_smoke_detection(
            detection_result, validated_hotspots
        )
        
        # 로그 출력
        print(f"DetectionModule: LightGBM hotspots: {len(lgbm_based_hotspots)}, "
              f"MOG2 hotspots: {len(mog_based_hotspots)}, "
              f"Cross-validated: {len(cross_validated_hotspots)}, "
              f"Final validated: {len(validated_hotspots)}")
        print(f"Validation stats: {validation_stats}")
        
        # GUI로 전달할 데이터 패키지 구성
        gui_data_package = {
            'time': current_time,
            'values': values,  # 원본 64개 값 (8x8)
            'sensor_degree': sensor_degree,
            'etc': etc,
            'fire_detected': updated_detection_result['fire_detected'],
            'smoke_detected': updated_detection_result['smoke_detected'],
            'anomaly_count': self.anomaly_total_count,
            'hotspots': validated_hotspots,  # 최종 검증된 열원 정보
            'detection_stats': {
                'max_temp': updated_detection_result['max_temp'],
                'avg_temp': updated_detection_result['avg_temp'],
                'detection_limit': updated_detection_result['detection_limit'],
                'lgbm_hotspot_count': len(lgbm_based_hotspots),
                'mog_hotspot_count': len(mog_based_hotspots),
                'cross_validated_count': len(cross_validated_hotspots),
                'final_validated_count': len(validated_hotspots)
            },
            'validation_stats': validation_stats  # 교차검증 통계
        }
        
        # GUI 큐로 전달
        self.output_queue.put(gui_data_package)

    def detect_anomalies(self, values_array, current_time):
        """
        온도 이상 감지 로직 (기존 로직 유지)
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
            if max_temp > (self.base_temperature + self.threshold * 2 + 30):  # 더 높은 온도는 화재
                fire_detected = True
            elif max_temp > detection_limit + 30:  # 임계값 초과는 연기
                smoke_detected = True
            
            # 로그 파일에 기록
            self.log_detection(current_time, filtered_array, detected_indices, average, detection_limit)
            
            print(f"DetectionModule: Temperature anomaly detected at {current_time}. Total: {self.anomaly_total_count}")

        return {
            'fire_detected': fire_detected,
            'smoke_detected': smoke_detected,
            'max_temp': max_temp,
            'avg_temp': average,
            'detection_limit': detection_limit,
            'anomaly_indices': detected_indices
        }

    def detect_hotspots_lgbm(self, values_array):
        """
        LightGBM 모델 기반 열원 감지 (기존 detect_hotspots_temperature 대체)
        """
        return self.lgbm_detector.detect_hotspots(values_array)

    def detect_hotspots_mog2(self, values_array):
        """
        MOG2 기반 열원 감지 (기존 로직 유지)
        """
        try:
            # MOG2로 전경 마스크 생성
            foreground_mask = self.mog_subtractor.apply(values_array)
            
            # MOG2 결과에서 열원 추출
            mog_hotspots = self.mog_subtractor.extract_hotspots(foreground_mask, values_array)
            
            return mog_hotspots
            
        except Exception as e:
            print(f"MOG2 detection error: {e}")
            return []

    def update_fire_smoke_detection(self, detection_result, validated_hotspots):
        """
        LightGBM 감지 결과를 반영하여 화재/연기 감지 상태 업데이트
        """
        updated_result = detection_result.copy()
        
        # LightGBM으로 감지된 열원이 있으면 화재/연기 감지 강화
        if validated_hotspots:
            for hotspot in validated_hotspots:
                # LightGBM 모델로 감지된 열원의 경우
                if hotspot.get('detection_method') == 'lgbm_model':
                    model_confidence = hotspot.get('model_confidence', 0.0)
                    
                    # # 높은 신뢰도의 경우 화재로 분류
                    # if model_confidence > 0.8:
                    #     updated_result['fire_detected'] = True
                    #     print(f"Fire detected by LightGBM with confidence: {model_confidence:.3f}")
                    # # 중간 신뢰도의 경우 연기로 분류
                    # elif model_confidence > 0.5:
                    #     updated_result['smoke_detected'] = True
                    #     print(f"Smoke detected by LightGBM with confidence: {model_confidence:.3f}")
                
                # 교차검증으로 확인된 열원의 경우 가중치 부여
                if hotspot.get('cross_validation', False):
                    consensus_score = hotspot.get('consensus_score', 0.0)
                    if consensus_score > 0.7:
                        if hotspot.get('max_temp', 0) > detection_result['avg_temp'] + 3:
                            updated_result['fire_detected'] = True
                        else:
                            updated_result['smoke_detected'] = True
        
        return updated_result

    def validate_hotspots_with_history(self, confirmed_hotspots):
        """
        버퍼에 저장된 이전 데이터와 비교하여 열원 유사성 검증
        """
        if len(self.data_buffer) < 3:  # 최소 3프레임 필요
            return confirmed_hotspots
        
        validated_hotspots = []
        
        for hotspot in confirmed_hotspots:
            similarity_score = self.calculate_hotspot_similarity(hotspot)
            
            # LightGBM 감지의 경우 더 관대한 임계값 적용
            if hotspot.get('detection_method') == 'lgbm_model':
                threshold = 0.5  # LightGBM은 50% 유사성으로 충분
            else:
                threshold = 0.6  # 다른 방법은 60% 유사성 필요
            
            if similarity_score > threshold:
                hotspot['similarity_score'] = float(similarity_score)
                hotspot['validation_status'] = 'history_validated'
                validated_hotspots.append(hotspot)
            else:
                hotspot['similarity_score'] = float(similarity_score)
                hotspot['validation_status'] = 'history_rejected'
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
            
            # LightGBM과 MOG2 기반 모두 시도
            frame_lgbm_hotspots = self.detect_hotspots_lgbm(frame_array)
            frame_mog_hotspots = self.detect_hotspots_mog2(frame_array)
            
            # 두 방식의 결과를 합쳐서 비교
            all_frame_hotspots = frame_lgbm_hotspots + frame_mog_hotspots
            
            # 현재 열원과 가장 가까운 이전 열원 찾기
            best_match_score = 0.0
            
            for prev_hotspot in all_frame_hotspots:
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
                
                # 감지 방법 일치성 보너스
                method_bonus = 0.15 if current_hotspot.get('detection_method') == prev_hotspot.get('detection_method') else 0
                
                # LightGBM 방법의 경우 추가 보너스
                lgbm_bonus = 0.1 if (current_hotspot.get('detection_method') == 'lgbm_model' and 
                                   prev_hotspot.get('detection_method') == 'lgbm_model') else 0
                
                # 전체 유사성 점수 (가중 평균)
                total_similarity = (position_similarity * 0.4 + 
                                  temp_similarity * 0.3 + 
                                  size_similarity * 0.2 + 
                                  method_bonus + lgbm_bonus)
                
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