import os
import time
import threading
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import joblib

class DetectionModule(threading.Thread):
    def __init__(
        self,
        data_signal_obj,
        detection_queue,
        output_queue,
        base_temperature: float = 25.0,
        threshold: float = 5.0,
        filename: str = "./_data/detected_values.txt", # 경로 수정
        sensor_id: int = 1,
    ):
        super().__init__()
        self.output_queue = output_queue
        self.detection_queue = detection_queue
        self.base_temperature = base_temperature
        self.threshold = threshold
        self.filename = filename
        self.running = True
        self.grid_size = 8
        self.sensor_id = sensor_id

        if hasattr(data_signal_obj, "parameter_update_signal"):
            data_signal_obj.parameter_update_signal.connect(self.update_detection_parameters)

        self.sensor_position = "corner"
        self.anomaly_total_count = 0
        self.data_buffer = deque(maxlen=20)
        self.min_temp, self.max_temp = 19.0, 32.0
        self.avg_temp, self.filter_temp_add, self.filter_temp = 0.0, 5.0, 0.0
        self.weight_list = [3.8, 3.9, 4.0, 5.1]
        self.weight_list_corner = [5, 5, 5, 5, 5, 5, 5, 5]
        self.interpolated_grid_size = 8
        self.suspect_fire_coordinate = tuple()
        self.total_log = ""
        self.high_temp_counter, self.safety_high_temp_counter = 0, 0
        self.safety_high_temp_dict = {"safety": (), "caution": (), "danger": ()}
        self.deque_size = 30
        self.suspect_fire_buffer = deque(maxlen=self.deque_size)
        self.fire_detection_buffer = deque(maxlen=self.deque_size)
        self.fire_detection_buffer2 = deque(maxlen=self.deque_size * 2)
        self.buffer_counter, self.high_temp_counter_init = 0, 0

        self.model = {}
        self.is_model_loaded = False
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rfm_path = os.path.join(current_dir, "_model", "smv_rfm_model.joblib")
            if os.path.exists(rfm_path): self.model["rfm"] = joblib.load(rfm_path); self.is_model_loaded = True
            self.encoder = joblib.load(os.path.join(current_dir, "_model", "smv_label_en.joblib"))
            self.scaler  = joblib.load(os.path.join(current_dir, "_model", "smv_mmx_sc.joblib"))
        except Exception as e:
            print(f"[Detector] Model load error: {e}"); self.is_model_loaded = False; self.encoder = None; self.scaler = None

    def update_detection_parameters(self, params: dict):
        for key, value in params.items(): setattr(self, key, value)

    def apply_heat_source_filtering_center(self, values):
        vals = np.array(values, np.float32).reshape(8, 8); filtered = vals.copy()
        self.avg_temp = float(np.mean(vals)); self.filter_temp = self.avg_temp + self.filter_temp_add
        for i in range(8):
            for j in range(8):
                cur = vals[i, j]
                if cur >= self.filter_temp:
                    if 3<=i<=4 and 3<=j<=4: filtered[i,j] = cur*self.weight_list[0]
                    elif 2<=i<=5 and 2<=j<=5: filtered[i,j] = cur*self.weight_list[1]
                    elif 1<=i<=6 and 1<=j<=6: filtered[i,j] = cur*self.weight_list[2]
                    else: filtered[i,j] = cur*self.weight_list[3]
        return filtered

    def apply_heat_source_filtering_corner(self, values):
        vals = np.array(values, np.float32).reshape(8, 8); filtered = vals.copy()
        self.avg_temp = float(np.mean(vals)); self.filter_temp = self.avg_temp + self.filter_temp_add
        for i in range(8):
            for j in range(8):
                if vals[i,j] >= self.filter_temp:
                    dist = i + (7-j); idx = min(dist, 7)
                    filtered[i,j] = vals[i,j] * self.weight_list_corner[idx]
        return filtered

    def interpolate_temperature_data(self, filtered_values):
        mul = self.interpolated_grid_size // 8
        return filtered_values if mul <= 1 else filtered_values.repeat(mul, axis=0).repeat(mul, axis=1)

    def detect_suspect_fire(self, values):
        coords = np.where((values < min(self.filter_temp, 33)) & (values > self.avg_temp + 2))
        self.suspect_fire_buffer.append(coords)
        if len(self.suspect_fire_buffer) == self.deque_size:
            flat = [item for t in self.suspect_fire_buffer for item in t[0]]
            vc = pd.Series(flat).value_counts()
            return vc[vc >= (len(self.suspect_fire_buffer) * 0.6)].index.tolist()
        return []

    def detect_fire(self, values) -> str:
        thr = min(self.filter_temp, 33)
        self.high_temp_counter = int(np.sum(values > thr))
        if self.high_temp_counter > 0 and self.buffer_counter == 0 and self.high_temp_counter != len(self.safety_high_temp_dict):
            self.buffer_counter = self.deque_size
            self.high_temp_counter_init = self.high_temp_counter
        if self.buffer_counter > 0: self.fire_detection_buffer.append(values); self.buffer_counter -= 1
        if self.safety_high_temp_dict["caution"]: self.fire_detection_buffer2.append(values)
        if len(self.fire_detection_buffer2) == self.deque_size*2 and self.high_temp_counter != len(self.safety_high_temp_dict):
            is_fire, last_coord = self.is_fire(self.fire_detection_buffer2, thr)
            self.safety_high_temp_dict["danger"] = last_coord if is_fire else tuple()
            if not is_fire: self.safety_high_temp_dict.update({"caution": tuple(), "safety": last_coord, "safety_high_temp_counter": self.high_temp_counter_init})
            self.fire_detection_buffer2.clear()
        if self.high_temp_counter != self.safety_high_temp_counter and len(self.fire_detection_buffer) == self.deque_size:
            is_fire, last_coord = self.is_fire(self.fire_detection_buffer, thr)
            if is_fire: self.safety_high_temp_dict["caution"] = last_coord
            else: self.safety_high_temp_dict["safety"] = last_coord; self.safety_high_temp_counter = self.high_temp_counter_init
        if self.safety_high_temp_dict["danger"]: return "화재"
        if self.safety_high_temp_dict["caution"] or self.high_temp_counter > 0: return "주의"
        return "정상"

    def is_fire(self, dq, thr):
        last_coord = np.where(dq[-1] > thr)[0]
        if int(np.sum(dq[-1] > thr)) > 6: return True, last_coord
        if dq and len(dq) > 1:
            half = dq.maxlen // 2; tmp = deque(dq)
            front = [int(np.sum(tmp.popleft() > thr)) for _ in range(half) if tmp]
            back = [int(np.sum(tmp.pop() > thr)) for _ in range(half) if tmp]
            if front and back: return (sum(back)/len(back) > sum(front)/len(front)), last_coord
        return False, last_coord

    def calculate_detection_stats(self, original_values, processed_values):
        return {"max_temp": float(np.max(original_values)), "avg_temp": float(np.mean(original_values)),
                "processed_max_temp": float(np.max(processed_values)), "processed_avg_temp": float(np.mean(processed_values)),
                "fire_detected": bool(self.safety_high_temp_dict["danger"])}

    def log_anomaly(self, current_time, values):
        strength = "danger" if self.safety_high_temp_dict["danger"] else "caution"
        self.total_log = f"{current_time[2:]} avg:{np.mean(values):.2f}°C max:{np.max(values):.2f}°C 강도:{strength}\n"
        try: os.makedirs(os.path.dirname(self.filename), exist_ok=True);
        except: pass
        try:
            with open(self.filename, "a", encoding="utf-8") as f: f.write(self.total_log)
        except Exception: pass

    # ==================== 수정된 부분: run 함수 ====================
    def run(self):
        while self.running:
            try:
                # 1. 큐에서 원본 데이터를 가져옴
                data_package = self.detection_queue.get(timeout=0.1)
                
                # 2. GUI로 보낼 최종 패키지의 기본틀을 원본으로 부터 복사
                gui_data_package = data_package.copy()
                
                # 3. 데이터 유효성 검사
                if 'values' not in data_package or len(data_package['values']) != 64:
                    continue

                values = np.array(data_package['values'])
                
                # 4. 열원 필터링 및 보간
                if self.sensor_position == 'corner':
                    filtered_values = self.apply_heat_source_filtering_corner(values)
                else:
                    filtered_values = self.apply_heat_source_filtering_center(values)
                interpolated_values = self.interpolate_temperature_data(filtered_values)
                
                # 5. 상태 판정 (정상/주의/화재)
                final_status = self.detect_fire(values)
                
                # 6. 통계 계산 및 로그 기록
                detection_stats = self.calculate_detection_stats(values, interpolated_values)
                if final_status in ["주의", "화재"]:
                    self.log_anomaly(data_package.get('time', ''), values)
                else:
                    self.total_log = ''
                
                # 7. 최종 GUI 패키지에 모든 처리 결과를 업데이트 (추가)
                gui_data_package.update({
                    "sensor_id": self.sensor_id,
                    "status": final_status, # <--- 최종 판정 상태 추가
                    "processed_values": interpolated_values.flatten().tolist(), # <--- 히트맵 데이터!
                    "interpolated_grid_size": self.interpolated_grid_size,
                    "detection_stats": detection_stats,
                    "heat_source_dict": self.safety_high_temp_dict,
                    "total_log": self.total_log,
                    "processing_params": {
                        'min_temp': self.min_temp,
                        'max_temp': self.max_temp,
                        'filter_temp': self.filter_temp,
                        'avg_temp': self.avg_temp
                    }
                })
                
                # 8. 모든 정보가 담긴 패키지를 GUI로 전송
                self.output_queue.put(gui_data_package)

            except queue.Empty:
                time.sleep(0.05) # 큐가 비었을 때 CPU 사용 줄이기

    def stop(self):
        self.running = False