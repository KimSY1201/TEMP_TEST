# detector.py
# -----------------------------------------------------------------------------
# 온도 기반 화재 감지 모듈 (GUI 대시보드 연동)
# - 연기/가스 로직 제거, "정상/주의/화재" 3단계만 사용
# - 필터링/가중치/보간을 Detector에서 처리하여 GUI는 표시에만 집중
# -----------------------------------------------------------------------------

import os
import time
import threading
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import joblib


class DetectionModule(threading.Thread):
    """
    온도 기반 화재 감지만 수행하는 Detector 스레드.

    입력 큐(detection_queue) 항목 예:
        {"time":"HH:MM:SS","values":[64개 float], "sensor_degree": 24.3, "etc":[...]}

    GUI로 내보내는 페이로드(output_queue.put) 예:
        {
          "sensor_id": 1,
          "status": "정상|주의|화재",
          "time": "HH:MM:SS",
          "values": [...64개...],
          "processed_values": [... (interp*interp)개 ...],
          "interpolated_grid_size": 8,
          "detection_stats": {...},
          "processing_params": {...},
          "suspect_fire": [...],
          "heat_source_dict": {"safety":(), "caution":(), "danger":()},
          "total_log": "..."
        }
    """

    def __init__(
        self,
        data_signal_obj,
        detection_queue,
        output_queue,
        base_temperature: float = 25.0,
        threshold: float = 5.0,
        filename: str = "./temp_app/_data/detected_values.txt",
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

        # GUI 파라미터 업데이트 연결
        self.data_signal = data_signal_obj
        if hasattr(self.data_signal, "parameter_update_signal"):
            self.data_signal.parameter_update_signal.connect(self.update_detection_parameters)

        # 설치 위치
        self.sensor_position = "corner"

        # 버퍼/누적
        self.anomaly_total_count = 0
        self.data_buffer = deque(maxlen=20)

        # 필터/가중치/보간
        self.min_temp = 19.0
        self.max_temp = 32.0
        self.avg_temp = 0.0
        self.filter_temp_add = 5.0
        self.filter_temp = 0.0
        self.weight_list = [3.8, 3.9, 4.0, 5.1]            # center용
        self.weight_list_corner = [5, 5, 5, 5, 5, 5, 5, 5] # corner용
        self.interpolated_grid_size = 8

        # 상태 판단 관련
        self.suspect_fire_coordinate = tuple()
        self.total_log = ""

        self.high_temp_counter = 0
        self.safety_high_temp_counter = 0
        self.safety_high_temp_dict = {"safety": (), "caution": (), "danger": ()}

        self.deque_size = 30
        self.suspect_fire_buffer = deque(maxlen=self.deque_size)
        self.fire_detection_buffer = deque(maxlen=self.deque_size)
        self.fire_detection_buffer2 = deque(maxlen=self.deque_size * 2)
        self.buffer_counter = 0
        self.high_temp_counter_init = 0

        # (선택) 모델 로드 (없어도 동작)
        self.model = {}
        self.is_model_loaded = False
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rfm_path = os.path.join(current_dir, "_model", "smv_rfm_model.joblib")
            if os.path.exists(rfm_path):
                self.model["rfm"] = joblib.load(rfm_path)
                self.is_model_loaded = True
            try:
                self.encoder = joblib.load(os.path.join(current_dir, "_model", "smv_label_en.joblib"))
                self.scaler  = joblib.load(os.path.join(current_dir, "_model", "smv_mmx_sc.joblib"))
            except Exception:
                self.encoder = None
                self.scaler = None
        except Exception as e:
            print(f"[Detector] Model load error: {e}")
            self.is_model_loaded = False
            self.encoder = None
            self.scaler = None

    # ---------------- 파라미터 업데이트 ----------------
    def update_detection_parameters(self, params: dict):
        if "sensor_position" in params:
            self.sensor_position = params["sensor_position"]
        if "min_temp" in params:
            self.min_temp = float(params["min_temp"])
        if "max_temp" in params:
            self.max_temp = float(params["max_temp"])
        if "filter_temp_add" in params:
            self.filter_temp_add = float(params["filter_temp_add"])
        if "weight_list" in params:
            self.weight_list = list(params["weight_list"])
        if "weight_list_corner" in params:
            self.weight_list_corner = list(params["weight_list_corner"])
        if "interpolated_grid_size" in params:
            self.interpolated_grid_size = int(params["interpolated_grid_size"])

    # ---------------- 전처리 ----------------
    def apply_heat_source_filtering_center(self, values):
        vals = np.array(values, np.float32).reshape(8, 8)
        filtered = vals.copy()
        self.avg_temp = float(np.mean(vals))
        self.filter_temp = self.avg_temp + self.filter_temp_add

        for i in range(8):
            for j in range(8):
                cur = vals[i, j]
                if cur >= self.filter_temp:
                    if 3 <= i <= 4 and 3 <= j <= 4:
                        filtered[i, j] = cur * self.weight_list[0]
                    elif 2 <= i <= 5 and 2 <= j <= 5:
                        filtered[i, j] = cur * self.weight_list[1]
                    elif 1 <= i <= 6 and 1 <= j <= 6:
                        filtered[i, j] = cur * self.weight_list[2]
                    else:
                        filtered[i, j] = cur * self.weight_list[3]
        return filtered

    def apply_heat_source_filtering_corner(self, values):
        vals = np.array(values, np.float32).reshape(8, 8)
        filtered = vals.copy()
        self.avg_temp = float(np.mean(vals))
        self.filter_temp = self.avg_temp + self.filter_temp_add

        for i in range(8):
            for j in range(8):
                cur = vals[i, j]
                if cur >= self.filter_temp:
                    distance = i + (7 - j)      # 기준: 우상단(0,7)
                    idx = int(distance) if distance < 7 else 7
                    filtered[i, j] = cur * self.weight_list_corner[idx]
        return filtered

    def interpolate_temperature_data(self, filtered_values):
        mul = self.interpolated_grid_size // 8
        if mul <= 1:
            return filtered_values
        return filtered_values.repeat(mul, axis=0).repeat(mul, axis=1)

    # ---------------- 판정 ----------------
    def detect_suspect_fire(self, values):
        coords = np.where((values < min(self.filter_temp, 33)) & (values > self.avg_temp + 2))
        self.suspect_fire_buffer.append(coords)
        if len(self.suspect_fire_buffer) == self.deque_size:
            flat = [item for t in self.suspect_fire_buffer for item in t[0]]
            vc = pd.Series(flat).value_counts()
            thr = len(self.suspect_fire_buffer) * 0.6
            return vc[vc >= thr].index.tolist()
        return []

    def detect_fire(self, values) -> str:
        """최종 상태 문자열 반환: '정상' | '주의' | '화재'"""
        thr = min(self.filter_temp, 33)
        self.high_temp_counter = int(np.sum(values > thr))

        if (self.high_temp_counter > 0 and self.buffer_counter == 0) and \
           self.high_temp_counter != len(self.safety_high_temp_dict):
            self.buffer_counter = self.deque_size
            self.high_temp_counter_init = int(np.sum(values > thr))

        if self.buffer_counter > 0:
            self.fire_detection_buffer.append(values)
            self.buffer_counter -= 1

        if self.safety_high_temp_dict["caution"] is not None:
            self.fire_detection_buffer2.append(values)

        if len(self.fire_detection_buffer2) == self.deque_size * 2 and \
           self.high_temp_counter != len(self.safety_high_temp_dict):
            is_fire, last_coord = self.is_fire(self.fire_detection_buffer2, thr)
            if is_fire:
                self.safety_high_temp_dict["danger"] = last_coord
            else:
                self.safety_high_temp_counter = self.high_temp_counter_init
                self.safety_high_temp_dict["danger"] = tuple()
                self.safety_high_temp_dict["caution"] = tuple()
                self.safety_high_temp_dict["safety"] = last_coord
            self.fire_detection_buffer2.clear()

        if (self.high_temp_counter != self.safety_high_temp_counter) and \
           len(self.fire_detection_buffer) == self.deque_size:
            is_fire, last_coord = self.is_fire(self.fire_detection_buffer, thr)
            if is_fire:
                self.safety_high_temp_dict["caution"] = last_coord
            else:
                self.safety_high_temp_counter = self.high_temp_counter_init
                self.safety_high_temp_dict["safety"] = last_coord

        if len(self.safety_high_temp_dict["danger"]):
            return "화재"
        if len(self.safety_high_temp_dict["caution"]) or self.high_temp_counter > 0:
            return "주의"
        return "정상"

    def is_fire(self, dq, thr):
        """deque 평균 비교로 증가 추세 판단 + 단일 프레임 초과 검사."""
        last_coord = np.where(dq[-1] > thr)[0]
        if int(np.sum(dq[-1] > thr)) > 6:
            return True, last_coord

        if dq:
            frontlist, backlist = [], []
            half = dq.maxlen // 2
            tmp = deque(dq)
            for _ in range(half):
                if len(tmp) <= 1:
                    break
                frontlist.append(int(np.sum(tmp.popleft() > thr)))
                backlist.append(int(np.sum(tmp.pop() > thr)))
            if not frontlist or not backlist:
                return False, last_coord
            return (sum(backlist)/len(backlist) > sum(frontlist)/len(frontlist)), last_coord
        return False, last_coord

    # ---------------- 통계/로그 ----------------
    def calculate_detection_stats(self, original_values, processed_values):
        return {
            "max_temp": float(np.max(original_values)),
            "avg_temp": float(np.mean(original_values)),
            "processed_max_temp": float(np.max(processed_values)),
            "processed_avg_temp": float(np.mean(processed_values)),
            "fire_detected": (len(self.safety_high_temp_dict["danger"]) > 0),
        }

    def log_anomaly(self, current_time, values):
        arr = np.array(values)
        avg = float(np.mean(arr))
        mx = float(np.max(arr))
        strength = "danger" if len(self.safety_high_temp_dict["danger"]) else "caution"
        self.total_log = f"{current_time[2:]} avg:{avg:.2f}°C max:{mx:.2f}°C 강도:{strength}\n"
        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(self.total_log)
        except Exception:
            pass

    # ---------------- 메인 루프 ----------------
    def run(self):
        while self.running:
            if not self.detection_queue or self.detection_queue.empty():
                time.sleep(0.1)
                continue

            data_package = self.detection_queue.get()

            if "values" not in data_package or "time" not in data_package:
                print("[Detector] invalid data_package (need 'values','time')")
                continue

            values = data_package["values"]
            if len(values) != 64:
                print(f"[Detector] Expected 64 values, got {len(values)}. Skip.")
                continue

            # 전처리
            if self.sensor_position == "corner":
                filtered = self.apply_heat_source_filtering_corner(values)
            else:
                filtered = self.apply_heat_source_filtering_center(values)
            interpolated = self.interpolate_temperature_data(filtered)

            # 상태 판단
            status = self.detect_fire(np.array(values, dtype=np.float32))
            self.suspect_fire_coordinate = self.detect_suspect_fire(np.array(values, dtype=np.float32))

            # 통계
            detection_stats = self.calculate_detection_stats(values, interpolated)

            # (옵션) 로그
            if len(self.safety_high_temp_dict["caution"]) or len(self.safety_high_temp_dict["danger"]):
                self.log_anomaly(data_package["time"], values)
            else:
                self.total_log = ""

            # GUI 페이로드
            gui_data_package = {
                "sensor_id": self.sensor_id,
                "status": status,
                "time": data_package["time"],
                "values": values,
                "processed_values": interpolated.flatten().tolist(),
                "interpolated_grid_size": self.interpolated_grid_size,
                "detection_stats": detection_stats,
                "processing_params": {
                    "min_temp": self.min_temp,
                    "max_temp": self.max_temp,
                    "filter_temp": self.filter_temp,
                    "avg_temp": self.avg_temp,
                },
                "suspect_fire": self.suspect_fire_coordinate,
                "heat_source_dict": self.safety_high_temp_dict,
                "total_log": self.total_log,
            }

            # 내부 버퍼(옵션)
            self.data_buffer.append({
                "time": data_package["time"],
                "values_array": np.array(values).reshape((self.grid_size, self.grid_size)),
                "timestamp": datetime.now(),
            })

            # GUI로 전달
            self.output_queue.put(gui_data_package)

    def stop(self):
        self.running = False