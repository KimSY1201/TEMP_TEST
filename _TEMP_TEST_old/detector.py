import numpy as np
from datetime import datetime
import threading
import time

class DetectionModule(threading.Thread):
    def __init__(self, detection_queue, base_temperature=25.0, threshold=5.0, filename="detected_values.txt"):
        super().__init__()
        self.detection_queue = detection_queue
        self.base_temperature = base_temperature
        self.threshold = threshold
        self.filename = filename
        self.running = True
        
        # === [수정됨] 누적 이상 감지 횟수를 저장할 변수 추가 ===
        self.anomaly_total_count = 0

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
                    else:
                        print(f"DetectionModule: Expected 64 values, got {len(values)}. Skipping processing.")
                else:
                    print(f"DetectionModule: Received invalid data_package. Keys 'values' or 'time' missing.")
            else:
                time.sleep(0.1) # 큐가 비어있으면 잠시 대기

    def process_values(self, current_time, values):
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

    def stop(self):
        self.running = False