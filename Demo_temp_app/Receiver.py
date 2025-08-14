import sys
import json
from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Event

# SerialReceiver 임포트 구문을 제거하고 TCPReceiver만 남깁니다.
from transmission import TCPReceiver, TransmissionSignal

def extract_sensor_id(payload: dict, fallback: int) -> int:
    """
    수신 페이로드에서 sensor_id를 최대한 찾아 반환.
    (이 함수는 원본과 동일하게 유지됩니다)
    """
    if not isinstance(payload, dict):
        return fallback
    for key in ("sensor_id", "id", "channel"):
        if key in payload:
            try: return int(payload[key])
            except Exception: pass
    inner = payload.get("data")
    if isinstance(inner, dict):
        for key in ("sensor_id", "id", "channel"):
            if key in inner:
                try: return int(inner[key])
                except Exception: pass
        inner2 = inner.get("data")
        if isinstance(inner2, dict):
            for key in ("sensor_id", "id", "channel"):
                if key in inner2:
                    try: return int(inner2[key])
                    except Exception: pass
    return fallback

class ReceiverWorker(Thread):
    """
    GUI가 없는 순수 데이터 수신 작업자 클래스 (TCP 전용).
    Detector가 원하는 형태로 데이터를 제공하도록 수정.
    """
    def __init__(self, detection_queue: Queue, port, baudrate=57600, default_sid=1):
        super().__init__()
        self.daemon = True
        self.detection_queue = detection_queue
        self.port = port
        self.baudrate = baudrate
        self.default_sensor_id = default_sid
        self.transmission_signal = TransmissionSignal()
        self.internal_queue = Queue()
        self.receiver_thread = None
        self._stop_event = Event()

    def run(self):
        try:
            if isinstance(self.port, int):
                self.receiver_thread = TCPReceiver(
                    self.transmission_signal,
                    self.internal_queue,
                    host="0.0.0.0",
                    port=self.port
                )
                print(f"[ReceiverWorker] TCP 리스너 시작 (포트: {self.port})")
            else:
                print(f"[ReceiverWorker] 에러: TCP 통신만 지원합니다. (입력된 포트: {self.port})")
                return
            
            self.receiver_thread.start()

            while not self._stop_event.is_set():
                try:
                    # 1. 외부로부터 원본 데이터를 받음
                    original_item = self.internal_queue.get(timeout=0.1)
                    
                    # 2. 원본 데이터에서 핵심 내용물(payload)을 꺼냄
                    #    (예: {'time': '...', 'values': [...]})
                    original_payload = original_item.get("data", {})
                    
                    # 3. Detector가 원하는 형태로 데이터를 '단순화'
                    if 'values' in original_payload and 'time' in original_payload:
                        
                        # Detector가 필요한 최소한의 정보만 담은 새 딕셔너리 생성
                        data_for_detector = {
                            'time': original_payload['time'],
                            'values': original_payload['values'],
                            # 혹시 Detector가 sensor_id를 사용할 경우를 대비해 추가
                            'sensor_id': self.default_sensor_id 
                        }
                        
                        # 4. 단순화된 데이터를 Detector에게 전달
                        self.detection_queue.put(data_for_detector)
                        
                except Empty:
                    continue
                except Exception as e:
                    print(f"[ReceiverWorker] 데이터 처리 중 에러 발생: {e}")

        except Exception as e:
            print(f"[ReceiverWorker] 리스너 시작 실패 ({self.port}): {e}")
        finally:
            self.stop_internal_receiver()
            print(f"[ReceiverWorker] 리스너 중지됨 ({self.port})")

    def stop(self):
        self._stop_event.set()
        self.stop_internal_receiver()

    def stop_internal_receiver(self):
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.stop()
            self.receiver_thread.join(timeout=1)
            self.receiver_thread = None