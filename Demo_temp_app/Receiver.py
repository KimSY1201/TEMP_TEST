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
    """
    def __init__(self, detection_queue: Queue, port, baudrate=57600, default_sid=1):
        super().__init__()
        self.daemon = True
        self.detection_queue = detection_queue
        self.port = port
        # baudrate는 TCP 통신에서 사용되지 않지만, main.py와의 호환성을 위해 남겨둡니다.
        self.baudrate = baudrate
        self.default_sensor_id = default_sid
        self.transmission_signal = TransmissionSignal()
        self.internal_queue = Queue()
        self.receiver_thread = None
        self._stop_event = Event()

    def run(self):
        """스레드가 시작될 때 실행되는 메인 로직"""
        try:
            # 포트가 정수(int)인 경우에만 TCP 리스너를 시작합니다.
            if isinstance(self.port, int):
                self.receiver_thread = TCPReceiver(
                    self.transmission_signal,
                    self.internal_queue,
                    host="0.0.0.0",
                    port=self.port
                )
                print(f"[ReceiverWorker] TCP 리스너 시작 (포트: {self.port})")
            else:
                # 시리얼 포트(문자열)가 들어오면 에러를 출력하고 종료합니다.
                print(f"[ReceiverWorker] 에러: TCP 통신만 지원합니다. 포트 번호는 정수여야 합니다. (입력된 포트: {self.port})")
                return
            
            self.receiver_thread.start()

            # 데이터 처리 루프
            while not self._stop_event.is_set():
                try:
                    item = self.internal_queue.get(timeout=0.1)
                    payload = item.get("data", {})
                    sid = extract_sensor_id(payload, self.default_sensor_id)
                    item["sensor_id"] = sid
                    self.detection_queue.put(item)
                except Empty:
                    continue
                except Exception as e:
                    print(f"[ReceiverWorker] 큐 처리 중 에러 발생: {e}")

        except Exception as e:
            print(f"[ReceiverWorker] 리스너 시작 실패 ({self.port}): {e}")
        finally:
            self.stop_internal_receiver()
            print(f"[ReceiverWorker] 리스너 중지됨 ({self.port})")

    def stop(self):
        """외부(main.py)에서 스레드를 중지시키기 위해 호출하는 메서드"""
        self._stop_event.set()
        self.stop_internal_receiver()

    def stop_internal_receiver(self):
        """내부 TCP 스레드를 안전하게 중지"""
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.stop()
            self.receiver_thread.join(timeout=1)
            self.receiver_thread = None