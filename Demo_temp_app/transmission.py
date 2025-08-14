"""
transmission.py
TCP/IP 송수신 모듈 (8080 송신 / 8081 수신 구조 유지)
- 길이 프리픽스(4바이트) + UTF-8 JSON 전송
- 송신/수신 통계 및 상태 시그널 강화
"""
import socket
import threading
import json
from datetime import datetime
from queue import Queue, Empty

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal


# ---- NumPy JSON 인코더 ----
class NumpyEncoder(json.JSONEncoder):
    """NumPy 타입을 JSON 직렬화 가능하게 변환"""
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return super().default(obj)


# ---- Qt Signals ----
class TransmissionSignal(QObject):
    status_update = pyqtSignal(dict)       # 통합 상태
    connection_status = pyqtSignal(dict)   # 서버/클라 연결/해제
    data_transmitted = pyqtSignal(dict)    # sent/received
    error_occurred = pyqtSignal(str)       # 오류 발생


# ---- TCP Sender (8080 송신) ----
class TCPSender(threading.Thread):
    def __init__(self, transmission_signal: TransmissionSignal, data_queue: Queue,
                 host='127.0.0.1', port=8080):
        super().__init__(daemon=True)
        self.signal = transmission_signal
        self.q = data_queue
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self.running = False
        self.connected = False
        self.reconnect_interval = 3.0
        self.stats = {
            "total_sent": 0, "total_bytes": 0, "errors": 0,
            "last_transmission": None, "start_time": None
        }

    def connect(self):
        try:
            if self.sock:
                try: self.sock.close()
                except: pass
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
            self.sock.connect((self.host, self.port))
            self.connected = True
            self.signal.connection_status.emit({
                "type": "tcp_sender", "connected": True,
                "host": self.host, "port": self.port,
                "message": f"TCP 서버 연결됨: {self.host}:{self.port}"
            })
            return True
        except Exception as e:
            self.connected = False
            self.signal.error_occurred.emit(f"TCP 연결 실패: {e}")
            return False

    def disconnect(self):
        self.connected = False
        if self.sock:
            try: self.sock.close()
            except: pass
            self.sock = None
        self.signal.connection_status.emit({
            "type": "tcp_sender", "connected": False, "message": "TCP 연결 해제됨"
        })

    def send_data(self, data):
        if not (self.connected and self.sock):
            return False
        try:
            if isinstance(data, dict):
                package = {
                    "timestamp": datetime.now().isoformat(),
                    "source": "detector",
                    "data": data
                }
            else:
                package = {
                    "timestamp": datetime.now().isoformat(),
                    "source": "detector",
                    "raw_data": str(data)
                }
            payload = json.dumps(package, ensure_ascii=False, cls=NumpyEncoder).encode("utf-8")
            length = len(payload).to_bytes(4, "big")
            self.sock.sendall(length)
            self.sock.sendall(payload)

            # 통계 업데이트
            self.stats["total_sent"] += 1
            self.stats["total_bytes"] += len(payload)
            self.stats["last_transmission"] = datetime.now()

            self.signal.data_transmitted.emit({
                "direction": "sent", "size": len(payload),
                "timestamp": datetime.now().isoformat(),
                "data_type": type(data).__name__
            })
            return True
        except Exception as e:
            self.stats["errors"] += 1
            self.signal.error_occurred.emit(f"데이터 전송 실패: {e}")
            self.disconnect()
            return False

    def run(self):
        self.running = True
        self.stats["start_time"] = datetime.now()
        self.signal.connection_status.emit({
            "type": "tcp_sender", "status": "starting", "message": "TCP 송신 모듈 시작됨"
        })
        while self.running:
            if not self.connected:
                if not self.connect():
                    import time; time.sleep(self.reconnect_interval)
                    continue
            try:
                data = self.q.get(timeout=1.0)
                if data is not None:
                    self.send_data(data)
            except Empty:
                continue
            except Exception as e:
                self.signal.error_occurred.emit(f"송신 루프 오류: {e}")
        self.disconnect()

    def stop(self):
        self.running = False
        self.disconnect()


# ---- TCP Receiver (8081 수신) ----
class TCPReceiver(threading.Thread):
    def __init__(self, transmission_signal: TransmissionSignal, received_data_queue: Queue,
                 host='0.0.0.0', port=8081):
        super().__init__(daemon=True)
        self.signal = transmission_signal
        self.q = received_data_queue
        self.host = host
        self.port = port
        self.server_socket: socket.socket | None = None
        self.client_sockets: list[socket.socket] = []
        self.running = False
        self.stats = {
            "total_received": 0, "total_bytes": 0,
            "client_count": 0, "total_connections": 0,
            "start_time": None, "last_received": None
        }

    def start_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.signal.connection_status.emit({
                "type": "tcp_receiver", "status": "listening",
                "host": self.host, "port": self.port,
                "message": f"TCP 서버 시작됨: {self.host}:{self.port}"
            })
            return True
        except Exception as e:
            self.signal.error_occurred.emit(f"TCP 서버 시작 실패: {e}")
            return False

    def stop_server(self):
        for s in self.client_sockets[:]:
            try: s.close()
            except: pass
        self.client_sockets.clear()
        if self.server_socket:
            try: self.server_socket.close()
            except: pass
            self.server_socket = None
        self.signal.connection_status.emit({
            "type": "tcp_receiver", "status": "stopped",
            "message": "TCP 서버 중지됨"
        })

    def handle_client(self, csock: socket.socket, addr):
        info = f"{addr[0]}:{addr[1]}"
        try:
            self.signal.connection_status.emit({
                "type": "tcp_receiver", "client_connected": True,
                "client_info": info, "message": f"클라이언트 연결됨: {info}"
            })
            while self.running:
                length_b = csock.recv(4)
                if not length_b:
                    break
                msg_len = int.from_bytes(length_b, "big")
                data = b""
                while len(data) < msg_len:
                    chunk = csock.recv(msg_len - len(data))
                    if not chunk:
                        break
                    data += chunk
                if len(data) != msg_len:
                    break

                try:
                    obj = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError as e:
                    self.signal.error_occurred.emit(f"JSON 파싱 오류: {e}")
                    continue

                self.q.put({
                    "source": info,
                    "timestamp": datetime.now().isoformat(),
                    "data": obj
                })

                self.stats["total_received"] += 1
                self.stats["total_bytes"] += len(data)
                self.stats["last_received"] = datetime.now()

                self.signal.data_transmitted.emit({
                    "direction": "received", "size": len(data),
                    "client": info, "timestamp": datetime.now().isoformat()
                })

        except ConnectionResetError:
            pass
        except Exception as e:
            if self.running:
                self.signal.error_occurred.emit(f"클라이언트 처리 오류: {e}")
        finally:
            try: csock.close()
            except: pass
            if csock in self.client_sockets:
                self.client_sockets.remove(csock)
                self.stats["client_count"] = max(0, self.stats["client_count"] - 1)
            self.signal.connection_status.emit({
                "type": "tcp_receiver", "client_connected": False,
                "client_info": info, "message": f"클라이언트 연결 해제됨: {info}"
            })

    def run(self):
        self.running = True
        self.stats["start_time"] = datetime.now()
        if not self.start_server():
            return
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                csock, addr = self.server_socket.accept()
                self.client_sockets.append(csock)
                self.stats["client_count"] += 1
                self.stats["total_connections"] += 1
                th = threading.Thread(target=self.handle_client, args=(csock, addr), daemon=True)
                th.start()
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                if self.running:
                    self.signal.error_occurred.emit(f"서버 실행 오류: {e}")
                break
        self.stop_server()

    def stop(self):
        self.running = False
        try:
            if self.server_socket:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.2)
                    s.connect((self.host, self.port))
        except Exception:
            pass
        self.stop_server()