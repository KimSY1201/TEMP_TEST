#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transmission.py
TCP/IP 송수신 모듈
detector 모듈의 데이터를 TCP/IP로 전송하고 상태를 모니터링
"""
import numpy as np
import socket
import threading
import time
import json
from datetime import datetime
from queue import Queue, Empty
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

class NumpyEncoder(json.JSONEncoder):
    """NumPy 타입을 JSON으로 변환하기 위한 커스텀 인코더"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # ndarray는 list로 변환
        if isinstance(obj, np.integer):
            return int(obj)      # NumPy 정수형은 Python int로 변환
        if isinstance(obj, np.floating):
            return float(obj)    # NumPy 실수형은 Python float로 변환
        return super(NumpyEncoder, self).default(obj)
    
    
class TransmissionSignal(QObject):
    """전송 모듈 관련 시그널"""
    status_update = pyqtSignal(dict)  # 전송 상태 업데이트
    connection_status = pyqtSignal(dict)  # 연결 상태 변경
    data_transmitted = pyqtSignal(dict)  # 데이터 전송 완료
    error_occurred = pyqtSignal(str)  # 오류 발생


class TCPSender(threading.Thread):
    """TCP 클라이언트 송신 모듈"""
    
    def __init__(self, transmission_signal, data_queue, host='localhost', port=8080):
        super().__init__(daemon=True)
        self.transmission_signal = transmission_signal
        self.data_queue = data_queue  # detector에서 받을 데이터 큐
        
        # TCP 설정
        self.host = host
        self.port = port
        self.socket = None
        
        # 제어 변수
        self.running = False
        self.connected = False
        self.reconnect_interval = 3.0
        
        # 전송 통계
        self.stats = {
            'total_sent': 0,
            'total_bytes': 0,
            'errors': 0,
            'last_transmission': None,
            'start_time': None
        }
    
    def connect(self):
        """TCP 서버에 연결"""
        try:
            if self.socket:
                self.socket.close()
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            
            self.connected = True
            self.transmission_signal.connection_status.emit({
                'type': 'tcp_sender',
                'connected': True,
                'host': self.host,
                'port': self.port,
                'message': f'TCP 서버 연결됨: {self.host}:{self.port}'
            })
            return True
            
        except Exception as e:
            self.connected = False
            self.transmission_signal.error_occurred.emit(f'TCP 연결 실패: {str(e)}')
            return False
    
    def disconnect(self):
        """TCP 연결 해제"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self.transmission_signal.connection_status.emit({
            'type': 'tcp_sender',
            'connected': False,
            'message': 'TCP 연결 해제됨'
        })
    
    def send_data(self, data):
        """데이터 전송"""
        if not self.connected or not self.socket:
            return False
        
        try:
            # 데이터를 JSON 형태로 패키징
            if isinstance(data, dict):
                # detector 모듈에서 오는 데이터 구조 처리
                package = {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'detector',
                    'data': data
                }
            else:
                package = {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'detector',
                    'raw_data': str(data)
                }
            
            # JSON 직렬화 및 전송
            json_data = json.dumps(package, ensure_ascii=False, cls=NumpyEncoder)
            message = json_data.encode('utf-8')
            
            # 메시지 길이를 먼저 전송 (4바이트)
            length = len(message)
            self.socket.sendall(length.to_bytes(4, byteorder='big'))
            self.socket.sendall(message)
            
            # 통계 업데이트
            self.stats['total_sent'] += 1
            self.stats['total_bytes'] += len(message)
            self.stats['last_transmission'] = datetime.now()
            
            # 전송 완료 시그널
            self.transmission_signal.data_transmitted.emit({
                'direction': 'sent',
                'size': len(message),
                'timestamp': datetime.now().isoformat(),
                'data_type': type(data).__name__
            })
            
            return True
            
        except Exception as e:
            self.stats['errors'] += 1
            self.transmission_signal.error_occurred.emit(f'데이터 전송 실패: {str(e)}')
            self.disconnect()
            return False
    
    def run(self):
        """메인 실행 루프"""
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        self.transmission_signal.connection_status.emit({
            'type': 'tcp_sender',
            'status': 'starting',
            'message': 'TCP 송신 모듈 시작됨'
        })
        
        while self.running:
            try:
                # 연결되지 않은 경우 재연결 시도
                if not self.connected:
                    if self.connect():
                        time.sleep(1)  # 연결 후 잠시 대기
                    else:
                        time.sleep(self.reconnect_interval)
                        continue
                
                # 데이터 큐에서 전송할 데이터 확인
                try:
                    data = self.data_queue.get(timeout=1.0)
                    if data is not None:
                        self.send_data(data)
                except Empty:
                    continue  # 타임아웃, 계속 진행
                
            except Exception as e:
                self.stats['errors'] += 1
                self.transmission_signal.error_occurred.emit(f'송신 루프 오류: {str(e)}')
                time.sleep(1)
        
        # 종료시 연결 해제
        self.disconnect()
    
    def stop(self):
        """송신 모듈 중지"""
        self.running = False
        self.disconnect()


class TCPReceiver(threading.Thread):
    """TCP 서버 수신 모듈"""
    
    def __init__(self, transmission_signal, received_data_queue, host='0.0.0.0', port=8081):
        super().__init__(daemon=True)
        self.transmission_signal = transmission_signal
        self.received_data_queue = received_data_queue  # 수신된 데이터를 담을 큐
        
        # TCP 서버 설정
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_sockets = []
        
        # 제어 변수
        self.running = False
        
        # 수신 통계
        self.stats = {
            'total_received': 0,
            'total_bytes': 0,
            'client_count': 0,
            'total_connections': 0,
            'start_time': None
        }
    
    def start_server(self):
        """TCP 서버 시작"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.transmission_signal.connection_status.emit({
                'type': 'tcp_receiver',
                'status': 'listening',
                'host': self.host,
                'port': self.port,
                'message': f'TCP 서버 시작됨: {self.host}:{self.port}'
            })
            
            return True
            
        except Exception as e:
            self.transmission_signal.error_occurred.emit(f'TCP 서버 시작 실패: {str(e)}')
            return False
    
    def handle_client(self, client_socket, client_addr):
        """개별 클라이언트 처리"""
        client_info = f"{client_addr[0]}:{client_addr[1]}"
        
        try:
            self.transmission_signal.connection_status.emit({
                'type': 'tcp_receiver',
                'client_connected': True,
                'client_info': client_info,
                'message': f'클라이언트 연결됨: {client_info}'
            })
            
            while self.running:
                try:
                    # 메시지 길이 수신 (4바이트)
                    length_data = client_socket.recv(4)
                    if not length_data:
                        break
                    
                    message_length = int.from_bytes(length_data, byteorder='big')
                    
                    # 실제 메시지 수신
                    message_data = b''
                    while len(message_data) < message_length:
                        chunk = client_socket.recv(message_length - len(message_data))
                        if not chunk:
                            break
                        message_data += chunk
                    
                    if len(message_data) == message_length:
                        # JSON 데이터 파싱
                        try:
                            json_str = message_data.decode('utf-8')
                            data = json.loads(json_str)
                            
                            # 수신된 데이터를 큐에 추가
                            self.received_data_queue.put({
                                'source': client_info,
                                'timestamp': datetime.now().isoformat(),
                                'data': data
                            })
                            
                            # 통계 업데이트
                            self.stats['total_received'] += 1
                            self.stats['total_bytes'] += len(message_data)
                            
                            # 수신 완료 시그널
                            self.transmission_signal.data_transmitted.emit({
                                'direction': 'received',
                                'size': len(message_data),
                                'client': client_info,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                        except json.JSONDecodeError as e:
                            self.transmission_signal.error_occurred.emit(f'JSON 파싱 오류: {str(e)}')
                    
                except socket.timeout:
                    continue
                except ConnectionResetError:
                    break
                except Exception as e:
                    self.transmission_signal.error_occurred.emit(f'클라이언트 처리 오류: {str(e)}')
                    break
                    
        finally:
            # 클라이언트 연결 해제
            try:
                client_socket.close()
                if client_socket in self.client_sockets:
                    self.client_sockets.remove(client_socket)
                self.stats['client_count'] -= 1
                
                self.transmission_signal.connection_status.emit({
                    'type': 'tcp_receiver',
                    'client_connected': False,
                    'client_info': client_info,
                    'message': f'클라이언트 연결 해제됨: {client_info}'
                })
                
            except:
                pass
    
    def run(self):
        """메인 실행 루프"""
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        if not self.start_server():
            return
        
        while self.running:
            try:
                # 클라이언트 접속 대기
                self.server_socket.settimeout(1.0)  # 1초 타임아웃
                client_socket, client_addr = self.server_socket.accept()
                
                self.client_sockets.append(client_socket)
                self.stats['client_count'] += 1
                self.stats['total_connections'] += 1
                
                # 클라이언트 처리 스레드 시작
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_addr),
                    daemon=True
                )
                client_thread.start()
                
            except socket.timeout:
                continue  # 타임아웃, 계속 진행
            except Exception as e:
                if self.running:
                    self.transmission_signal.error_occurred.emit(f'서버 실행 오류: {str(e)}')
                break
        
        # 종료시 모든 연결 해제
        self.stop_server()
    
    def stop_server(self):
        """TCP 서버 중지"""
        # 모든 클라이언트 연결 해제
        for client_socket in self.client_sockets[:]:
            try:
                client_socket.close()
            except:
                pass
        self.client_sockets.clear()
        
        # 서버 소켓 해제
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        self.transmission_signal.connection_status.emit({
            'type': 'tcp_receiver',
            'status': 'stopped',
            'message': 'TCP 서버 중지됨'
        })
    
    def stop(self):
        """수신 모듈 중지"""
        self.running = False
        self.stop_server()


class TransmissionManager:
    """송수신 모듈을 통합 관리하는 클래스"""
    
    def __init__(self, transmission_signal):
        self.transmission_signal = transmission_signal
        
        # 데이터 큐들
        self.send_queue = Queue()  # detector -> TCP 송신
        self.receive_queue = Queue()  # TCP 수신 -> 다른 모듈
        
        # 송수신 모듈들
        self.sender = None
        self.receiver = None
        
        # 설정
        self.sender_config = {
            'host': '192.168.1.100',
            'port': 8080
        }
        self.receiver_config = {
            'host': '0.0.0.0',
            'port': 8080
        }
        
        # 상태 모니터링 타이머
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.setInterval(2000)  # 2초마다 상태 업데이트
    
    def configure_sender(self, host, port):
        """송신 모듈 설정"""
        self.sender_config['host'] = host
        self.sender_config['port'] = port
    
    def configure_receiver(self, host, port):
        """수신 모듈 설정"""
        self.receiver_config['host'] = host
        self.receiver_config['port'] = port
    
    def start_sender(self):
        """송신 모듈 시작"""
        if self.sender and self.sender.is_alive():
            self.stop_sender()
        
        self.sender = TCPSender(
            self.transmission_signal,
            self.send_queue,
            self.sender_config['host'],
            self.sender_config['port']
        )
        self.sender.start()
        return True
    
    def start_receiver(self):
        """수신 모듈 시작"""
        if self.receiver and self.receiver.is_alive():
            self.stop_receiver()
        
        self.receiver = TCPReceiver(
            self.transmission_signal,
            self.receive_queue,
            self.receiver_config['host'],
            self.receiver_config['port']
        )
        self.receiver.start()
        return True
    
    def stop_sender(self):
        """송신 모듈 중지"""
        if self.sender:
            self.sender.stop()
            self.sender.join(timeout=3)
            self.sender = None
    
    def stop_receiver(self):
        """수신 모듈 중지"""
        if self.receiver:
            self.receiver.stop()
            self.receiver.join(timeout=3)
            self.receiver = None
    
    def send_detector_data(self, data):
        """detector 모듈의 데이터를 전송 큐에 추가"""
        try:
            self.send_queue.put(data, timeout=1)
            return True
        except:
            return False
    
    def get_received_data(self):
        """수신된 데이터 반환 (논블로킹)"""
        try:
            return self.receive_queue.get_nowait()
        except:
            return None
    
    def update_status(self):
        """통합 상태 업데이트"""
        status = {
            'sender': {
                'running': self.sender.running if self.sender else False,
                'connected': self.sender.connected if self.sender else False,
                'stats': self.sender.stats.copy() if self.sender else {}
            },
            'receiver': {
                'running': self.receiver.running if self.receiver else False,
                'stats': self.receiver.stats.copy() if self.receiver else {}
            },
            'queue_sizes': {
                'send_queue': self.send_queue.qsize(),
                'receive_queue': self.receive_queue.qsize()
            }
        }
        
        self.transmission_signal.status_update.emit(status)
    
    def start_status_monitoring(self):
        """상태 모니터링 시작"""
        self.status_timer.start()
    
    def stop_status_monitoring(self):
        """상태 모니터링 중지"""
        self.status_timer.stop()
    
    def start_all(self):
        """모든 송수신 모듈 시작"""
        sender_ok = self.start_sender()
        receiver_ok = self.start_receiver()
        
        if sender_ok or receiver_ok:
            self.start_status_monitoring()
        
        return sender_ok, receiver_ok
    
    def stop_all(self):
        """모든 송수신 모듈 중지"""
        self.stop_status_monitoring()
        self.stop_sender()
        self.stop_receiver()