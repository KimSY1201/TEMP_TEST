#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
receiver_app.py
TCP 데이터 수신 전용 애플리케이션
detector 모듈에서 보내는 데이터를 수신하여 처리
"""

import sys
import time
import json
from datetime import datetime
from queue import Queue, Empty
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QPushButton, QLabel, QSpinBox, QGroupBox, QGridLayout
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# 송수신 모듈 import
from transmission import TCPReceiver, TransmissionSignal


class DataDisplayWidget(QWidget):
    """수신된 데이터를 표시하는 위젯"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # 수신된 데이터 저장
        self.received_data = []
        self.max_display_items = 100
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 제목
        title = QLabel("수신된 Detector 데이터")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # 데이터 표시 영역
        self.data_display = QTextEdit()
        self.data_display.setFont(QFont("Consolas", 9))
        self.data_display.setReadOnly(True)
        layout.addWidget(self.data_display)
        
        # 제어 버튼들
        button_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("화면 지우기")
        self.clear_button.clicked.connect(self.clear_display)
        button_layout.addWidget(self.clear_button)
        
        self.save_button = QPushButton("데이터 저장")
        self.save_button.clicked.connect(self.save_data)
        button_layout.addWidget(self.save_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def add_received_data(self, data_info):
        """수신된 데이터 추가"""
        self.received_data.append(data_info)
        
        # 최대 표시 개수 제한
        if len(self.received_data) > self.max_display_items:
            self.received_data.pop(0)
        
        self.update_display()
    
    def update_display(self):
        """화면 업데이트"""
        self.data_display.clear()
        
        for i, data_info in enumerate(reversed(self.received_data[-50:])):  # 최근 50개만 표시
            timestamp = data_info.get('timestamp', '')
            source = data_info.get('source', 'unknown')
            data = data_info.get('data', {})
            
            # 시간 포맷팅
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp[:8] if len(timestamp) > 8 else timestamp
            
            # 데이터 내용 추출
            if isinstance(data, dict):
                if 'data' in data and isinstance(data['data'], dict):
                    # detector 모듈 데이터 구조 처리
                    detector_data = data['data']
                    content_parts = []
                    
                    # 주요 데이터 필드들 표시
                    for key, value in detector_data.items():
                        if key in ['detected_value', 'threshold', 'status', 'raw_data']:
                            content_parts.append(f"{key}={value}")
                    
                    content = ", ".join(content_parts) if content_parts else str(detector_data)
                else:
                    content = str(data)[:100]  # 첫 100자만
            else:
                content = str(data)[:100]
            
            # 한 줄로 표시
            line = f"[{time_str}] {source} → {content}"
            if len(line) > 150:
                line = line[:147] + "..."
            
            self.data_display.append(line)
        
        # 자동 스크롤 (최신 데이터가 아래에)
        cursor = self.data_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.data_display.setTextCursor(cursor)
    
    def clear_display(self):
        """표시 화면 지우기"""
        self.received_data.clear()
        self.data_display.clear()
    
    def save_data(self):
        """수신된 데이터를 파일로 저장"""
        if not self.received_data:
            return
        
        filename = f"received_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.received_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.data_display.append(f"\n데이터 저장 완료: {filename}\n")
            
        except Exception as e:
            self.data_display.append(f"\n데이터 저장 실패: {str(e)}\n")


class StatusWidget(QWidget):
    """수신 상태를 표시하는 위젯"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # 통계 정보
        self.stats = {
            'total_received': 0,
            'total_bytes': 0,
            'client_count': 0,
            'start_time': None,
            'last_received': None
        }
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 상태 그룹박스
        status_group = QGroupBox("수신 상태")
        status_layout = QGridLayout()
        
        self.status_label = QLabel("대기 중...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        status_layout.addWidget(QLabel("서버 상태:"), 0, 0)
        status_layout.addWidget(self.status_label, 0, 1)
        
        self.port_label = QLabel("-")
        status_layout.addWidget(QLabel("수신 포트:"), 1, 0)
        status_layout.addWidget(self.port_label, 1, 1)
        
        self.client_label = QLabel("0")
        status_layout.addWidget(QLabel("연결된 클라이언트:"), 2, 0)
        status_layout.addWidget(self.client_label, 2, 1)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 통계 그룹박스
        stats_group = QGroupBox("수신 통계")
        stats_layout = QGridLayout()
        
        self.received_label = QLabel("0")
        stats_layout.addWidget(QLabel("수신 패킷:"), 0, 0)
        stats_layout.addWidget(self.received_label, 0, 1)
        
        self.bytes_label = QLabel("0")
        stats_layout.addWidget(QLabel("수신 바이트:"), 1, 0)
        stats_layout.addWidget(self.bytes_label, 1, 1)
        
        self.uptime_label = QLabel("-")
        stats_layout.addWidget(QLabel("가동 시간:"), 2, 0)
        stats_layout.addWidget(self.uptime_label, 2, 1)
        
        self.last_received_label = QLabel("-")
        stats_layout.addWidget(QLabel("마지막 수신:"), 3, 0)
        stats_layout.addWidget(self.last_received_label, 3, 1)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_connection_status(self, connected, port=None, message=""):
        """연결 상태 업데이트"""
        if connected:
            self.status_label.setText("수신 중")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            if port:
                self.port_label.setText(str(port))
        else:
            self.status_label.setText("중지됨")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.client_label.setText("0")
    
    def update_client_count(self, count):
        """클라이언트 수 업데이트"""
        self.client_label.setText(str(count))
    
    def update_statistics(self, stats):
        """통계 정보 업데이트"""
        self.stats.update(stats)
        
        self.received_label.setText(str(self.stats.get('total_received', 0)))
        self.bytes_label.setText(f"{self.stats.get('total_bytes', 0):,}")
        
        # 가동 시간 계산
        if self.stats.get('start_time'):
            try:
                if isinstance(self.stats['start_time'], str):
                    start_time = datetime.fromisoformat(self.stats['start_time'])
                else:
                    start_time = self.stats['start_time']
                
                uptime = datetime.now() - start_time
                uptime_str = str(uptime).split('.')[0]  # 초 단위 제거
                self.uptime_label.setText(uptime_str)
            except:
                self.uptime_label.setText("-")
        
        # 마지막 수신 시간
        if self.stats.get('last_received'):
            try:
                last_time = self.stats['last_received']
                if isinstance(last_time, str):
                    last_time = datetime.fromisoformat(last_time)
                time_str = last_time.strftime('%H:%M:%S')
                self.last_received_label.setText(time_str)
            except:
                self.last_received_label.setText("-")


class ReceiverMainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # 수신 모듈 관련
        self.transmission_signal = TransmissionSignal()
        self.received_data_queue = Queue()
        self.receiver = None
        self.port = 8080
        
        # 타이머들
        self.data_check_timer = QTimer()
        self.data_check_timer.timeout.connect(self.check_received_data)
        self.data_check_timer.start(100)  # 100ms마다 체크
        
        self.stats_update_timer = QTimer()
        self.stats_update_timer.timeout.connect(self.update_statistics)
        self.stats_update_timer.start(1000)  # 1초마다 통계 업데이트
        
        # 시그널 연결
        self.setup_signals()
        
        # 자동 시작
        self.start_receiver()
    
    def setup_ui(self):
        self.setWindowTitle("TCP 데이터 수신기")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout()
        
        # 왼쪽: 데이터 표시
        self.data_widget = DataDisplayWidget()
        layout.addWidget(self.data_widget, 3)  # 3배 크기
        
        # 오른쪽: 상태 및 제어
        right_layout = QVBoxLayout()
        
        # 포트 설정
        port_group = QGroupBox("설정")
        port_layout = QGridLayout()
        
        port_layout.addWidget(QLabel("수신 포트:"), 0, 0)
        self.port_spinbox = QSpinBox()
        self.port_spinbox.setRange(1024, 65535)
        self.port_spinbox.setValue(8080)
        self.port_spinbox.valueChanged.connect(self.on_port_changed)
        port_layout.addWidget(self.port_spinbox, 0, 1)
        
        self.start_button = QPushButton("시작")
        self.start_button.clicked.connect(self.start_receiver)
        port_layout.addWidget(self.start_button, 1, 0)
        
        self.stop_button = QPushButton("중지")
        self.stop_button.clicked.connect(self.stop_receiver)
        port_layout.addWidget(self.stop_button, 1, 1)
        
        port_group.setLayout(port_layout)
        right_layout.addWidget(port_group)
        
        # 상태 위젯
        self.status_widget = StatusWidget()
        right_layout.addWidget(self.status_widget)
        
        right_container = QWidget()
        right_container.setLayout(right_layout)
        right_container.setMaximumWidth(300)
        layout.addWidget(right_container, 1)  # 1배 크기
        
        central_widget.setLayout(layout)
    
    def setup_signals(self):
        """시그널 연결 설정"""
        self.transmission_signal.connection_status.connect(self.on_connection_status_changed)
        self.transmission_signal.data_transmitted.connect(self.on_data_received_signal)
        self.transmission_signal.error_occurred.connect(self.on_error_occurred)
    
    def on_port_changed(self, port):
        """포트 변경"""
        self.port = port
    
    def start_receiver(self):
        """수신 시작"""
        if self.receiver and self.receiver.is_alive():
            self.stop_receiver()
        
        try:
            self.receiver = TCPReceiver(
                self.transmission_signal,
                self.received_data_queue,
                host='0.0.0.0',
                port=self.port
            )
            self.receiver.start()
            
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.port_spinbox.setEnabled(False)
            
            self.data_widget.data_display.append(f"\n=== TCP 수신 시작 (포트: {self.port}) ===\n")
            
        except Exception as e:
            self.data_widget.data_display.append(f"\n수신 시작 실패: {str(e)}\n")
    
    def stop_receiver(self):
        """수신 중지"""
        if self.receiver:
            self.receiver.stop()
            self.receiver.join(timeout=3)
            self.receiver = None
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.port_spinbox.setEnabled(True)
        
        self.status_widget.update_connection_status(False)
        self.data_widget.data_display.append(f"\n=== TCP 수신 중지 ===\n")
    
    def check_received_data(self):
        """수신된 데이터 확인"""
        while not self.received_data_queue.empty():
            try:
                data_info = self.received_data_queue.get_nowait()
                self.data_widget.add_received_data(data_info)
            except Empty:
                break
    
    def update_statistics(self):
        """통계 업데이트"""
        if self.receiver:
            stats = self.receiver.stats.copy()
            stats['start_time'] = self.receiver.stats.get('start_time')
            self.status_widget.update_statistics(stats)
    
    def on_connection_status_changed(self, status_info):
        """연결 상태 변경"""
        if status_info.get('type') == 'tcp_receiver':
            if 'client_connected' in status_info:
                # 클라이언트 연결/해제
                client_info = status_info.get('client_info', '')
                connected = status_info.get('client_connected', False)
                message = status_info.get('message', '')
                
                self.data_widget.data_display.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
                
                # 클라이언트 수 업데이트
                if self.receiver:
                    self.status_widget.update_client_count(len(self.receiver.client_sockets))
            else:
                # 서버 상태 변경
                message = status_info.get('message', '')
                if 'TCP 서버 시작됨' in message:
                    self.status_widget.update_connection_status(True, self.port)
                elif 'TCP 서버 중지됨' in message:
                    self.status_widget.update_connection_status(False)
    
    def on_data_received_signal(self, transmission_info):
        """데이터 수신 시그널"""
        if transmission_info.get('direction') == 'received':
            # 통계 업데이트는 별도 타이머에서 처리
            pass
    
    def on_error_occurred(self, error_message):
        """오류 발생"""
        self.data_widget.data_display.append(f"\n[ERROR] {error_message}\n")
    
    def closeEvent(self, event):
        """창 닫기"""
        self.stop_receiver()
        event.accept()


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 애플리케이션 스타일
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 8px 16px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 12px;
            margin: 2px 1px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QTextEdit {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
        }
    """)
    
    window = ReceiverMainWindow()
    window.show()
    
    print("TCP 데이터 수신기가 시작되었습니다.")
    print(f"포트 8080에서 detector 데이터를 수신 대기 중입니다.")
    print("송신측에서 데이터를 보내면 실시간으로 표시됩니다.")
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()