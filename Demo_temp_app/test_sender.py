import socket
import json
import time
import random
from datetime import datetime

# 이 파일은 다른 파일에 의존하지 않고 독립적으로 실행됩니다.

def send_data_packet(sock, data_payload):
    """
    [수정] 받은 데이터 페이로드를 바로 JSON으로 변환하여 전송합니다.
    (더 이상 'data' 키로 감싸지 않습니다.)
    """
    try:
        # 1. 패키지를 JSON 문자열로 변환하고, UTF-8 바이트로 인코딩합니다.
        payload_bytes = json.dumps(data_payload, ensure_ascii=False).encode("utf-8")
        
        # 2. 데이터의 길이를 계산하여 4바이트 길이의 헤더를 만듭니다.
        length_bytes = len(payload_bytes).to_bytes(4, "big")

        # 3. 길이 헤더와 실제 데이터를 차례로 전송합니다.
        sock.sendall(length_bytes)
        sock.sendall(payload_bytes)
        
        # 터미널에 성공 메시지를 출력합니다. (데이터가 너무 길므로 앞부분만 표시)
        print(f"[{time.strftime('%H:%M:%S')}] 데이터 전송 성공: {str(data_payload)[:100]}...")
        return True

    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
        print("연결이 끊어졌습니다. 재연결을 시도합니다.")
        return False
    except Exception as e:
        print(f"데이터 전송 중 오류 발생: {e}")
        return False

def generate_fake_data():
    """
    [수정] 실제 수신 데이터와 동일한, 단순한 형태의 데이터를 생성합니다.
    """
    # 1. 8x8 (64개)의 랜덤 온도 값을 생성합니다.
    grid_size = 8
    base_temp = round(random.uniform(22.0, 35.0), 2)
    heatmap_values = [round(base_temp + random.uniform(-2.0, 2.0), 2) for _ in range(grid_size * grid_size)]

    # 2. 실제 데이터와 동일한, 단순한 딕셔너리 형태로 구성합니다.
    fake_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "values": heatmap_values
    }
    return fake_data


def main():
    HOST = '127.0.0.1'  # 내 컴퓨터 자신
    PORT = 8080         # main.py가 기다리고 있는 포트

    current_socket = None

    while True:
        try:
            print(f"{HOST}:{PORT} 에 연결을 시도합니다...")
            current_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            current_socket.connect((HOST, PORT))
            print("연결 성공! 1.5초마다 데이터를 전송합니다.")
            
            while True:
                # 가짜 데이터 생성
                fake_data = generate_fake_data()

                # 데이터 전송
                if not send_data_packet(current_socket, fake_data):
                    break
                
                time.sleep(1.5)

        except ConnectionRefusedError:
            print("연결이 거부되었습니다. 서버(main.py)가 실행 중인지, 방화벽을 확인하세요.")
        except Exception as e:
            print(f"알 수 없는 오류 발생: {e}")
        finally:
            if current_socket:
                current_socket.close()
        
        print("\n5초 후 다시 연결을 시도합니다...\n")
        time.sleep(5)

if __name__ == '__main__':
    main()