import socket
import json
import time
import random
from datetime import datetime

# 이 파일은 다른 파일(transmission.py 등)에 의존하지 않고 독립적으로 실행됩니다.

def send_data_packet(sock, data_payload):
    """
    데이터 페이로드를 받아 완전한 패키지로 감싸고, 
    길이 헤더와 함께 전송하는 함수.
    """
    try:
        # 1. 최종 전송될 전체 패키지 형태를 만듭니다.
        #    실제 송신 측에서 보내는 데이터 구조와 최대한 유사하게 만듭니다.
        package_to_send = {
            "timestamp": datetime.now().isoformat(),
            "source": "test_sender (가짜 클라이언트)",
            "data": data_payload  # 핵심 데이터를 'data' 키 안에 넣습니다.
        }

        # 2. 패키지를 JSON 문자열로 변환하고, UTF-8 바이트로 인코딩합니다.
        payload_bytes = json.dumps(package_to_send, ensure_ascii=False).encode("utf-8")
        
        # 3. 데이터의 길이를 계산하여 4바이트 길이의 헤더를 만듭니다.
        length_bytes = len(payload_bytes).to_bytes(4, "big")

        # 4. 길이 헤더 먼저 전송
        sock.sendall(length_bytes)
        # 5. 실제 데이터 전송
        sock.sendall(payload_bytes)
        
        # 터미널에 성공 메시지를 출력합니다.
        print(f"[{time.strftime('%H:%M:%S')}] 데이터 전송 성공: {data_payload}")
        return True

    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
        print("연결이 끊어졌습니다. 재연결을 시도합니다.")
        return False
    except Exception as e:
        print(f"데이터 전송 중 오류 발생: {e}")
        return False

def generate_fake_data():
    """히트맵을 포함한 가짜 센서 데이터를 생성하는 함수"""
    
    # 1. 기본 상태 및 온도 데이터 생성
    temp = round(random.uniform(22.0, 45.0), 2)
    status = "정상"
    if temp > 40.0:
        status = "화재"
    elif temp > 35.0:
        status = "주의"

    # 2. 8x8 히트맵 데이터 생성
    grid_size = 8
    heatmap_values = [round(temp + random.uniform(-2.0, 2.0), 2) for _ in range(grid_size * grid_size)]
    # 중앙 부근에 더 높은 온도를 추가하여 변화를 잘 보이게 함
    hot_spot_index = random.randint(27, 36) # 중앙 부근 인덱스
    heatmap_values[hot_spot_index] += random.uniform(5.0, 10.0)

    # 3. 최종 데이터 페이로드 구성
    #    이 구조가 main.py의 detector.py와 gui.py가 기대하는 데이터 구조입니다.
    fake_data = {
        "sensor_id": 1, # main.py의 TCP_PORT_SENSOR_MAP에 따라 8081은 2번 센서
        "status": status,
        "temp": temp, # 개별 온도 값 (필요 시 사용)
        "grid": {
            "h": grid_size,
            "w": grid_size,
            "values": heatmap_values
        }
    }
    return fake_data


def main():
    # 접속할 서버의 주소. '127.0.0.1'은 '내 컴퓨터 자신'을 의미합니다.
    HOST = '127.0.0.1'  
    # main.py의 TCPReceiver가 기다리고 있는 포트 번호
    PORT = 8081         

    current_socket = None

    while True: # 프로그램이 끝나지 않고 계속 재연결을 시도하도록 무한 루프
        try:
            # 1. 소켓 생성 및 서버에 연결
            print(f"{HOST}:{PORT} 에 연결을 시도합니다...")
            current_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            current_socket.connect((HOST, PORT))
            print("✅ 연결 성공! 1.5초마다 데이터를 전송합니다.")
            
            # 2. 연결이 유지되는 동안 데이터 반복 전송
            while True:
                # 가짜 데이터 생성
                fake_data = generate_fake_data()

                # 데이터 전송 시도
                if not send_data_packet(current_socket, fake_data):
                    break # 전송 실패 시, 내부 루프를 빠져나가 재연결 시도
                
                # 1.5초 대기
                time.sleep(1.5)

        except ConnectionRefusedError:
            print("❌ 연결이 거부되었습니다. 서버(main.py)가 실행 중인지, 방화벽을 확인하세요.")
        except Exception as e:
            print(f"알 수 없는 오류 발생: {e}")
        finally:
            # 연결이 끊어지면 소켓을 닫습니다.
            if current_socket:
                current_socket.close()
        
        print("\n5초 후 다시 연결을 시도합니다...\n")
        time.sleep(5)

if __name__ == '__main__':
    main()