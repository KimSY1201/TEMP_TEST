import sys, time, random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QComboBox, QPushButton, QFrame, QSizePolicy, QSplitter
)

from signals import DataSignal

# === 1920x1080 해상도에 맞춰 폰트 크기 상향 조정 ===
GLOBAL_STYLESHEET = """
/* 전체 위젯 기본 스타일 */
QWidget {
    color: #E0E0E0;
    font-family: 'Malgun Gothic', 'Segoe UI', 'Arial';
}

/* 메인 윈도우 스타일 */
#MainWindow {
    background-color: #2E2E2E;
}

/* 정보 패널(카드) 스타일 */
#TopPanel, #LeftPanel {
    background-color: #080F25;
    border-radius: 8px;
    padding: 10px;
}

/* 연결 상태 패널 스타일 */
#ConnectionPanel {
    background-color: #4A4A4A;
    border-radius: 6px;
    padding: 8px;
    border: 2px solid #666666;
}

#ConnectionPanel[connected="true"] {
    border-color: #4CAF50;
    background-color: #2E4A2E;
}

#ConnectionPanel[connected="false"] {
    border-color: #F44336;
    background-color: #4A2E2E;
}

.Panel {
    background-color: #353535;
    border-color: #F44336;
    border-radius: 6px;
    padding: 8px;
}

/* 제목 레이블 스타일 */
QLabel[class="TitleLabel"] {
    font-size: 16pt;
    font-weight: bold;
    color: #FFFFFF;
    padding: 2px;
}

/* 일반 정보 레이블 스타일 */
QLabel[class="InfoLabel"] {
    font-size: 14pt;
    font-weight: bold;
}

/* 연결 상태 레이블 스타일 */
QLabel[class="ConnectionLabel"] {
    font-size: 12px;
    font-weight: bold;
}

/* 이상 감지 카운트 레이블 특별 스타일 */
#AnomalyCountLabel {
    font-size: 16px;
    font-weight: bold;
    color: #F44336;
    padding: 2px;
}

/* 히트맵 셀 스타일 */
QLabel[class="HeatmapCell"] {
    font-weight: bold;
    border: none;
}

/* 버튼 스타일 */
QPushButton {
    background-color: #555555;
    color: #FFFFFF;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #6A6A6A;
}
QPushButton:pressed {
    background-color: #4A4A4A;
}

/* 연결 버튼 특별 스타일 */
QPushButton[buttonType="connect"] {
    background-color: #4CAF50;
}
QPushButton[buttonType="connect"]:hover {
    background-color: #5CBF60;
}

QPushButton[buttonType="disconnect"] {
    background-color: #F44336;
}
QPushButton[buttonType="disconnect"]:hover {
    background-color: #FF5346;
}

/* 온도 조절 버튼 작은 스타일 */
QPushButton[class="AdjustButton"] {
    font-size: 12px;
    font-weight: bold;
    padding: 4px 8px;
    min-width: 30px;
}

/* QComboBox 스타일 */
QComboBox {
    background-color: #2E2E2E;
    color: #FFFFFF;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 5px;
    font-size: 12px;
    font-weight: bold;
    min-width: 80px;
}
QComboBox::drop-down {
    border: none;
}
QComboBox::down-arrow {
    border: none;
}

/* QDoubleSpinBox 스타일 */
QDoubleSpinBox {
    background-color: #2E2E2E;
    color: #FFFFFF;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 5px;
    font-size: 12px;
    font-weight: bold;
}

/* 상태 표시기(LED) 스타일 */
QLabel[styleClass="Indicator"] {
    min-width: 20px;
    max-width: 20px;
    min-height: 20px;
    max-height: 20px;
    border-radius: 10px;
    border: 1px solid rgba(0, 0, 0, 100);
}

/* 상태에 따른 색상 */
QLabel[styleClass="Indicator"][state="stable"] {
    background-color: qlineargradient(spread:pad, x1:0.145, y1:0.16, x2:0.92, y2:0.915,
                                     stop:0 rgba(0, 255, 0, 255),
                                     stop:1 rgba(0, 128, 0, 255));
}
QLabel[styleClass="Indicator"][state="detected"] {
    background-color: qlineargradient(spread:pad, x1:0.145, y1:0.16, x2:0.92, y2:0.915,
                                     stop:0 rgba(255, 0, 0, 255),
                                     stop:1 rgba(128, 0, 0, 255));
}

.temp_widget { 
                    background-color: #37446B;
                    border: 1px solid #1D2439;
                    border-radius: 5px;
                    padding: 5px 10px;
                    }
                
                
#safety {
    border: 3px solid green;
}
#caution {
    border: 3px solid yellow;
}
#danger {
    border: 3px solid red;
}
"""

def clamp(v, a, b): return a if v < a else (b if v > b else v)

def status_colors(s: str):
    s = (s or "").lower()
    if s in ("화재", "fire"): return QColor("#F44336"), QColor("#FFFFFF")
    if s in ("주의", "warn", "warning"): return QColor("#FFEB3B"), QColor("#222222")
    return QColor("#4CAF50"), QColor("#FFFFFF")  # 정상

# -------------------- 카운트 배지 --------------------
class CountPill(QLabel):
    def __init__(self, label: str, bg: QColor, fg: QColor):
        super().__init__(f"{label} 0")
        self.bg, self.fg = bg, fg
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(88)
        self.setStyleSheet(
            f"border-radius:16px; padding:6px 12px; "
            f"background:{bg.name()}; color:{fg.name()}; font-weight:800;"
        )

    def set_count(self, n: int):
        name = self.text().split()[0]
        self.setText(f"{name} {n}")

# -------------------- 센서 카드 --------------------
class SensorCard(QFrame):
    def __init__(self, name: str):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self.t = QLabel(name); self.t.setStyleSheet("font-weight:800;")
        self.s = QLabel("대기 중"); self.w = QLabel("-")
        lay = QVBoxLayout(self); lay.setContentsMargins(14,10,14,10); lay.setSpacing(4)
        lay.addWidget(self.t); lay.addWidget(self.s); lay.addWidget(self.w)
        self.apply("정상")

    def apply(self, status: str, ts: Optional[float]=None):
        self.s.setText("화재 발생" if status in ("화재","fire") else ("주의" if status in ("주의","warn","warning") else "정상"))
        self.w.setText(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts or time.time())))
        bg, fg = status_colors(status)
        pal = self.palette(); pal.setColor(QPalette.ColorRole.Window, bg); pal.setColor(QPalette.ColorRole.WindowText, fg)
        self.setPalette(pal)
        for w in (self.t,self.s,self.w): w.setStyleSheet(f"color:{fg.name()};")

# -------------------- 도면 + 마커 --------------------
class MapView(QWidget):
    def __init__(self, img_path: str, positions: Dict[int, Tuple[int,int]]):
        super().__init__()
        self.positions = positions
        self.base = QLabel(); self.base.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pm = QPixmap(img_path)
        if pm.isNull(): self.base.setText("floorplan.png 를 찾지 못했습니다."); self.base.setStyleSheet("border:1px dashed #999;")
        else: self.base.setPixmap(pm)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.addWidget(self.base)
        self.btns: Dict[int, QPushButton] = {}
        for sid in self.positions:
            b = QPushButton(f"sensor {sid}", self)
            b.setStyleSheet("QPushButton{border:0; font-weight:700; padding:6px 8px; border-radius:14px;}")
            self.btns[sid] = b

    def resizeEvent(self, e):
        super().resizeEvent(e)
        pm = self.base.pixmap()
        if not pm or pm.isNull():
            return

        iw, ih = pm.width(), pm.height()
        lw, lh = self.base.width(), self.base.height()
        
        if iw == 0 or ih == 0:
            return

        scale = min(lw/iw, lh/ih)
        offx, offy = (lw - iw*scale)/2, (lh - ih*scale)/2
        for sid,(x,y) in self.positions.items():
            self.btns[sid].move(int(offx + x*scale)-28, int(offy + y*scale)-12)

    def set_status(self, sid:int, status:str, room:str=""):
        b = self.btns.get(sid)
        if not b: return
        bg, fg = status_colors(status)
        b.setText(f"sensor {sid}\n{room}" if room else f"sensor {sid}")
        b.setStyleSheet(
            "QPushButton{border:0; font-weight:700; padding:6px 8px; border-radius:16px;"
            f"background:{bg.name()}; color:{fg.name()};}}"
        )

# -------------------- 히트맵(온도) --------------------
class Heatmap(QWidget):
    def __init__(self, n=8):
        super().__init__()
        self.n = n
        self.min_t, self.max_t = 20.0, 40.0
        self.show_numbers = True
        self.grid = QGridLayout(self); self.grid.setSpacing(1); self.grid.setContentsMargins(0,0,0,0)
        self.cells: List[List[QLabel]] = []
        self._build(n)

    def _build(self, n):
        for r in self.cells:
            for c in r: self.grid.removeWidget(c); c.deleteLater()
        self.cells=[]
        for i in range(n):
            row=[]
            for j in range(n):
                lb = QLabel("--"); lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lb.setMinimumSize(28,20); lb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                lb.setStyleSheet("border:0; font-weight:700;")
                self.grid.addWidget(lb,i,j); row.append(lb)
            self.cells.append(row)

    def set_grid(self, n):
        if n != self.n:
            self.n = n; self._build(n)

    def _color(self, v: float) -> QColor:
        t = 0.0 if self.max_t==self.min_t else (v-self.min_t)/(self.max_t-self.min_t)
        t = clamp(t,0.0,1.0)
        if t<=0.25:     r,g,b = 0, int(4*t*255), 255
        elif t<=0.5:    r,g,b = 0, 255, int(255*(1-(t-0.25)*4))
        elif t<=0.75:   r,g,b = int(255*(t-0.5)*4), 255, 0
        else:           r,g,b = 255, int(255*(1-(t-0.75)*4)), 0
        return QColor(r,g,b)

    def render(self, mat: List[List[float]]):
        n = len(mat)
        if not n or len(mat[0])!=n: return
        self.set_grid(n)
        vals = [v for row in mat for v in row]
        if vals:
            self.min_t = min(self.min_t, min(vals))
            self.max_t = max(self.max_t, max(vals))
        for i in range(n):
            for j in range(n):
                v = mat[i][j]
                col = self._color(v)
                lb = self.cells[i][j]
                lb.setStyleSheet(f"background:{col.name()}; color:#000; font-weight:700;")
                lb.setText(f"{v:.1f}" if self.show_numbers else "")

@dataclass
class SensorState:
    status: str = "정상"
    room: str = ""

class Dashboard(QMainWindow):
    # === 수정 1: 센서 개수를 설정하는 변수 추가 ===
    NUM_SENSORS = 1

    def __init__(self, floorplan="floorplan.png"):
        super().__init__()
        self.setWindowTitle("화재 감지 대시보드 (온도 전용)")
        self.resize(1280, 820)

        # === 수정 2: 센서 개수 변수를 사용하여 초기화 ===
        self.sensors: Dict[int, SensorState] = {i: SensorState() for i in range(1, self.NUM_SENSORS + 1)}

        splitter = QSplitter(); self.setCentralWidget(splitter)

        # 왼쪽: 도면 (센서 1개에 대한 위치 정보만 남김)
        all_positions = {1:(140,400), 2:(110,190), 3:(250,190), 4:(360,190), 5:(320,410)}
        positions = {i: all_positions[i] for i in range(1, self.NUM_SENSORS + 1) if i in all_positions}
        self.map = MapView(floorplan, positions)
        lwrap = QWidget(); ll = QVBoxLayout(lwrap);
        title = QLabel("현장 MAP"); title.setStyleSheet("font-weight:800;")
        ll.addWidget(title); ll.addWidget(self.map,1)
        splitter.addWidget(lwrap)

        # 오른쪽: 상단 카운트바 + 카드 + 컨트롤 + 히트맵
        right = QWidget(); rlay = QVBoxLayout(right); rlay.setSpacing(10)

        # --- 상단 집계 바 (변경 없음) ---
        fire_bg, fire_fg   = status_colors("화재")
        warn_bg, warn_fg   = status_colors("주의")
        normal_bg, nor_fg  = status_colors("정상")
        self.pill_fire   = CountPill("화재", fire_bg, fire_fg)
        self.pill_warn   = CountPill("주의", warn_bg, warn_fg)
        self.pill_normal = CountPill("정상", normal_bg, nor_fg)
        counts_bar = QHBoxLayout(); counts_bar.setSpacing(8)
        counts_bar.addWidget(self.pill_fire); counts_bar.addWidget(self.pill_warn); counts_bar.addWidget(self.pill_normal)
        counts_bar.addStretch(1)
        rlay.addLayout(counts_bar)

        # === 수정 3: 센서 카드 레이아웃을 1개에 맞게 단순화 ===
        # 센서 카드를 1개만 생성
        self.cards = [SensorCard(f"Sensor {i}") for i in range(1, self.NUM_SENSORS + 1)]
        
        # 복잡한 그리드 대신, 카드 1개를 표시할 간단한 레이아웃 사용
        card_layout = QHBoxLayout()
        if self.cards:
            card_layout.addWidget(self.cards[0]) # 첫 번째 카드만 추가
            card_layout.addStretch(1) # 오른쪽에 빈 공간 추가
        rlay.addLayout(card_layout)

        # === 수정 4: 불필요한 센서 선택 콤보박스 제거 ===
        top = QHBoxLayout()
        # self.sensor_cb = QComboBox() # 콤보박스 생성 코드 제거
        # for i in range(1,6): self.sensor_cb.addItem(f"Sensor {i}", i) # 콤보박스 항목 추가 코드 제거
        
        self.toggle_numbers_btn = QPushButton("숫자 표시 토글")
        self.toggle_numbers_btn.clicked.connect(self._toggle_numbers)
        
        # top.addWidget(self.sensor_cb) # 레이아웃에 콤보박스 추가 코드 제거
        top.addStretch(1) # 왼쪽에 빈 공간 추가
        top.addWidget(self.toggle_numbers_btn) # 숫자 토글 버튼만 오른쪽에 배치
        rlay.addLayout(top)

        # --- 히트맵 (변경 없음) ---
        self.heatmap = Heatmap(8)
        rlay.addWidget(self.heatmap, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0,1); splitter.setStretchFactor(1,1)

        self._refresh_counts()

    def _toggle_numbers(self):
        self.heatmap.show_numbers = not self.heatmap.show_numbers

    def update_from_detector(self, payload: dict):
        # === 수정 5: 콤보박스가 없으므로 데이터 업데이트 로직 수정 ===
        # 콤보박스에서 ID를 가져오는 부분을 제거하고, 없으면 1로 간주
        sid = payload.get("sensor_id") or payload.get("sid") or 1

        # sid가 self.sensors에 있는지 확인 (e.g. sid가 10처럼 큰 값이 들어올 경우 방지)
        if sid in self.sensors:
            status = payload.get("status") or payload.get("status_kor") or self.sensors[sid].status
            room = payload.get("room", self.sensors[sid].room)

            self.sensors[sid].status = status
            self.sensors[sid].room = room

            # sid-1을 인덱스로 사용하여 올바른 카드와 맵 마커를 업데이트
            if 0 <= sid - 1 < len(self.cards):
                self.cards[sid-1].apply(status, time.time())
                self.map.set_status(sid, status, room)
            
            self._refresh_counts()

        # 히트맵 로직 (변경 없음)
        mat = None
        if isinstance(payload.get("grid"), dict):
            g = payload["grid"]; h,w = int(g.get("h",8)), int(g.get("w",8))
            vals = g.get("values", [])
            if vals and len(vals)>=h*w: mat = [vals[i*w:(i+1)*w] for i in range(h)]
        elif "processed_values" in payload:
            n = int(payload.get("interpolated_grid_size", 8))
            vals = payload["processed_values"]
            if vals and len(vals)>=n*n: mat = [vals[i*n:(i+1)*n] for i in range(n)]
        if mat: self.heatmap.render(mat)

    def _refresh_counts(self):
        fire = warn = normal = 0
        # self.sensors 딕셔너리에 이제 1개의 센서만 있으므로 올바르게 계산됨
        for s in self.sensors.values():
            st = (s.status or "").lower()
            if st in ("화재","fire"): fire += 1
            elif st in ("주의","warn","warning"): warn += 1
            else: normal += 1
        self.pill_fire.set_count(fire)
        self.pill_warn.set_count(warn)
        self.pill_normal.set_count(normal)


class OutputModule(QWidget):
    def __init__(self, data_signal_obj: DataSignal, available_rect):
        super().__init__()
        self.win = Dashboard("floorplan.png")
        data_signal_obj.update_data_signal.connect(self.update_from_detector)
        self._status_label = QLabel("", self.win)
        self.win.statusBar().showMessage("대기 중") if hasattr(self.win, "statusBar") else None

    def show(self):
        self.win.show()

    def update_connection_status(self, connected: bool, port, error_msg: str = ""):
        text = f"연결됨: {port}" if connected else ("연결 실패: " + error_msg if error_msg else "연결 안됨")
        if hasattr(self.win, "statusBar"):
            self.win.statusBar().showMessage(text)
        else:
            self.win.setWindowTitle(f"화재 감지 대시보드 - {text}")

    def update_from_detector(self, payload: dict):
        self.win.update_from_detector(payload)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Dashboard("floorplan.png")
    win.show()
    sys.exit(app.exec())