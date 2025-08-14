import sys, time, random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QFrame, QSizePolicy, QSplitter
)

from signals import DataSignal

# (스타일시트 코드는 생략하지 않음)
GLOBAL_STYLESHEET = """
/* 전체 위젯 기본 스타일 */
QWidget { color: #E0E0E0; font-family: 'Malgun Gothic', 'Segoe UI', 'Arial'; }
#MainWindow { background-color: #2E2E2E; }
#TopPanel, #LeftPanel { background-color: #080F25; border-radius: 8px; padding: 10px; }
#ConnectionPanel { background-color: #4A4A4A; border-radius: 6px; padding: 8px; border: 2px solid #666666; }
#ConnectionPanel[connected="true"] { border-color: #4CAF50; background-color: #2E4A2E; }
#ConnectionPanel[connected="false"] { border-color: #F44336; background-color: #4A2E2E; }
.Panel { background-color: #353535; border-color: #F44336; border-radius: 6px; padding: 8px; }
QLabel[class="TitleLabel"] { font-size: 16pt; font-weight: bold; color: #FFFFFF; padding: 2px; }
QLabel[class="InfoLabel"] { font-size: 14pt; font-weight: bold; }
QLabel[class="ConnectionLabel"] { font-size: 12px; font-weight: bold; }
#AnomalyCountLabel { font-size: 16px; font-weight: bold; color: #F44336; padding: 2px; }
QLabel[class="HeatmapCell"] { font-weight: bold; border: none; }
QPushButton { background-color: #555555; color: #FFFFFF; border: 1px solid #666666; border-radius: 4px; padding: 8px 12px; font-size: 12px; font-weight: bold; }
QPushButton:hover { background-color: #6A6A6A; }
QPushButton:pressed { background-color: #4A4A4A; }
QPushButton[buttonType="connect"] { background-color: #4CAF50; }
QPushButton[buttonType="connect"]:hover { background-color: #5CBF60; }
QPushButton[buttonType="disconnect"] { background-color: #F44336; }
QPushButton[buttonType="disconnect"]:hover { background-color: #FF5346; }
QPushButton[class="AdjustButton"] { font-size: 12px; font-weight: bold; padding: 4px 8px; min-width: 30px; }
QComboBox { background-color: #2E2E2E; color: #FFFFFF; border: 1px solid #666666; border-radius: 4px; padding: 5px; font-size: 12px; font-weight: bold; min-width: 80px; }
QComboBox::drop-down { border: none; }
QComboBox::down-arrow { border: none; }
QDoubleSpinBox { background-color: #2E2E2E; color: #FFFFFF; border: 1px solid #666666; border-radius: 4px; padding: 5px; font-size: 12px; font-weight: bold; }
QLabel[styleClass="Indicator"] { min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px; border-radius: 10px; border: 1px solid rgba(0, 0, 0, 100); }
QLabel[styleClass="Indicator"][state="stable"] { background-color: qlineargradient(spread:pad, x1:0.145, y1:0.16, x2:0.92, y2:0.915, stop:0 rgba(0, 255, 0, 255), stop:1 rgba(0, 128, 0, 255)); }
QLabel[styleClass="Indicator"][state="detected"] { background-color: qlineargradient(spread:pad, x1:0.145, y1:0.16, x2:0.92, y2:0.915, stop:0 rgba(255, 0, 0, 255), stop:1 rgba(128, 0, 0, 255)); }
.temp_widget { background-color: #37446B; border: 1px solid #1D2439; border-radius: 5px; padding: 5px 10px; }
#safety { border: 3px solid green; }
#caution { border: 3px solid yellow; }
#danger { border: 3px solid red; }
"""

def clamp(v, a, b): return a if v < a else (b if v > b else v)

def status_colors(s: str):
    s = (s or "").lower()
    if s in ("화재", "fire"): return QColor("#F44336"), QColor("#FFFFFF")
    if s in ("주의", "warn", "warning"): return QColor("#FFEB3B"), QColor("#222222")
    return QColor("#4CAF50"), QColor("#FFFFFF")

class CountPill(QLabel):
    def __init__(self, label, bg, fg):
        super().__init__(f"{label} 0")
        self.bg, self.fg = bg, fg
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(88)
        self.setStyleSheet(f"border-radius:16px; padding:6px 12px; background:{bg.name()}; color:{fg.name()}; font-weight:800;")
    def set_count(self, n):
        name = self.text().split()[0]
        self.setText(f"{name} {n}")

class SensorCard(QFrame):
    def __init__(self, name):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self.t = QLabel(name)
        self.t.setStyleSheet("font-weight:800;")
        self.s = QLabel("대기 중")
        self.w = QLabel("-")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14,10,14,10)
        lay.setSpacing(4)
        lay.addWidget(self.t)
        lay.addWidget(self.s)
        lay.addWidget(self.w)
        self.apply("정상")
    def apply(self, status, ts=None):
        self.s.setText("화재 발생" if status in ("화재","fire") else ("주의" if status in ("주의","warn","warning") else "정상"))
        self.w.setText(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts or time.time())))
        bg, fg = status_colors(status)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, bg)
        pal.setColor(QPalette.ColorRole.WindowText, fg)
        self.setPalette(pal)
        for w in (self.t, self.s, self.w):
            w.setStyleSheet(f"color:{fg.name()};")

class MapView(QWidget):
    def __init__(self, img_path, positions):
        super().__init__()
        self.positions = positions
        self.base = QLabel()
        self.base.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pm = QPixmap(img_path)
        if pm.isNull():
            self.base.setText("floorplan.png 를 찾지 못했습니다.")
            self.base.setStyleSheet("border:1px dashed #999;")
        else:
            self.base.setPixmap(pm)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.base)
        self.btns = {}
        for sid in self.positions:
            b = QPushButton(f"sensor {sid}", self)
            b.setStyleSheet("QPushButton{border:0; font-weight:700; padding:6px 8px; border-radius:14px;}")
            self.btns[sid] = b
    def resizeEvent(self, e):
        super().resizeEvent(e)
        pm = self.base.pixmap()
        if not pm or pm.isNull(): return
        iw, ih = pm.width(), pm.height()
        lw, lh = self.base.width(), self.base.height()
        if iw == 0 or ih == 0: return
        scale = min(lw/iw, lh/ih)
        offx, offy = (lw - iw*scale)/2, (lh - ih*scale)/2
        for sid,(x,y) in self.positions.items():
            self.btns[sid].move(int(offx + x*scale)-28, int(offy + y*scale)-12)
    def set_status(self, sid, status, room=""):
        b = self.btns.get(sid)
        if not b: return
        bg, fg = status_colors(status)
        b.setText(f"sensor {sid}\n{room}" if room else f"sensor {sid}")
        b.setStyleSheet("QPushButton{border:0; font-weight:700; padding:6px 8px; border-radius:16px;" f"background:{bg.name()}; color:{fg.name()};}}")

class Heatmap(QWidget):
    def __init__(self, n=8):
        super().__init__()
        self.n = n
        self.min_t, self.max_t = 20.0, 40.0
        self.show_numbers = True
        self.grid = QGridLayout(self)
        self.grid.setSpacing(1)
        self.grid.setContentsMargins(0,0,0,0)
        self.cells = []
        self._build(n)
    def _build(self, n):
        for row in self.cells:
            for cell in row: cell.deleteLater()
        self.cells=[]
        for i in range(n):
            row = []
            for j in range(n):
                lb = QLabel("--")
                lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lb.setMinimumSize(28,20)
                lb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                lb.setStyleSheet("border:0; font-weight:700;")
                self.grid.addWidget(lb,i,j)
                row.append(lb)
            self.cells.append(row)
    def set_grid(self, n):
        if n != self.n:
            self.n = n
            self._build(n)
    def _color(self, v):
        t = clamp((v-self.min_t)/(self.max_t-self.min_t), 0.0, 1.0) if self.max_t != self.min_t else 0
        if t<=0.25: r,g,b = 0, int(4*t*255), 255
        elif t<=0.5: r,g,b = 0, 255, int(255*(1-(t-0.25)*4))
        elif t<=0.75: r,g,b = int(255*(t-0.5)*4), 255, 0
        else: r,g,b = 255, int(255*(1-(t-0.75)*4)), 0
        return QColor(r,g,b)
    def render(self, mat):
        n = len(mat)
        if not n or len(mat[0])!=n: return
        self.set_grid(n)
        vals = [v for row in mat for v in row]
        if vals:
            self.min_t = min(self.min_t, min(vals))
            self.max_t = max(self.max_t, max(vals))
        for i in range(n):
            for j in range(n):
                self.cells[i][j].setStyleSheet(f"background:{self._color(mat[i][j]).name()}; color:#000; font-weight:700;")
                self.cells[i][j].setText(f"{mat[i][j]:.1f}" if self.show_numbers else "")

@dataclass
class SensorState:
    status: str = "정상"
    room: str = ""

class Dashboard(QMainWindow):
    NUM_SENSORS = 1
    def __init__(self, floorplan="floorplan.png"):
        super().__init__()
        self.setWindowTitle("화재 감지 대시보드 (온도 전용)")
        self.resize(1280, 820)
        self.sensors = {i: SensorState() for i in range(1, self.NUM_SENSORS + 1)}
        splitter = QSplitter(); self.setCentralWidget(splitter)
        all_positions = {1:(140,400), 2:(110,190), 3:(250,190), 4:(360,190), 5:(320,410)}
        positions = {i: all_positions[i] for i in range(1, self.NUM_SENSORS + 1) if i in all_positions}
        self.map = MapView(floorplan, positions)
        lwrap = QWidget(); ll = QVBoxLayout(lwrap); title = QLabel("현장 MAP"); title.setStyleSheet("font-weight:800;"); ll.addWidget(title); ll.addWidget(self.map,1); splitter.addWidget(lwrap)
        right = QWidget(); rlay = QVBoxLayout(right); rlay.setSpacing(10)
        fire_bg, fire_fg = status_colors("화재"); warn_bg, warn_fg = status_colors("주의"); normal_bg, nor_fg = status_colors("정상")
        self.pill_fire, self.pill_warn, self.pill_normal = CountPill("화재", fire_bg, fire_fg), CountPill("주의", warn_bg, warn_fg), CountPill("정상", normal_bg, nor_fg)
        counts_bar = QHBoxLayout(); counts_bar.setSpacing(8); [counts_bar.addWidget(p) for p in (self.pill_fire, self.pill_warn, self.pill_normal)]; counts_bar.addStretch(1); rlay.addLayout(counts_bar)
        self.cards = [SensorCard(f"Sensor {i}") for i in range(1, self.NUM_SENSORS + 1)]
        card_layout = QHBoxLayout(); [card_layout.addWidget(self.cards[0]), card_layout.addStretch(1)] if self.cards else None; rlay.addLayout(card_layout)
        top = QHBoxLayout(); self.toggle_numbers_btn = QPushButton("숫자 표시 토글"); self.toggle_numbers_btn.clicked.connect(self._toggle_numbers); top.addStretch(1); top.addWidget(self.toggle_numbers_btn); rlay.addLayout(top)
        self.heatmap = Heatmap(8); rlay.addWidget(self.heatmap, 1); splitter.addWidget(right); splitter.setStretchFactor(0,1); splitter.setStretchFactor(1,1); self._refresh_counts()

    def _toggle_numbers(self):
        self.heatmap.show_numbers = not self.heatmap.show_numbers

    def update_from_detector(self, payload: dict):
        data_content = payload.get('data', {})
        sid = data_content.get("sensor_id") or data_content.get("sid") or 1
        if sid in self.sensors:
            status = data_content.get("status") or data_content.get("status_kor") or self.sensors[sid].status
            room = data_content.get("room", self.sensors[sid].room)
            self.sensors[sid].status, self.sensors[sid].room = status, room
            if 0 <= sid-1 < len(self.cards):
                self.cards[sid-1].apply(status, time.time())
                self.map.set_status(sid, status, room)
            self._refresh_counts()
        print("\n--- 히트맵 데이터 분석 시작 ---")
        print(f"  [DEBUG-1] 'data_content'에 'grid' 키 존재 여부: {'grid' in data_content}")
        mat = None
        grid_value = data_content.get("grid")
        print(f"  [DEBUG-2] 'grid' 키의 값 타입: {type(grid_value)}")
        if isinstance(grid_value, dict):
            print("  [DEBUG-3] 'grid'는 딕셔너리입니다. 내부 분석 시작...")
            h, w = int(grid_value.get("h", 0)), int(grid_value.get("w", 0))
            vals = grid_value.get("values", [])
            print(f"    - h={h}, w={w}, values 리스트 길이={len(vals)}")
            if vals and len(vals) >= h * w:
                mat = [vals[i * w:(i + 1) * w] for i in range(h)]
                print("    - 'mat' 매트릭스 생성 성공!")
            else:
                print("    - 'values'가 비어있거나 길이가 부족하여 'mat' 생성 실패.")
        else:
            print("  [DEBUG-3] 'grid'가 딕셔너리가 아니어서 히트맵을 처리할 수 없습니다.")
        print(f"  [DEBUG-4] 최종 'mat' 객체 생성 여부: {'성공 (객체 있음)' if mat else '실패 (객체 없음)'}")
        print("--- 히트맵 데이터 분석 종료 ---\n")
        if mat: self.heatmap.render(mat)

    # ==================== 수정된 부분 ====================
    def _refresh_counts(self):
        # 1. 카운트할 변수를 0으로 초기화합니다.
        fire = 0
        warn = 0
        normal = 0
        # 2. 모든 센서를 하나씩 확인하면서 상태를 셉니다.
        for s in self.sensors.values():
            st = (s.status or "").lower()
            if st in ("화재", "fire"):
                fire += 1
            elif st in ("주의", "warn", "warning"):
                warn += 1
            else:
                normal += 1
        # 3. 계산된 숫자로 화면의 배지를 업데이트합니다.
        self.pill_fire.set_count(fire)
        self.pill_warn.set_count(warn)
        self.pill_normal.set_count(normal)
    # =====================================================

class OutputModule(QWidget):
    def __init__(self, data_signal_obj, available_rect):
        super().__init__()
        self.win = Dashboard("floorplan.png")
        data_signal_obj.update_data_signal.connect(self.update_from_detector)
    def show(self):
        self.win.show()
    def update_connection_status(self, connected, port, error_msg=""):
        text = f"연결됨: {port}" if connected else (f"연결 실패: {error_msg}" if error_msg else "연결 안됨")
        status_bar = getattr(self.win, "statusBar", None)
        if status_bar:
            status_bar().showMessage(text)
        else:
            self.win.setWindowTitle(f"화재 감지 대시보드 - {text}")
    def update_from_detector(self, payload: dict):
        print(f"[GUI DEBUG] 수신된 전체 데이터 (요약): {str(payload)[:300]}...")
        self.win.update_from_detector(payload)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Dashboard("floorplan.png")
    win.show()
    sys.exit(app.exec())