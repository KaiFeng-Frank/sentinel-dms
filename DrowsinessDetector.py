import math
import queue
import threading
import time
from collections import deque
try:
    import winsound  # Windows only
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from slow_system import SlowSystem, SlowSystemConfig
from decision_fusion import DecisionFusion


# ----- color helpers for the multi-dimensional VLM panel ------------------

def _level_color(level) -> str:
    try:
        v = float(level)
    except (TypeError, ValueError):
        return "#9E9E9E"
    if v < 3:
        return "#2E7D32"   # green
    if v < 5:
        return "#FBC02D"   # yellow
    if v < 7:
        return "#F57C00"   # orange
    return "#C62828"       # red


def _severity_color(sev: str) -> str:
    return {
        "none": "#2E7D32",
        "low": "#FBC02D",
        "medium": "#F57C00",
        "high": "#C62828",
    }.get(str(sev).lower(), "#9E9E9E")


def _action_color(action: str) -> str:
    return {
        "none": "#2E7D32",
        "verbal_warning": "#FBC02D",
        "alarm": "#F57C00",
        "pull_over": "#C62828",
    }.get(str(action).lower(), "#9E9E9E")


def _action_label(action: str) -> str:
    return {
        "none": "无需动作",
        "verbal_warning": "语音提醒",
        "alarm": "报警",
        "pull_over": "立即靠边停车",
    }.get(str(action).lower(), str(action))


def _bool_dot(v: bool, true_color: str = "#C62828", false_color: str = "#2E7D32") -> str:
    return f"<span style='color:{true_color if v else false_color};font-size:14px;'>●</span>"


def _safe(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


# MediaPipe face-mesh landmark indices for the standard 6-point EAR computation
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [263, 387, 385, 362, 380, 373]


def _ear_from_landmarks(landmarks, idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx]

    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    horiz = d(pts[0], pts[3])
    if horiz < 1e-6:
        return 0.0
    return (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2.0 * horiz)

class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        self.yawn_state = ''
        self.left_eye_state =''
        self.right_eye_state= ''
        self.alert_text = ''

        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0 

        self.left_eye_still_closed = False  
        self.right_eye_still_closed = False 
        self.yawn_in_progress = False  
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]

        # ---- sliding-window state for Fast System metrics ----
        # ~10 s @ 30 fps for both PERCLOS and YOLO confidence
        self._eye_history = deque(maxlen=300)
        self._conf_history = deque(maxlen=300)
        self._ear_value = 0.0
        self._slow_submit_counter = 0

        # Fast / Slow / Fused state shared with the GUI
        self._fast_state = {
            "drowsiness_level": 0.0,
            "confidence": 0.0,
            "perclos": 0.0,
            "ear": 0.0,
            "microsleeps": 0.0,
            "yawns": 0,
            "yawn_duration": 0.0,
        }
        self._slow_state = None
        self._fusion = DecisionFusion(slow_max_age_s=30.0)
        self._fusion_result = None

        # =================== Product-grade UI ===================
        self.setWindowTitle("Driver Monitoring System · Fast + Slow VLM")
        self.setGeometry(40, 40, 1600, 960)
        self.setStyleSheet(
            "QMainWindow { background-color: #0b1220; }"
            "QWidget { background-color: #0b1220; }"
        )

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        root_layout = QVBoxLayout(self.central_widget)
        root_layout.setContentsMargins(24, 18, 24, 18)
        root_layout.setSpacing(14)

        # ---------- Header bar ----------
        self.title_label = QLabel(
            "<span style='color:#f8fafc;font-size:26px;font-weight:bold;'>"
            "🚗 驾驶员监控系统</span>"
            "<span style='color:#64748b;font-size:14px;'>"
            "  ·  Driver Monitoring System  ·  Fast + Slow VLM Architecture</span>"
        )
        self.title_label.setStyleSheet(
            "QLabel { background: transparent; padding: 4px 4px 8px 4px; }"
        )
        root_layout.addWidget(self.title_label)

        # ---------- Middle content row ----------
        content_row = QWidget()
        content_row.setStyleSheet("background: transparent;")
        content_layout = QHBoxLayout(content_row)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        # Left: video card
        self.video_label = QLabel()
        self.video_label.setFixedSize(720, 540)
        self.video_label.setStyleSheet(
            "QLabel { background: #000000; border: 2px solid #1e293b; "
            "border-radius: 10px; }"
        )
        self.video_label.setAlignment(Qt.AlignCenter)
        content_layout.addWidget(self.video_label)

        # Right: card column
        right_col = QWidget()
        right_col.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        card_style = (
            "QLabel { "
            "background: #0f172a; "
            "border: 1px solid #1e293b; "
            "border-radius: 12px; "
            "padding: 20px; "
            "color: #e2e8f0; }"
        )

        self.hero_label = QLabel()
        self.hero_label.setStyleSheet(card_style)
        self.hero_label.setMinimumWidth(780)
        self.hero_label.setAlignment(Qt.AlignCenter)
        self.hero_label.setWordWrap(True)

        self.fast_label = QLabel()
        self.fast_label.setStyleSheet(card_style)
        self.fast_label.setWordWrap(True)
        self.fast_label.setAlignment(Qt.AlignTop)

        self.slow_label = QLabel()
        self.slow_label.setStyleSheet(card_style)
        self.slow_label.setWordWrap(True)
        self.slow_label.setAlignment(Qt.AlignTop)

        right_layout.addWidget(self.hero_label, 3)
        right_layout.addWidget(self.fast_label, 2)
        right_layout.addWidget(self.slow_label, 4)

        content_layout.addWidget(right_col, 1)
        root_layout.addWidget(content_row, 1)

        # ---------- Bottom full-width VLM explanation strip ----------
        self.final_label = QLabel()
        self.final_label.setStyleSheet(
            "QLabel { background: #0f172a; border: 1px solid #1e293b; "
            "border-radius: 12px; padding: 18px 22px; color: #cbd5e1; }"
        )
        self.final_label.setWordWrap(True)
        self.final_label.setAlignment(Qt.AlignTop)
        self.final_label.setMinimumHeight(140)
        root_layout.addWidget(self.final_label)

        self.update_info()

        self.detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
        self.detecteye = YOLO("runs/detecteye/train/weights/best.pt")

        # Slow System (VLM) — DashScope OpenAI-compatible Qwen-Omni.
        # Credentials MUST be supplied via env vars (no hardcoded keys):
        #   DASHSCOPE_API_KEY   required
        #   DASHSCOPE_BASE_URL  optional, defaults to DashScope compat endpoint
        #   DASHSCOPE_MODEL     optional, defaults to qwen3.5-omni-plus
        # If DASHSCOPE_API_KEY is unset the SlowSystem falls back to mock mode
        # so the UI still exercises the full pipeline without an API key.
        import os
        _api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        self._slow_system = SlowSystem(
            SlowSystemConfig(
                interval_seconds=10.0,
                mock_mode=(_api_key == ""),
                base_url=os.environ.get(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                api_key=_api_key,
                model_name=os.environ.get(
                    "DASHSCOPE_MODEL", "qwen3.5-omni-plus"
                ),
                request_timeout=40.0,
            )
        )
        self._slow_system.start()

        self.cap = cv2.VideoCapture(0)
        time.sleep(1.000)

        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()
        
    def update_info(self):
        fast = self._fast_state
        slow = self._slow_state
        fused = self._fusion_result

        # Product-grade dark-theme palette
        GREEN = "#10b981"
        YELLOW = "#eab308"
        ORANGE = "#f97316"
        RED = "#ef4444"
        MUTED = "#64748b"
        FG = "#e2e8f0"

        def risk_color_for(label: str) -> str:
            return {
                "正常": GREEN, "轻度疲劳": YELLOW,
                "中度疲劳": ORANGE, "严重疲劳": RED,
            }.get(label, MUTED)

        def action_color_for(a: str) -> str:
            return {
                "none": GREEN, "verbal_warning": YELLOW,
                "alarm": ORANGE, "pull_over": RED,
            }.get(str(a).lower(), MUTED)

        # ============================================================
        #   HERO CARD — big overall-risk number + recommended action
        # ============================================================
        if slow is not None:
            overall_val = float(_safe(slow, "overall_risk", default=0) or 0)
        elif fused is not None:
            overall_val = float(fused.drowsiness_level)
        else:
            overall_val = 0.0

        risk_label = fused.risk_label if fused is not None else "初始化中"
        r_color = risk_color_for(risk_label)

        action = _safe(slow, "recommended_action", default="none") or "none"
        a_color = action_color_for(action)
        a_label = _action_label(action)

        fw = fused.fast_weight if fused else 1.0
        sw = fused.slow_weight if fused else 0.0

        hero_html = (
            "<table width='100%' cellpadding='0' cellspacing='0'>"
            "<tr><td align='center' valign='middle'>"
            f"<div style='color:{MUTED};font-size:13px;letter-spacing:2px;'>"
            "OVERALL RISK  ·  综合风险评估"
            "</div>"
            "<div style='margin-top:10px;'>"
            f"<span style='color:{r_color};font-size:84px;font-weight:bold;'>"
            f"{overall_val:.1f}"
            "</span>"
            f"<span style='color:{MUTED};font-size:34px;'> / 10</span>"
            "</div>"
            f"<div style='color:{r_color};font-size:26px;font-weight:bold;margin-top:2px;'>"
            f"{risk_label}"
            "</div>"
            f"<div style='color:{MUTED};font-size:13px;margin-top:8px;'>"
            f"Fusion weights · Fast {fw:.0%} + Slow VLM {sw:.0%}"
            "</div>"
            "</td></tr>"
            "<tr><td height='14'></td></tr>"
            f"<tr><td bgcolor='{a_color}' align='center'>"
            "<table width='100%' cellpadding='14' cellspacing='0'><tr><td align='center'>"
            "<span style='color:#ffffff;font-size:20px;'>⚠ 建议动作  ·  </span>"
            f"<span style='color:#ffffff;font-size:24px;font-weight:bold;'>{a_label}</span>"
            "</td></tr></table>"
            "</td></tr>"
            "</table>"
        )
        self.hero_label.setText(hero_html)

        # ============================================================
        #   FAST SYSTEM CARD — realtime metrics grid
        # ============================================================
        fast_drowsy_color = _level_color(fast["drowsiness_level"])
        perclos_color = _level_color(fast["perclos"] * 12)

        def metric(label, value_html):
            return (
                f"<td width='25%' valign='top'>"
                f"<div style='color:{MUTED};font-size:13px;'>{label}</div>"
                f"<div style='margin-top:2px;'>{value_html}</div>"
                "</td>"
            )

        unit = (
            f"<span style='color:{MUTED};font-size:14px;'>"
        )

        fast_html = (
            "<div style='font-family: Arial, sans-serif;'>"
            f"<div style='color:{MUTED};font-size:13px;font-weight:bold;letter-spacing:2px;'>"
            "● FAST SYSTEM  ·  实时检测 ~30 FPS"
            "</div>"
            "<table width='100%' cellpadding='6' cellspacing='0' "
            "style='margin-top:12px;'>"
            "<tr>"
            + metric("眨眼次数",
                     f"<b style='color:{FG};font-size:22px;'>{self.blinks}</b>")
            + metric("微睡眠",
                     f"<b style='color:{FG};font-size:22px;'>"
                     f"{round(self.microsleeps, 2)}</b>{unit} s</span>")
            + metric("哈欠次数",
                     f"<b style='color:{FG};font-size:22px;'>{self.yawns}</b>")
            + metric("PERCLOS",
                     f"<b style='color:{perclos_color};font-size:22px;'>"
                     f"{fast['perclos'] * 100:.1f}</b>{unit} %</span>")
            + "</tr><tr><td height='10'></td></tr><tr>"
            + metric("EAR",
                     f"<b style='color:{FG};font-size:22px;'>{fast['ear']:.3f}</b>")
            + metric("哈欠时长",
                     f"<b style='color:{FG};font-size:22px;'>"
                     f"{round(self.yawn_duration, 2)}</b>{unit} s</span>")
            + metric("Fast 疲劳度",
                     f"<b style='color:{fast_drowsy_color};font-size:22px;'>"
                     f"{fast['drowsiness_level']:.1f}</b>{unit} /10</span>")
            + metric("Fast 置信度",
                     f"<b style='color:{FG};font-size:22px;'>"
                     f"{fast['confidence']:.2f}</b>")
            + "</tr></table>"
            "</div>"
        )
        self.fast_label.setText(fast_html)

        # ============================================================
        #   SLOW SYSTEM CARD — VLM 5 dimensions
        # ============================================================
        if slow is None:
            slow_html = (
                "<div style='font-family: Arial, sans-serif;'>"
                f"<div style='color:{MUTED};font-size:13px;font-weight:bold;"
                "letter-spacing:2px;'>"
                "◆ SLOW SYSTEM  ·  VLM 多维度分析"
                "</div>"
                "<table width='100%' height='180'><tr><td align='center' valign='middle'>"
                f"<span style='color:{MUTED};font-size:17px;'>"
                "🔄 等待首次 VLM 分析（每 10 秒）…"
                "</span>"
                "</td></tr></table>"
                "</div>"
            )
        else:
            age = max(0.0, time.time() - float(slow.get("timestamp", time.time())))

            d_level = _safe(slow, "drowsiness", "level", default=0) or 0
            d_conf = _safe(slow, "drowsiness", "confidence", default=0.0) or 0.0
            d_color = _level_color(d_level)

            di_det = bool(_safe(slow, "distraction", "detected", default=False))
            di_type = _safe(slow, "distraction", "type", default="none") or "none"
            di_conf = float(_safe(slow, "distraction", "confidence", default=0.0) or 0.0)
            di_color = RED if di_det else GREEN
            di_text = di_type if di_det else "未检测到"

            an_det = bool(_safe(slow, "anomaly", "detected", default=False))
            an_sev = _safe(slow, "anomaly", "severity", default="none") or "none"
            an_desc = _safe(slow, "anomaly", "description", default="") or ""
            an_color = {
                "none": GREEN, "low": YELLOW,
                "medium": ORANGE, "high": RED,
            }.get(str(an_sev).lower(), MUTED)
            if an_det and an_desc:
                an_text = an_desc
            elif an_det:
                an_text = an_sev
            else:
                an_text = "无异常"

            occ_types = _safe(slow, "occlusion", "type", default=["none"]) or ["none"]
            occ_impact = float(_safe(slow, "occlusion", "impact_on_reliability",
                                     default=0.0) or 0.0)
            occ_text = "、".join(occ_types) if occ_types else "无"
            occ_color = _level_color(occ_impact * 10)

            lighting = _safe(slow, "context", "lighting", default="good") or "good"
            light_color = {
                "good": GREEN, "dim": YELLOW, "dark": RED,
            }.get(str(lighting).lower(), MUTED)
            passengers = bool(_safe(slow, "context", "passengers_detected", default=False))

            di_conf_tail = (
                f"  <span style='color:{MUTED};font-size:13px;'>"
                f"conf {di_conf:.2f}</span>"
                if di_det else ""
            )

            def row(label, value_html):
                return (
                    "<tr>"
                    f"<td width='160' valign='middle' "
                    f"style='color:{MUTED};font-size:14px;'>{label}</td>"
                    f"<td valign='middle' style='color:{FG};font-size:18px;'>"
                    f"{value_html}</td>"
                    "</tr>"
                )

            rows = (
                row("疲劳 Drowsiness",
                    f"<b style='color:{d_color};'>{d_level}</b>"
                    f"<span style='color:{MUTED};'> / 10</span>"
                    f"  <span style='color:{MUTED};font-size:13px;'>conf {d_conf}</span>")
                + row("分心 Distraction",
                      f"<b style='color:{di_color};'>{di_text}</b>{di_conf_tail}")
                + row("异常 Anomaly",
                      f"<b style='color:{an_color};'>{an_text}</b>")
                + row("遮挡 Occlusion",
                      f"<b>{occ_text}</b>"
                      f"  <span style='color:{occ_color};font-size:13px;'>"
                      f"影响可信度 {occ_impact:.2f}</span>")
                + row("场景 Context",
                      f"光线 <b style='color:{light_color};'>{lighting}</b>"
                      f"  ·  乘客 <b>{'有' if passengers else '无'}</b>")
            )

            slow_html = (
                "<div style='font-family: Arial, sans-serif;'>"
                f"<div style='color:{MUTED};font-size:13px;font-weight:bold;"
                "letter-spacing:2px;'>"
                f"◆ SLOW SYSTEM  ·  VLM {slow.get('source','?')}"
                "</div>"
                "<table width='100%' cellpadding='7' cellspacing='0' "
                "style='margin-top:10px;'>"
                f"{rows}"
                "</table>"
                f"<div style='color:#475569;font-size:11px;margin-top:10px;'>"
                f"更新于 {age:.1f}s 前  ·  推理耗时 {slow.get('latency_s', 0)}s"
                "</div>"
                "</div>"
            )
        self.slow_label.setText(slow_html)

        # ============================================================
        #   BOTTOM full-width VLM explanation strip
        # ============================================================
        full_explanation = _safe(slow, "explanation", default="") or ""
        if not full_explanation:
            final_html = (
                "<div style='font-family: Arial, sans-serif;'>"
                f"<div style='color:{MUTED};font-size:13px;font-weight:bold;"
                "letter-spacing:2px;'>"
                "📝 VLM 实时分析报告  ·  REAL-TIME ANALYSIS"
                "</div>"
                "<div style='margin-top:10px;'>"
                f"<span style='color:{MUTED};font-size:16px;'>"
                "等待 VLM 首次分析完成…"
                "</span></div>"
                "</div>"
            )
        else:
            final_html = (
                "<div style='font-family: Arial, sans-serif;'>"
                f"<div style='color:{MUTED};font-size:13px;font-weight:bold;"
                "letter-spacing:2px;'>"
                "📝 VLM 实时分析报告  ·  REAL-TIME ANALYSIS"
                "</div>"
                "<div style='margin-top:10px;'>"
                f"<span style='color:#cbd5e1;font-size:17px;line-height:1.6;'>"
                f"{full_explanation}</span></div>"
                "</div>"
            )
        self.final_label.setText(final_html)


    def predict_eye(self, eye_frame, eye_state):
        results_eye = self.detecteye.predict(eye_frame, verbose=False)
        boxes = results_eye[0].boxes
        if len(boxes) == 0:
            return eye_state, 0.0

        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        max_confidence_index = int(np.argmax(confidences))
        class_id = int(class_ids[max_confidence_index])
        conf = float(confidences[max_confidence_index])

        if class_id == 1:
            eye_state = "Close Eye"
        elif class_id == 0 and conf > 0.30:
            eye_state = "Open Eye"

        return eye_state, conf

    def predict_yawn(self, yawn_frame):
        results_yawn = self.detectyawn.predict(yawn_frame, verbose=False)
        boxes = results_yawn[0].boxes

        if len(boxes) == 0:
            return self.yawn_state

        confidences = boxes.conf.cpu().numpy()  
        class_ids = boxes.cls.cpu().numpy()  
        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])

        if class_id == 0:
            self.yawn_state = "Yawn"
        elif class_id == 1 and confidences[max_confidence_index] > 0.50 :
            self.yawn_state = "No Yawn"
                            

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
            else:
                break

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Throttled push to Slow System (~ every 15 frames ≈ 0.5 s).
            # Slow System still only fires its VLM call every interval_seconds.
            self._slow_submit_counter = (self._slow_submit_counter + 1) % 15
            if self._slow_submit_counter == 0:
                self._slow_system.submit_frame(frame)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            conf_l = conf_r = 0.0
            face_seen = False

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_seen = True
                    ih, iw, _ = frame.shape

                    # ---- EAR from MediaPipe landmarks ----
                    self._ear_value = (
                        _ear_from_landmarks(face_landmarks.landmark, LEFT_EYE_EAR_IDX, iw, ih)
                        + _ear_from_landmarks(face_landmarks.landmark, RIGHT_EYE_EAR_IDX, iw, ih)
                    ) / 2.0

                    points = []
                    for point_id in self.points_ids:
                        lm = face_landmarks.landmark[point_id]
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        points.append((x, y))

                    if len(points) != 0:
                        x1, y1 = points[0]
                        x2, _ = points[1]
                        _, y3 = points[2]

                        x4, y4 = points[3]
                        x5, y5 = points[4]

                        x6, y6 = points[5]
                        x7, y7 = points[6]

                        x6, x7 = min(x6, x7), max(x6, x7)
                        y6, y7 = min(y6, y7), max(y6, y7)

                        mouth_roi = frame[y1:y3, x1:x2]
                        right_eye_roi = frame[y4:y5, x4:x5]
                        left_eye_roi = frame[y6:y7, x6:x7]

                        try:
                            self.left_eye_state, conf_l = self.predict_eye(
                                left_eye_roi, self.left_eye_state
                            )
                            self.right_eye_state, conf_r = self.predict_eye(
                                right_eye_roi, self.right_eye_state
                            )
                            self.predict_yawn(mouth_roi)
                        except Exception as e:
                            print(f"Error al realizar la predicción: {e}")

                        # ---- update sliding-window state for PERCLOS / fast conf ----
                        both_closed = (
                            self.left_eye_state == "Close Eye"
                            and self.right_eye_state == "Close Eye"
                        )
                        self._eye_history.append(both_closed)
                        self._conf_history.append(max(conf_l, conf_r))

                        if both_closed:
                            if not self.left_eye_still_closed and not self.right_eye_still_closed:
                                self.left_eye_still_closed, self.right_eye_still_closed = True, True
                                self.blinks += 1
                            self.microsleeps += 45 / 1000
                        else:
                            if self.left_eye_still_closed and self.right_eye_still_closed:
                                self.left_eye_still_closed, self.right_eye_still_closed = False, False
                            self.microsleeps = 0

                        if self.yawn_state == "Yawn":
                            if not self.yawn_in_progress:
                                self.yawn_in_progress = True
                                self.yawns += 1
                            self.yawn_duration += 45 / 1000
                        else:
                            if self.yawn_in_progress:
                                self.yawn_in_progress = False
                                self.yawn_duration = 0

            # Always run fast-state aggregation, fusion, and GUI refresh — even
            # when no face is detected, so the VLM panel stays responsive and
            # the user can see the fast-system idle state.
            self._update_fast_state(face_seen)
            self._poll_and_fuse()
            self.update_info()
            self.display_frame(frame)
            # cv2.waitKey omitted: opencv-python-headless has no GUI module
            # and the PyQt5 main window already handles quit events.

    # ------------------------------------------------------------------
    # Fast state aggregation + fusion
    # ------------------------------------------------------------------
    def _update_fast_state(self, face_seen: bool):
        n = max(len(self._eye_history), 1)
        perclos = sum(self._eye_history) / n
        avg_conf = (
            sum(self._conf_history) / max(len(self._conf_history), 1)
            if self._conf_history else 0.0
        )
        # If no face this frame, the YOLO confidence drops sharply for the
        # current sample — represent that by a small penalty so fusion learns
        # to lean on the slow system.
        if not face_seen:
            avg_conf *= 0.5

        # Fast drowsiness level 0..10 from PERCLOS, microsleeps, yawn duration
        level = 0.0
        level += min(perclos * 12.0, 5.0)            # 0..5 from PERCLOS
        level += min(self.microsleeps * 1.5, 3.0)    # 0..3 from microsleeps (s)
        level += min(self.yawn_duration * 0.5, 2.0)  # 0..2 from yawn duration (s)
        level = min(level, 10.0)

        self._fast_state.update({
            "drowsiness_level": level,
            "confidence": float(avg_conf),
            "perclos": float(perclos),
            "ear": float(self._ear_value),
            "microsleeps": float(self.microsleeps),
            "yawns": int(self.yawns),
            "yawn_duration": float(self.yawn_duration),
        })

    def _poll_and_fuse(self):
        self._slow_state = self._slow_system.poll_result()
        self._fusion_result = self._fusion.fuse(self._fast_state, self._slow_state)

    # ------------------------------------------------------------------
    # Qt
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.stop_event.set()
        try:
            self._slow_system.stop()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        super().closeEvent(event)

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(
            716, 536, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def play_alert_sound(self):
            frequency = 1000
            duration = 500
            if _HAS_WINSOUND:
                winsound.Beep(frequency, duration)
            else:
                # Linux fallback: terminal bell + best-effort beep
                try:
                    import subprocess
                    subprocess.run(
                        ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                        check=False, timeout=1,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    print("\a", end="", flush=True)

    def play_sound_in_thread(self):
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()
        
    def show_alert_on_frame(self, frame, text="Alerta!"):
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)
        font_scale = 1
        font_color = (0, 0, 255) 
        line_type = 2

        cv2.putText(frame, text, position, font, font_scale, font_color, line_type)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())