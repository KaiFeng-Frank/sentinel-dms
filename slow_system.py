"""
Slow System: periodically samples a frame and sends it to a VLM
(Qwen3-Omni via OpenAI-compatible API) for **multi-dimensional** driver-state
reasoning that the Fast System cannot do.

Runs in a background thread so it never blocks the Fast System real-time loop.
The Fast System pushes the latest frame in via `submit_frame()` (cheap, just
stores a reference under a lock), and the worker wakes every `interval_seconds`
to grab the most recent frame, call the VLM, and stash the result.

The VLM is asked to fill the following schema (also what `poll_result()`
returns, plus `source` / `timestamp` / `latency_s` injected by SlowSystem):

    {
        "drowsiness":  {"level": 0..10, "confidence": 0..1},
        "distraction": {"detected": bool, "type": str, "confidence": 0..1},
        "anomaly":     {"detected": bool, "description": str|null, "severity": "low|medium|high|none"},
        "occlusion":   {"type": list[str], "impact_on_reliability": 0..1},
        "context":     {"lighting": "good|dim|dark", "passengers_detected": bool},
        "overall_risk": 0..10,
        "explanation": str,
        "recommended_action": "none|verbal_warning|alarm|pull_over",
    }

If the VLM call fails or returns malformed JSON, SlowSystem returns
`EMPTY_RESULT` with `source="error"` and the error message in `explanation` so
the GUI can render it without NoneType crashes.
"""

from __future__ import annotations

import base64
import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    _HAS_OPENAI = False


DEFAULT_PROMPT = """你是一名专业的驾驶员状态分析专家。请对图片中的驾驶员进行**多维度**分析：

需要同时分析以下五个维度：
1. **疲劳状态 (drowsiness)**：不要只看眼睛开闭。综合面部肌肉松弛度、头部倾斜角度、整体姿态（含肩颈是否下垂）、表情张力等线索做整体判断。
2. **分心检测 (distraction)**：判断驾驶员是否正在看手机、吃东西、与乘客交谈、低头操作中控/物品、回头等。
3. **异常行为 (anomaly)**：是否有疑似药物或酒精中毒迹象（眼神涣散无焦点、面色异常、肢体不协调）、情绪异常（哭泣、愤怒、惊恐）、身体不适（捂胸口、按头、僵硬）。
4. **遮挡情况 (occlusion)**：识别口罩 (mask)、墨镜 (sunglasses)、帽子 (hat) 等遮挡物，并明确说明这些遮挡是否削弱了你判断的可信度。
5. **场景上下文 (context)**：车内光线条件、是否能看到乘客、其他可推断的驾驶环境信息。

**严格按照下面的 JSON 结构返回**（字段名必须完全一致；不要添加 markdown、注释或额外说明文字；缺失字段会导致后处理失败）：

{
  "drowsiness": {
    "level": 0-10 之间的整数 (0=完全清醒, 10=极度疲劳),
    "confidence": 0-1 之间的浮点数
  },
  "distraction": {
    "detected": true 或 false,
    "type": "phone" / "eating" / "talking" / "looking_away" / "operating" / "other" / "none",
    "confidence": 0-1
  },
  "anomaly": {
    "detected": true 或 false,
    "description": "如果检测到异常，用一句话描述；否则填 null",
    "severity": "low" / "medium" / "high" / "none"
  },
  "occlusion": {
    "type": ["mask" / "sunglasses" / "hat" / "none" 的列表，未遮挡则为 ["none"]],
    "impact_on_reliability": 0-1 之间的浮点数 (0=完全不影响判断, 1=完全无法判断)
  },
  "context": {
    "lighting": "good" / "dim" / "dark",
    "passengers_detected": true 或 false
  },
  "overall_risk": 0-10 之间的整数，综合疲劳/分心/异常/遮挡得出的总风险评分,
  "explanation": "用中文写一段完整的多维度分析报告，覆盖你观察到的所有关键线索",
  "recommended_action": "none" / "verbal_warning" / "alarm" / "pull_over"
}
"""


# Neutral default returned on error or missing fields, so the GUI can read
# nested fields without NoneType crashes.
EMPTY_RESULT = {
    "drowsiness":  {"level": 0, "confidence": 0.0},
    "distraction": {"detected": False, "type": "none", "confidence": 0.0},
    "anomaly":     {"detected": False, "description": None, "severity": "none"},
    "occlusion":   {"type": ["none"], "impact_on_reliability": 0.0},
    "context":     {"lighting": "good", "passengers_detected": False},
    "overall_risk": 0,
    "explanation": "",
    "recommended_action": "none",
}


def _normalize(raw: dict) -> dict:
    """Fill missing fields with EMPTY_RESULT defaults so GUI never KeyErrors."""
    out = {k: (v.copy() if isinstance(v, dict) else (list(v) if isinstance(v, list) else v))
           for k, v in EMPTY_RESULT.items()}
    if not isinstance(raw, dict):
        return out
    for k, default in EMPTY_RESULT.items():
        if k not in raw:
            continue
        if isinstance(default, dict) and isinstance(raw[k], dict):
            merged = dict(default)
            merged.update(raw[k])
            out[k] = merged
        else:
            out[k] = raw[k]
    return out


@dataclass
class SlowSystemConfig:
    interval_seconds: float = 10.0
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    model_name: str = "qwen3-omni-flash"
    prompt: str = DEFAULT_PROMPT
    mock_mode: bool = True
    request_timeout: float = 30.0
    jpeg_quality: int = 85


class SlowSystem:
    """Background VLM analyzer thread."""

    def __init__(self, config: Optional[SlowSystemConfig] = None):
        self.config = config or SlowSystemConfig()

        self._frame_lock = threading.Lock()
        self._latest_frame = None  # numpy ndarray BGR

        self._result_lock = threading.Lock()
        self._latest_result: Optional[dict] = None

        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="SlowSystemWorker", daemon=True
        )

        self._client = None
        if not self.config.mock_mode and _HAS_OPENAI and self.config.api_key:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def submit_frame(self, frame) -> None:
        """Non-blocking: store the latest frame for the next analysis cycle."""
        if frame is None:
            return
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def poll_result(self) -> Optional[dict]:
        """Return the latest analysis result (or None). Non-blocking."""
        with self._result_lock:
            return self._latest_result

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        # Wait the interval first so we don't fire on cold start
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.config.interval_seconds):
                return

            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()

            if frame is None:
                continue

            t0 = time.time()
            try:
                result = self._analyze(frame)
            except Exception as exc:  # pragma: no cover - safety net
                result = dict(EMPTY_RESULT)
                result["explanation"] = f"VLM 调用失败: {exc}"
                result["source"] = "error"
            result["timestamp"] = time.time()
            result["latency_s"] = round(time.time() - t0, 3)

            with self._result_lock:
                self._latest_result = result

    # ------------------------------------------------------------------- VLM

    def _analyze(self, frame) -> dict:
        if self.config.mock_mode or self._client is None:
            return self._mock_analyze(frame)
        return self._qwen_analyze(frame)

    def _mock_analyze(self, frame) -> dict:
        """Plausible multi-dimensional random response for offline pipeline tests."""
        time.sleep(random.uniform(0.2, 0.5))  # simulate network latency

        level = random.randint(0, 8)
        distraction_types = ["phone", "eating", "talking", "looking_away", "operating", "none"]
        d_type = random.choice(distraction_types)
        distracted = d_type != "none"

        anomaly_descs = [
            None, None, None,
            "驾驶员眼神涣散、面色苍白，疑似不适",
            "驾驶员情绪激动，表情紧绷",
            "驾驶员肢体僵硬，存在身体不适迹象",
        ]
        a_desc = random.choice(anomaly_descs)
        a_severity = "none" if a_desc is None else random.choice(["low", "medium", "high"])

        occ_pool = ["mask", "sunglasses", "hat"]
        occlusions = random.sample(occ_pool, k=random.randint(0, 2)) or ["none"]
        occ_impact = round(random.uniform(0.0, 0.4) if "none" in occlusions
                           else random.uniform(0.2, 0.7), 2)

        lighting = random.choice(["good", "good", "dim", "dark"])
        passengers = random.random() < 0.3

        overall = max(level, 7 if a_desc and a_severity == "high" else 0,
                      6 if distracted else 0)
        overall = min(10, overall)

        if overall >= 8:
            action = "pull_over"
        elif overall >= 6:
            action = "alarm"
        elif overall >= 4:
            action = "verbal_warning"
        else:
            action = "none"

        explanation = (
            f"[MOCK] 综合观察：疲劳度 {level}/10；"
            f"{'检测到分心：' + d_type if distracted else '未发现明显分心'}；"
            f"{'异常: ' + a_desc if a_desc else '无异常行为'}；"
            f"遮挡 {occlusions}；车内光线 {lighting}；"
            f"{'有乘客' if passengers else '未见乘客'}。"
        )

        return _normalize({
            "drowsiness":  {"level": level, "confidence": round(random.uniform(0.65, 0.95), 2)},
            "distraction": {"detected": distracted, "type": d_type,
                            "confidence": round(random.uniform(0.6, 0.95), 2)},
            "anomaly":     {"detected": a_desc is not None,
                            "description": a_desc, "severity": a_severity},
            "occlusion":   {"type": occlusions, "impact_on_reliability": occ_impact},
            "context":     {"lighting": lighting, "passengers_detected": passengers},
            "overall_risk": overall,
            "explanation": explanation,
            "recommended_action": action,
            "source": "mock",
        })

    def _qwen_analyze(self, frame) -> dict:
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.config.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }
            ],
            timeout=self.config.request_timeout,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"VLM 返回非合法 JSON: {exc}; raw={content[:200]}"
            ) from exc
        data = _normalize(data)
        data["source"] = self.config.model_name
        return data
