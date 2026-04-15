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


DEFAULT_PROMPT = """You are a professional driver-state analysis expert. Perform a **multi-dimensional** analysis of the driver in the image.

Analyse the following five dimensions in parallel:
1. **Drowsiness** — do not look only at eye open/close. Combine facial muscle slackness, head tilt angle, overall posture (shoulder/neck slump), and expression tension.
2. **Distraction** — determine whether the driver is on the phone, eating, talking to passengers, looking down at the console/objects, looking away, etc.
3. **Anomaly** — signs of possible drug/alcohol intoxication (unfocused gaze, abnormal complexion, uncoordinated movement), emotional distress (crying, anger, panic), or physical discomfort (clutching chest, head in hand, stiffness).
4. **Occlusion** — identify mask / sunglasses / hat and clearly state whether these occlusions reduce your judgment confidence.
5. **Scene context** — cabin lighting condition, whether passengers are visible, and any other inferable driving environment info.

**Return STRICT JSON** in the schema below. Field names must match exactly. Do NOT add markdown, comments, or any extra text. Missing fields will break downstream parsing.

{
  "drowsiness": {
    "level": integer 0-10 (0 = fully alert, 10 = extremely drowsy),
    "confidence": float 0-1
  },
  "distraction": {
    "detected": true or false,
    "type": "phone" / "eating" / "talking" / "looking_away" / "operating" / "other" / "none",
    "confidence": 0-1
  },
  "anomaly": {
    "detected": true or false,
    "description": "one-sentence description if detected; otherwise null",
    "severity": "low" / "medium" / "high" / "none"
  },
  "occlusion": {
    "type": [list of "mask" / "sunglasses" / "hat" / "none"; use ["none"] if no occlusion],
    "impact_on_reliability": float 0-1 (0 = no impact, 1 = judgment impossible)
  },
  "context": {
    "lighting": "good" / "dim" / "dark",
    "passengers_detected": true or false
  },
  "overall_risk": integer 0-10, blended from drowsiness/distraction/anomaly/occlusion,
  "explanation": "one full paragraph in ENGLISH covering all key cues you observed",
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
    # Minimum gap between request *starts*. If <= 0 the worker fires
    # back-to-back, bounded only by VLM round-trip latency (max throughput).
    interval_seconds: float = 10.0
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    model_name: str = "qwen3-omni-flash"
    prompt: str = DEFAULT_PROMPT
    mock_mode: bool = True
    request_timeout: float = 30.0
    jpeg_quality: int = 80
    # Image sent to the VLM is downscaled so that max(width, height) fits
    # inside this box, which cuts upload size ~4x at 480 vs raw 640. Smaller
    # image → faster round-trip → higher effective sample rate.
    image_max_side: int = 480


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
        while not self._stop_event.is_set():
            cycle_start = time.time()

            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()

            if frame is None:
                # No frame yet — short yield to avoid a tight CPU loop.
                if self._stop_event.wait(0.1):
                    return
                continue

            try:
                result = self._analyze(frame)
            except Exception as exc:  # pragma: no cover - safety net
                result = dict(EMPTY_RESULT)
                result["explanation"] = f"VLM call failed: {exc}"
                result["source"] = "error"
            result["timestamp"] = time.time()
            result["latency_s"] = round(time.time() - cycle_start, 3)

            with self._result_lock:
                self._latest_result = result

            # Floor the inter-request gap with `interval_seconds`. If the
            # VLM call already took longer than the interval the remaining
            # wait is zero — effectively back-to-back and rate-limited only
            # by VLM latency. Set interval_seconds <= 0 for max throughput.
            elapsed = time.time() - cycle_start
            remaining = self.config.interval_seconds - elapsed
            if remaining > 0:
                if self._stop_event.wait(remaining):
                    return

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
            "Driver's gaze unfocused and complexion pale — possible unwellness",
            "Driver appears emotionally agitated, facial tension elevated",
            "Driver's posture is rigid — possible physical discomfort",
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
            f"[MOCK] Composite observation — drowsiness {level}/10; "
            f"{'distraction detected: ' + d_type if distracted else 'no distraction detected'}; "
            f"{'anomaly: ' + a_desc if a_desc else 'no anomaly'}; "
            f"occlusion {occlusions}; cabin lighting {lighting}; "
            f"{'passengers present' if passengers else 'no passengers visible'}."
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
        # Pre-scale the frame so the longest side fits image_max_side.
        # Cuts upload bytes ~4x going from 640 → 480 and ~9x going 640 → 320,
        # directly reducing round-trip time.
        h, w = frame.shape[:2]
        max_side = max(1, int(self.config.image_max_side))
        longest = max(h, w)
        if longest > max_side:
            scale = max_side / longest
            frame = cv2.resize(
                frame,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

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
                f"VLM returned invalid JSON: {exc}; raw={content[:200]}"
            ) from exc
        data = _normalize(data)
        data["source"] = self.config.model_name
        return data
