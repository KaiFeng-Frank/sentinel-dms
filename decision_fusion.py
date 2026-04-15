"""
Decision Fusion: combine the Fast System (real-time PERCLOS / EAR /
microsleeps / yawns from YOLO + MediaPipe) with the Slow System (VLM-based
multi-dimensional reasoning) into a single drowsiness decision.

**Scope**: this module ONLY fuses the *drowsiness* dimension. The Slow
System's other outputs (distraction / anomaly / occlusion / context /
recommended_action) are unique VLM-only capabilities and are passed through
to the GUI directly without fusion — Fast System has no equivalent signal.

Weighting rule (driven by Fast System self-reported confidence):

    fast_conf >= 0.8           →  fast 0.8 / slow 0.2
    fast_conf <= 0.5           →  fast 0.2 / slow 0.8
    in between (0.5..0.8)      →  linear interpolation

If the Slow System has no drowsiness result yet, or its result is older
than `slow_max_age_s`, we fall back to Fast-only (weight 1.0 / 0.0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FusionResult:
    drowsiness_level: float        # 0..10
    risk_label: str                # "NORMAL" / "MILD FATIGUE" / "MODERATE FATIGUE" / "SEVERE FATIGUE"
    fast_weight: float
    slow_weight: float
    used_slow: bool
    fast_level: float
    slow_level: Optional[float]
    explanation: str


class DecisionFusion:
    def __init__(self, slow_max_age_s: float = 30.0):
        self.slow_max_age_s = slow_max_age_s

    @staticmethod
    def _weights(fast_conf: float) -> Tuple[float, float]:
        """Map fast confidence to (fast_weight, slow_weight)."""
        if fast_conf >= 0.8:
            return 0.8, 0.2
        if fast_conf <= 0.5:
            return 0.2, 0.8
        # Linear interpolation between (0.5 → 0.2) and (0.8 → 0.8)
        t = (fast_conf - 0.5) / (0.8 - 0.5)
        fast_w = 0.2 + t * (0.8 - 0.2)
        return fast_w, 1.0 - fast_w

    @staticmethod
    def _label(level: float) -> str:
        if level < 3:
            return "NORMAL"
        if level < 5:
            return "MILD FATIGUE"
        if level < 7:
            return "MODERATE FATIGUE"
        return "SEVERE FATIGUE"

    @staticmethod
    def _slow_drowsiness(slow_state: Optional[dict]) -> Optional[float]:
        """Extract slow drowsiness level from the new multi-dim schema."""
        if not isinstance(slow_state, dict):
            return None
        d = slow_state.get("drowsiness")
        if not isinstance(d, dict):
            return None
        level = d.get("level")
        if level is None:
            return None
        try:
            return float(level)
        except (TypeError, ValueError):
            return None

    def fuse(
        self,
        fast_state: dict,
        slow_state: Optional[dict],
    ) -> FusionResult:
        fast_level = float(fast_state.get("drowsiness_level", 0.0))
        fast_conf = float(fast_state.get("confidence", 0.0))

        slow_level = self._slow_drowsiness(slow_state)
        slow_fresh = (
            slow_level is not None
            and (time.time() - float(slow_state.get("timestamp", 0)))
            < self.slow_max_age_s
        )

        if not slow_fresh:
            return FusionResult(
                drowsiness_level=fast_level,
                risk_label=self._label(fast_level),
                fast_weight=1.0,
                slow_weight=0.0,
                used_slow=False,
                fast_level=fast_level,
                slow_level=None,
                explanation="Slow System has no recent drowsiness sample; relying on Fast System only.",
            )

        fw, sw = self._weights(fast_conf)
        fused = fw * fast_level + sw * slow_level

        return FusionResult(
            drowsiness_level=fused,
            risk_label=self._label(fused),
            fast_weight=fw,
            slow_weight=sw,
            used_slow=True,
            fast_level=fast_level,
            slow_level=slow_level,
            explanation=slow_state.get("explanation", ""),
        )
