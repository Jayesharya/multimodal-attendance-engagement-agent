"""Central configuration for the offline attendance and engagement agent."""

from __future__ import annotations

from dataclasses import dataclass


# Attendance and tracking. Identities are ephemeral labels for one process only.
PRESENCE_FRAME_THRESHOLD = 15
MAX_MISSING_FRAMES = 20
TRACKING_MATCH_DISTANCE_PIXELS = 120.0
METRICS_LOG_INTERVAL_SECONDS = 5.0

# Engagement model. The weights are deliberately explicit and sum to 1.0.
FOCUS_WEIGHT = 0.55
MOVEMENT_WEIGHT = 0.30
AUDIO_WEIGHT = 0.15
MOVEMENT_WINDOW_FRAMES = 30
MOVEMENT_REFERENCE_PIXELS = 5.0
INACTIVITY_MOVEMENT_THRESHOLD_PIXELS = 0.8
INACTIVITY_FOCUS_THRESHOLD = 35.0
INACTIVITY_AUDIO_THRESHOLD = 15.0
CRITICAL_ENGAGEMENT_THRESHOLD = 50.0
SUSTAINED_INACTIVITY_PROFILE_RATIO = 0.60

# Audio is normalized to the 0-1 range from RMS dBFS measurements.
AUDIO_SAMPLE_RATE = 16_000
AUDIO_BLOCK_SIZE = 1_024
AUDIO_WINDOW_SECONDS = 1.0
AUDIO_SILENCE_DBFS = -55.0

# The Ollama endpoint is deliberately loopback-only to preserve offline operation.
OLLAMA_GENERATE_ENDPOINT = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL_TAG = "phi3:mini"
OLLAMA_REQUEST_TIMEOUT_SECONDS = 45
LLM_TRIGGER_COOLDOWN_SECONDS = 60.0

# Streamlit dashboard defaults.
DASHBOARD_DEFAULT_MAX_FRAMES = 300
DASHBOARD_FRAME_UPDATE_INTERVAL = 5

# OpenCV cascade settings.
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE_PIXELS = (48, 48)
EYE_SCALE_FACTOR = 1.1
EYE_MIN_NEIGHBORS = 4


@dataclass(frozen=True)
class EngagementWeights:
    """Weights for the bounded engagement score."""

    focus: float = FOCUS_WEIGHT
    movement: float = MOVEMENT_WEIGHT
    audio: float = AUDIO_WEIGHT

    def __post_init__(self) -> None:
        total = self.focus + self.movement + self.audio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Engagement weights must sum to 1.0.")

