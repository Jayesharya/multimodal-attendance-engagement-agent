"""Offline vision, audio, attendance, and engagement computations.

This module intentionally keeps all face labels process-local. It does not
create face embeddings, perform recognition, or persist biometric imagery.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math

import cv2
import numpy as np
import pandas as pd

import config


Point = Tuple[float, float]


def utc_now_iso() -> str:
    """Return a timezone-aware ISO 8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, lower: float, upper: float) -> float:
    """Bound a float without changing its type."""
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class FaceDetection:
    """A face box and optionally inferred landmark positions in image pixels."""

    x: int
    y: int
    width: int
    height: int
    confidence: float
    landmarks: Mapping[str, Point] = field(default_factory=dict)
    focus_score: Optional[float] = None

    @property
    def center(self) -> Point:
        return (self.x + (self.width / 2.0), self.y + (self.height / 2.0))


@dataclass
class TrackedFace:
    """Ephemeral tracking state for a single detection trajectory."""

    profile_id: str
    last_center: Point
    first_seen_at: str
    last_seen_at: str
    cumulative_frames: int = 0
    consecutive_frames: int = 0
    missing_frames: int = 0
    movement_pixels: float = 0.0
    status: str = "Absent"
    active: bool = True


@dataclass(frozen=True)
class TrackingUpdate:
    """Result from associating a frame's detections to active tracks."""

    tracks: Sequence[TrackedFace]
    assignments: Mapping[str, FaceDetection]


class FaceTracker:
    """Centroid tracker with cumulative sustained-presence attendance logic."""

    def __init__(
        self,
        presence_frame_threshold: int = config.PRESENCE_FRAME_THRESHOLD,
        max_missing_frames: int = config.MAX_MISSING_FRAMES,
        match_distance_pixels: float = config.TRACKING_MATCH_DISTANCE_PIXELS,
    ) -> None:
        if presence_frame_threshold < 1:
            raise ValueError("presence_frame_threshold must be at least one.")
        self.presence_frame_threshold = presence_frame_threshold
        self.max_missing_frames = max_missing_frames
        self.match_distance_pixels = match_distance_pixels
        self._tracks: Dict[str, TrackedFace] = {}
        self._next_profile_number = 1

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        return math.dist(first, second)

    def _new_profile_id(self) -> str:
        profile_id = f"PROFILE-{self._next_profile_number:03d}"
        self._next_profile_number += 1
        return profile_id

    def update(
        self, detections: Sequence[FaceDetection], timestamp: Optional[str] = None
    ) -> TrackingUpdate:
        """Associate detections and advance attendance state for one video frame."""
        observed_at = timestamp or utc_now_iso()
        active_tracks = [track for track in self._tracks.values() if track.active]
        unmatched_tracks = {track.profile_id for track in active_tracks}
        unmatched_detection_indexes = set(range(len(detections)))
        matched_pairs: List[Tuple[str, int]] = []

        candidates: List[Tuple[float, str, int]] = []
        for track in active_tracks:
            for detection_index, detection in enumerate(detections):
                distance = self._distance(track.last_center, detection.center)
                if distance <= self.match_distance_pixels:
                    candidates.append((distance, track.profile_id, detection_index))
        for _, profile_id, detection_index in sorted(candidates):
            if profile_id in unmatched_tracks and detection_index in unmatched_detection_indexes:
                matched_pairs.append((profile_id, detection_index))
                unmatched_tracks.remove(profile_id)
                unmatched_detection_indexes.remove(detection_index)

        assignments: Dict[str, FaceDetection] = {}
        for profile_id, detection_index in matched_pairs:
            track = self._tracks[profile_id]
            detection = detections[detection_index]
            movement = self._distance(track.last_center, detection.center)
            track.last_center = detection.center
            track.last_seen_at = observed_at
            track.cumulative_frames += 1
            track.consecutive_frames += 1
            track.missing_frames = 0
            track.movement_pixels = movement
            if track.cumulative_frames >= self.presence_frame_threshold:
                track.status = "Present"
            assignments[profile_id] = detection

        for detection_index in sorted(unmatched_detection_indexes):
            detection = detections[detection_index]
            profile_id = self._new_profile_id()
            track = TrackedFace(
                profile_id=profile_id,
                last_center=detection.center,
                first_seen_at=observed_at,
                last_seen_at=observed_at,
                cumulative_frames=1,
                consecutive_frames=1,
            )
            if self.presence_frame_threshold == 1:
                track.status = "Present"
            self._tracks[profile_id] = track
            assignments[profile_id] = detection

        for profile_id in unmatched_tracks:
            track = self._tracks[profile_id]
            track.missing_frames += 1
            track.consecutive_frames = 0
            track.movement_pixels = 0.0
            if track.missing_frames > self.max_missing_frames:
                track.active = False

        return TrackingUpdate(tracks=list(self._tracks.values()), assignments=assignments)

    def attendance_dataframe(self) -> pd.DataFrame:
        """Expose process-local attendance state as a structured DataFrame."""
        columns = [
            "profile_id",
            "status",
            "cumulative_frames",
            "consecutive_frames",
            "first_seen_at",
            "last_seen_at",
            "active",
        ]
        rows = [
            {
                "profile_id": track.profile_id,
                "status": track.status,
                "cumulative_frames": track.cumulative_frames,
                "consecutive_frames": track.consecutive_frames,
                "first_seen_at": track.first_seen_at,
                "last_seen_at": track.last_seen_at,
                "active": track.active,
            }
            for track in self._tracks.values()
        ]
        return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True)
class HeadPose:
    """Approximate head orientation, bounded to a conservative visual-focus score."""

    yaw_degrees: float
    pitch_degrees: float
    focus_score: float


def estimate_head_pose(landmarks: Mapping[str, Point]) -> HeadPose:
    """Estimate pose from left/right eye, nose, forehead, and chin positions.

    The geometry is intentionally approximate: it is an engagement indicator,
    not a biometric or medical measurement.
    """
    required = {"left_eye", "right_eye", "nose", "forehead", "chin"}
    if not required.issubset(landmarks):
        return HeadPose(yaw_degrees=0.0, pitch_degrees=0.0, focus_score=50.0)

    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    nose = landmarks["nose"]
    forehead = landmarks["forehead"]
    chin = landmarks["chin"]
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    eye_distance = max(abs(right_eye[0] - left_eye[0]), 1e-6)
    face_height = max(abs(chin[1] - forehead[1]), 1e-6)

    yaw_degrees = clamp(((nose[0] - eye_mid_x) / (eye_distance / 2.0)) * 30.0, -60.0, 60.0)
    vertical_midpoint = (forehead[1] + chin[1]) / 2.0
    pitch_degrees = clamp(((nose[1] - vertical_midpoint) / (face_height / 2.0)) * 30.0, -45.0, 45.0)
    focus_score = clamp(100.0 - (abs(yaw_degrees) * 1.1) - (abs(pitch_degrees) * 0.9), 0.0, 100.0)
    return HeadPose(yaw_degrees=yaw_degrees, pitch_degrees=pitch_degrees, focus_score=focus_score)


def estimate_pose_from_face_and_eyes(
    face: FaceDetection, eye_centers: Sequence[Point]
) -> HeadPose:
    """Build approximate landmarks from a Haar face box and detected eyes."""
    if len(eye_centers) < 2:
        return HeadPose(yaw_degrees=0.0, pitch_degrees=0.0, focus_score=50.0)
    eyes = sorted(eye_centers[:2], key=lambda point: point[0])
    face_center_x = face.x + (face.width / 2.0)
    face_center_y = face.y + (face.height / 2.0)
    eye_mid_x = (eyes[0][0] + eyes[1][0]) / 2.0
    eye_mid_y = (eyes[0][1] + eyes[1][1]) / 2.0
    inferred_nose = (face_center_x + ((eye_mid_x - face_center_x) * 0.55), face_center_y)
    landmarks = {
        "left_eye": eyes[0],
        "right_eye": eyes[1],
        "nose": inferred_nose,
        "forehead": (face_center_x, face.y + (face.height * 0.16)),
        "chin": (face_center_x, face.y + (face.height * 0.84)),
    }
    return estimate_head_pose(landmarks)


class CascadeFaceDetector:
    """Fast offline Haar cascade face and eye detector supplied by OpenCV."""

    def __init__(self) -> None:
        cascade_directory = Path(cv2.data.haarcascades)
        self._face_cascade = cv2.CascadeClassifier(
            str(cascade_directory / "haarcascade_frontalface_default.xml")
        )
        self._eye_cascade = cv2.CascadeClassifier(str(cascade_directory / "haarcascade_eye.xml"))
        if self._face_cascade.empty() or self._eye_cascade.empty():
            raise RuntimeError("OpenCV Haar cascade data could not be loaded.")

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces and infer eye-based landmarks without saving image data."""
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=config.FACE_SCALE_FACTOR,
            minNeighbors=config.FACE_MIN_NEIGHBORS,
            minSize=config.FACE_MIN_SIZE_PIXELS,
        )
        detections: List[FaceDetection] = []
        for x, y, width, height in boxes:
            face_region = grayscale[y : y + height, x : x + width]
            eye_boxes = self._eye_cascade.detectMultiScale(
                face_region,
                scaleFactor=config.EYE_SCALE_FACTOR,
                minNeighbors=config.EYE_MIN_NEIGHBORS,
            )
            eye_centers = [
                (x + eye_x + (eye_width / 2.0), y + eye_y + (eye_height / 2.0))
                for eye_x, eye_y, eye_width, eye_height in eye_boxes
            ]
            detection = FaceDetection(int(x), int(y), int(width), int(height), 1.0)
            pose = estimate_pose_from_face_and_eyes(detection, eye_centers)
            landmarks = {
                "left_eye": eye_centers[0],
                "right_eye": eye_centers[1],
                "nose": (detection.center[0], detection.center[1]),
                "forehead": (detection.center[0], y + (height * 0.16)),
                "chin": (detection.center[0], y + (height * 0.84)),
            } if len(eye_centers) >= 2 else {}
            detections.append(
                FaceDetection(
                    int(x), int(y), int(width), int(height), 1.0,
                    landmarks=landmarks,
                    focus_score=pose.focus_score,
                )
            )
        return detections


class MediaPipeFaceLandmarker:
    """Optional offline landmark estimator backed by a local MediaPipe task model."""

    def __init__(self, model_path: str) -> None:
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"MediaPipe face landmarker model was not found: {model_file}")
        try:
            import mediapipe as media_pipe
        except ImportError as error:
            raise RuntimeError("MediaPipe is not installed. Install requirements.txt first.") from error

        self._media_pipe = media_pipe
        options = media_pipe.tasks.vision.FaceLandmarkerOptions(
            base_options=media_pipe.tasks.BaseOptions(model_asset_path=str(model_file)),
            running_mode=media_pipe.tasks.vision.RunningMode.IMAGE,
            num_faces=10,
        )
        self._landmarker = media_pipe.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect_poses(self, frame: np.ndarray) -> List[HeadPose]:
        """Return approximate poses for a BGR frame without retaining landmark data."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._media_pipe.Image(
            image_format=self._media_pipe.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self._landmarker.detect(image)
        height, width = frame.shape[:2]
        poses: List[HeadPose] = []
        for face_landmarks in result.face_landmarks:
            landmarks = {
                "left_eye": (face_landmarks[33].x * width, face_landmarks[33].y * height),
                "right_eye": (face_landmarks[263].x * width, face_landmarks[263].y * height),
                "nose": (face_landmarks[1].x * width, face_landmarks[1].y * height),
                "forehead": (face_landmarks[10].x * width, face_landmarks[10].y * height),
                "chin": (face_landmarks[152].x * width, face_landmarks[152].y * height),
            }
            poses.append(estimate_head_pose(landmarks))
        return poses

    def close(self) -> None:
        """Release MediaPipe's native resources."""
        self._landmarker.close()


@dataclass(frozen=True)
class AudioSample:
    """One bounded room-audio activity estimate."""

    rms: float
    decibels_full_scale: float
    normalized_level: float
    captured_at: str
    activity_label: str = "quiet"
    speech_likelihood: float = 0.0


@dataclass(frozen=True)
class AudioWindow:
    """Audio activity features associated with one elapsed-time interval."""

    start_seconds: float
    end_seconds: float
    sample: AudioSample


@dataclass(frozen=True)
class AudioActivity:
    """Lightweight, non-semantic audio activity classification."""

    label: str
    speech_likelihood: float


def classify_audio_activity(samples: np.ndarray, sample_rate: int) -> AudioActivity:
    """Classify quiet, speech-like, or noise-like activity from RMS and ZCR.

    This is deliberately an activity heuristic, not speech recognition and not
    a claim about the content or identity of a speaker.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return AudioActivity(label="quiet", speech_likelihood=0.0)
    rms = float(np.sqrt(np.mean(np.square(values))))
    dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
    if dbfs <= -45.0:
        return AudioActivity(label="quiet", speech_likelihood=0.0)
    zero_crossing_rate = float(np.mean(np.abs(np.diff(np.signbit(values)))))
    speech_band_score = clamp(1.0 - (abs(zero_crossing_rate - 0.08) / 0.20), 0.0, 1.0)
    level_score = clamp((dbfs + 45.0) / 30.0, 0.0, 1.0)
    speech_likelihood = round(speech_band_score * level_score, 3)
    if 0.005 <= zero_crossing_rate <= 0.25 and speech_likelihood >= 0.35:
        return AudioActivity(label="speech_likely", speech_likelihood=speech_likelihood)
    return AudioActivity(label="ambient_noise", speech_likelihood=speech_likelihood)


def analyze_audio_samples(samples: np.ndarray, sample_rate: int) -> AudioSample:
    """Convert in-memory mono samples into an aggregate activity measure."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if samples.size == 0:
        return AudioSample(0.0, config.AUDIO_SILENCE_DBFS, 0.0, utc_now_iso())
    normalized_samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(normalized_samples))))
    dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
    level = clamp((dbfs - config.AUDIO_SILENCE_DBFS) / abs(config.AUDIO_SILENCE_DBFS), 0.0, 1.0)
    activity = classify_audio_activity(normalized_samples, sample_rate)
    return AudioSample(rms, dbfs, level, utc_now_iso(), activity.label, activity.speech_likelihood)


def analyze_audio_file(file_path: str) -> AudioSample:
    """Load an optional local audio file with Librosa and discard samples after analysis."""
    source = Path(file_path)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file was not found: {source}")
    try:
        import librosa
    except ImportError as error:
        raise RuntimeError("Librosa is not installed. Install requirements.txt first.") from error
    samples, sample_rate = librosa.load(str(source), sr=None, mono=True)
    return analyze_audio_samples(samples, int(sample_rate))


def analyze_audio_file_windows(
    file_path: str,
    segment_seconds: float = config.AUDIO_WINDOW_SECONDS,
) -> List[AudioWindow]:
    """Analyze local audio in fixed windows suitable for video-time alignment."""
    source = Path(file_path)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file was not found: {source}")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive.")
    try:
        import librosa
    except ImportError as error:
        raise RuntimeError("Librosa is not installed. Install requirements.txt first.") from error

    samples, sample_rate = librosa.load(
        str(source), sr=config.AUDIO_SAMPLE_RATE, mono=True
    )
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    window_size = max(1, int(sample_rate * segment_seconds))
    windows: List[AudioWindow] = []
    for start_index in range(0, samples.size, window_size):
        end_index = min(start_index + window_size, samples.size)
        windows.append(
            AudioWindow(
                start_seconds=start_index / sample_rate,
                end_seconds=end_index / sample_rate,
                sample=analyze_audio_samples(samples[start_index:end_index], sample_rate),
            )
        )
    return windows


def audio_sample_at(
    windows: Sequence[AudioWindow], elapsed_seconds: float
) -> AudioSample:
    """Return the audio features for a time offset, or silence outside the file."""
    for window in windows:
        if window.start_seconds <= elapsed_seconds < window.end_seconds:
            return window.sample
    return AudioSample(
        rms=0.0,
        decibels_full_scale=config.AUDIO_SILENCE_DBFS,
        normalized_level=0.0,
        captured_at=utc_now_iso(),
        activity_label="quiet",
        speech_likelihood=0.0,
    )


class AudioActivityMonitor:
    """Best-effort microphone RMS monitor; it never records audio to disk."""

    def __init__(self, sample_rate: int = config.AUDIO_SAMPLE_RATE, block_size: int = config.AUDIO_BLOCK_SIZE) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._latest = AudioSample(0.0, config.AUDIO_SILENCE_DBFS, 0.0, utc_now_iso())
        self._lock = Lock()
        self._stream: object | None = None
        self.available = False

    def _on_audio(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        del frames, time_info, status
        sample = analyze_audio_samples(indata, self.sample_rate)
        with self._lock:
            self._latest = sample

    def start(self) -> bool:
        """Start a non-persistent microphone stream and report whether it opened."""
        try:
            import sounddevice as sound_device

            stream = sound_device.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=self._on_audio,
            )
            stream.start()
            self._stream = stream
            self.available = True
        except Exception:
            self._stream = None
            self.available = False
        return self.available

    def snapshot(self) -> AudioSample:
        """Return the most recent aggregate value, never raw audio."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        """Close the microphone resource if it was successfully opened."""
        stream = self._stream
        self._stream = None
        self.available = False
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


def simulate_audio_amplitude(frame_index: int) -> float:
    """Deterministic fallback for demos without an accessible microphone."""
    return clamp(0.35 + (0.25 * math.sin(frame_index / 15.0)), 0.0, 1.0)


@dataclass(frozen=True)
class EngagementScore:
    """A transparent engagement score and its component measurements."""

    score: float
    focus_score: float
    movement_score: float
    audio_score: float
    sustained_inactivity: bool


class EngagementScorer:
    """Fuse focus, movement, and room-audio activity into a 0-100 score."""

    def __init__(
        self,
        movement_window: int = config.MOVEMENT_WINDOW_FRAMES,
        weights: config.EngagementWeights = config.EngagementWeights(),
    ) -> None:
        if movement_window < 1:
            raise ValueError("movement_window must be at least one.")
        self.movement_window = movement_window
        self.weights = weights

    def score(
        self,
        focus_scores: Iterable[float],
        movement_values: Iterable[float],
        audio_level: float,
    ) -> EngagementScore:
        """Score current signals while treating absent faces as no visual evidence."""
        focus = [clamp(float(value), 0.0, 100.0) for value in focus_scores]
        movement = [max(0.0, float(value)) for value in movement_values]
        focus_score = float(np.mean(focus)) if focus else 0.0
        recent_movement = movement[-self.movement_window :]
        mean_movement = float(np.mean(recent_movement)) if recent_movement else 0.0
        movement_score = clamp(
            (mean_movement / config.MOVEMENT_REFERENCE_PIXELS) * 100.0, 0.0, 100.0
        )
        sustained_inactivity = bool(recent_movement) and (
            len(recent_movement) >= self.movement_window
            and mean_movement <= config.INACTIVITY_MOVEMENT_THRESHOLD_PIXELS
            and focus_score <= config.INACTIVITY_FOCUS_THRESHOLD
            and audio_level * 100.0 <= config.INACTIVITY_AUDIO_THRESHOLD
        )
        audio_score = clamp(audio_level, 0.0, 1.0) * 100.0
        movement_contribution = 50.0 + movement_score * 0.1
        combined = (
            (focus_score * self.weights.focus)
            + (movement_contribution * self.weights.movement)
            + (audio_score * self.weights.audio)
        )
        return EngagementScore(
            score=round(clamp(combined, 0.0, 100.0), 2),
            focus_score=round(focus_score, 2),
            movement_score=round(movement_score, 2),
            audio_score=round(audio_score, 2),
            sustained_inactivity=sustained_inactivity,
        )


class MetricsLogger:
    """Periodic, structured in-memory event log suitable for CSV export."""

    def __init__(self, interval_seconds: float = config.METRICS_LOG_INTERVAL_SECONDS) -> None:
        self.interval_seconds = interval_seconds
        self._last_logged_at: Optional[datetime] = None
        self._records: List[Dict[str, object]] = []

    def append_if_due(
        self,
        tracked_faces: Sequence[TrackedFace],
        engagement: EngagementScore,
        audio: AudioSample,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Append one aggregate event at the configured cadence."""
        observed_at = timestamp or datetime.now(timezone.utc)
        if self._last_logged_at is not None:
            elapsed = (observed_at - self._last_logged_at).total_seconds()
            if elapsed < self.interval_seconds:
                return False
        present_count = sum(face.status == "Present" for face in tracked_faces)
        self._records.append(
            {
                "timestamp": observed_at.isoformat(),
                "detected_faces": len([face for face in tracked_faces if face.active]),
                "present_profiles": present_count,
                "engagement_score": engagement.score,
                "focus_score": engagement.focus_score,
                "movement_score": engagement.movement_score,
                "audio_score": engagement.audio_score,
                "audio_dbfs": audio.decibels_full_scale,
                "audio_activity": audio.activity_label,
                "speech_likelihood": audio.speech_likelihood,
                "sustained_inactivity": engagement.sustained_inactivity,
            }
        )
        self._last_logged_at = observed_at
        return True

    def dataframe(self) -> pd.DataFrame:
        """Return structured logs without performing any I/O."""
        return pd.DataFrame(self._records)


def generate_session_analytics(
    attendance: pd.DataFrame,
    metrics: pd.DataFrame,
    expected_participants: Optional[int] = None,
) -> Dict[str, float | int | None]:
    """Summarize tracked IDs separately from roster-based attendance."""
    if expected_participants is not None and expected_participants < 1:
        raise ValueError("expected_participants must be positive when provided.")
    tracked_profiles = int(len(attendance.index))
    present_profiles = int((attendance.get("status", pd.Series(dtype=str)) == "Present").sum())
    track_confirmation_rate = (
        present_profiles / tracked_profiles * 100.0 if tracked_profiles else 0.0
    )
    attendance_rate = (
        present_profiles / expected_participants * 100.0
        if expected_participants is not None
        else None
    )
    engagement_column = metrics.get("engagement_score", pd.Series(dtype=float))
    mean_engagement = float(engagement_column.mean()) if not engagement_column.empty else 0.0
    minimum_engagement = float(engagement_column.min()) if not engagement_column.empty else 0.0
    return {
        "tracked_profiles": tracked_profiles,
        "present_profiles": present_profiles,
        "track_confirmation_rate": round(track_confirmation_rate, 2),
        "expected_participants": expected_participants,
        "attendance_rate": round(attendance_rate, 2) if attendance_rate is not None else None,
        "mean_engagement": round(mean_engagement, 2),
        "minimum_engagement": round(minimum_engagement, 2),
    }
