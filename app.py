"""Run the offline multimodal attendance and engagement agent."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Union

import cv2

import config
from analytics_engine import (
    AudioActivityMonitor,
    AudioSample,
    CascadeFaceDetector,
    EngagementScorer,
    EngagementScore,
    FaceTracker,
    MetricsLogger,
    TrackingUpdate,
    simulate_audio_amplitude,
)
from llm_agent import LocalLLMError, LocalTeachingAgent, TeachingSuggestion


def parse_source(value: str) -> Union[int, str]:
    """Treat numeric source values as camera indexes and all others as file paths."""
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    """Build a deliberately small terminal interface."""
    parser = argparse.ArgumentParser(description="Offline multimodal attendance and engagement agent")
    parser.add_argument("--source", default="0", help="Camera index (default: 0) or local video file path")
    parser.add_argument("--no-display", action="store_true", help="Run without the OpenCV preview window")
    parser.add_argument("--no-audio", action="store_true", help="Disable microphone monitoring")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames; 0 means no limit")
    parser.add_argument("--log-csv", default="session_metrics.csv", help="Local path for aggregate metrics CSV")
    return parser


def _focus_from_assignment(assignment: object) -> float:
    focus = getattr(assignment, "focus_score", None)
    return float(focus) if isinstance(focus, (int, float)) else 50.0


def calculate_frame_engagement(
    update: TrackingUpdate,
    scorer: EngagementScorer,
    movement_history: Dict[str, Deque[float]],
    audio_level: float,
) -> EngagementScore:
    """Aggregate each active profile's visual indicators for the current frame."""
    focus_scores: List[float] = []
    all_movement: List[float] = []
    for profile_id, detection in update.assignments.items():
        focus_scores.append(_focus_from_assignment(detection))
        track = next(track for track in update.tracks if track.profile_id == profile_id)
        movement_history[profile_id].append(track.movement_pixels)
        all_movement.extend(movement_history[profile_id])
    return scorer.score(focus_scores, all_movement, audio_level)


def metrics_payload(update: TrackingUpdate, engagement: EngagementScore, audio: AudioSample) -> Dict[str, object]:
    """Build the aggregate-only payload sent to loopback Ollama."""
    active_tracks = [track for track in update.tracks if track.active]
    present_tracks = [track for track in active_tracks if track.status == "Present"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_profiles": len(active_tracks),
        "present_profiles": len(present_tracks),
        "engagement_score": engagement.score,
        "focus_score": engagement.focus_score,
        "movement_score": engagement.movement_score,
        "audio_score": engagement.audio_score,
        "audio_dbfs": round(audio.decibels_full_scale, 2),
        "sustained_inactivity": engagement.sustained_inactivity,
    }


def should_trigger_agent(engagement: EngagementScore, active_profile_count: int) -> bool:
    """Trigger only for meaningful low-engagement or multi-profile inactivity conditions."""
    low_engagement = engagement.score < config.CRITICAL_ENGAGEMENT_THRESHOLD
    widespread_inactivity = engagement.sustained_inactivity and active_profile_count >= 2
    return low_engagement or widespread_inactivity


def draw_overlay(frame: object, update: TrackingUpdate, engagement: EngagementScore) -> None:
    """Render ephemeral profile labels and aggregate status on the local preview."""
    for profile_id, detection in update.assignments.items():
        track = next(track for track in update.tracks if track.profile_id == profile_id)
        color = (0, 180, 0) if track.status == "Present" else (0, 180, 255)
        cv2.rectangle(frame, (detection.x, detection.y), (detection.x + detection.width, detection.y + detection.height), color, 2)
        label = f"{profile_id} {track.status} {track.cumulative_frames}f"
        cv2.putText(frame, label, (detection.x, max(22, detection.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    state = "INACTIVE" if engagement.sustained_inactivity else "ACTIVE"
    cv2.putText(frame, f"Engagement: {engagement.score:.1f}/100 ({state})", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def print_suggestions(suggestions: Sequence[TeachingSuggestion]) -> None:
    """Print a concise instructor-facing suggestion set."""
    print("\nLocal teaching suggestions:")
    for index, suggestion in enumerate(suggestions, start=1):
        print(f"{index}. {suggestion.title}: {suggestion.action} Why: {suggestion.rationale}")


def run(arguments: argparse.Namespace) -> int:
    """Own the capture, analytics, local-agent, and orderly shutdown lifecycle."""
    source = parse_source(arguments.source)
    if isinstance(source, str) and not Path(source).is_file():
        print(f"Video file was not found: {source}")
        return 2
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Unable to open video source: {arguments.source}")
        return 2

    try:
        detector = CascadeFaceDetector()
    except RuntimeError as error:
        capture.release()
        print(f"Unable to initialize offline face detection: {error}")
        return 2
    tracker = FaceTracker()
    scorer = EngagementScorer()
    logger = MetricsLogger()
    audio_monitor = AudioActivityMonitor()
    audio_enabled = not arguments.no_audio and audio_monitor.start()
    if not arguments.no_audio and not audio_enabled:
        print("Microphone unavailable; using deterministic audio-activity simulation.")

    agent = LocalTeachingAgent()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-teaching-agent")
    agent_future: Optional[Future[List[TeachingSuggestion]]] = None
    last_agent_trigger_at: Optional[datetime] = None
    movement_history: Dict[str, Deque[float]] = defaultdict(
        lambda: deque(maxlen=config.MOVEMENT_WINDOW_FRAMES)
    )
    frame_index = 0
    print("Running locally. Press q in the preview window to stop.")

    try:
        while True:
            captured, frame = capture.read()
            if not captured:
                break
            frame_index += 1
            detections = detector.detect(frame)
            update = tracker.update(detections)
            audio = audio_monitor.snapshot() if audio_enabled else AudioSample(
                rms=0.0,
                decibels_full_scale=config.AUDIO_SILENCE_DBFS,
                normalized_level=simulate_audio_amplitude(frame_index),
                captured_at=datetime.now(timezone.utc).isoformat(),
            )
            engagement = calculate_frame_engagement(
                update, scorer, movement_history, audio.normalized_level
            )
            logger.append_if_due(update.tracks, engagement, audio)

            now = datetime.now(timezone.utc)
            cooldown_elapsed = last_agent_trigger_at is None or (
                now - last_agent_trigger_at
            ).total_seconds() >= config.LLM_TRIGGER_COOLDOWN_SECONDS
            active_profile_count = sum(track.active for track in update.tracks)
            if agent_future is None and cooldown_elapsed and should_trigger_agent(engagement, active_profile_count):
                agent_future = executor.submit(agent.request_suggestions, metrics_payload(update, engagement, audio))
                last_agent_trigger_at = now
            if agent_future is not None and agent_future.done():
                try:
                    print_suggestions(agent_future.result())
                except LocalLLMError as error:
                    print(f"Local teaching agent unavailable: {error}")
                agent_future = None

            if not arguments.no_display:
                draw_overlay(frame, update, engagement)
                cv2.imshow("Multimodal Attendance & Engagement Agent", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if arguments.max_frames and frame_index >= arguments.max_frames:
                break
    except KeyboardInterrupt:
        print("Stopping on keyboard interrupt.")
    finally:
        capture.release()
        audio_monitor.stop()
        executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()

    metrics = logger.dataframe()
    metrics.to_csv(arguments.log_csv, index=False)
    attendance = tracker.attendance_dataframe()
    attendance_path = Path(arguments.log_csv).with_name("attendance_" + Path(arguments.log_csv).name)
    attendance.to_csv(attendance_path, index=False)
    print(f"Saved aggregate metrics to {Path(arguments.log_csv).resolve()}")
    print(f"Saved session-local attendance to {attendance_path.resolve()}")
    return 0


def main() -> int:
    """Parse arguments and return a process exit code."""
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
