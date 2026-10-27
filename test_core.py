"""Behavioral tests for the privacy-preserving analytics and local agent."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analytics_engine import (
    EngagementScorer,
    AudioSample,
    AudioWindow,
    FaceDetection,
    FaceTracker,
    analyze_audio_samples,
    analyze_audio_file_windows,
    audio_sample_at,
    classify_audio_activity,
    estimate_head_pose,
    generate_session_analytics,
)
from llm_agent import LocalTeachingAgent
from reporting import build_session_pdf
from session_store import (
    clear_history,
    list_sessions,
    load_session,
    save_session,
    update_session_roster,
)


class FaceTrackerTests(unittest.TestCase):
    def test_cumulative_observations_mark_attendance_only_at_threshold(self) -> None:
        """Removing cumulative presence must keep a participant absent."""
        tracker = FaceTracker(presence_frame_threshold=3, max_missing_frames=1)
        detection = FaceDetection(10, 10, 80, 80, confidence=0.99)

        tracker.update([detection], timestamp="2026-10-24T09:00:00+00:00")
        tracker.update([detection], timestamp="2026-10-24T09:00:01+00:00")
        self.assertEqual(tracker.attendance_dataframe().iloc[0]["status"], "Absent")

        tracker.update([], timestamp="2026-10-24T09:00:02+00:00")
        tracker.update([detection], timestamp="2026-10-24T09:00:03+00:00")
        attendance = tracker.attendance_dataframe()
        self.assertEqual(attendance.iloc[0]["status"], "Present")
        self.assertEqual(int(attendance.iloc[0]["cumulative_frames"]), 3)


class EngagementTests(unittest.TestCase):
    def test_audio_file_is_loaded_as_sequential_one_second_windows(self) -> None:
        sample_rate = 16_000
        time_axis = np.arange(sample_rate * 2) / sample_rate
        samples = 0.2 * np.sin(2.0 * np.pi * 200.0 * time_axis)
        pcm_samples = np.int16(samples * 32_767)

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            with wave.open(str(audio_path), "wb") as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(2)
                audio_file.setframerate(sample_rate)
                audio_file.writeframes(pcm_samples.tobytes())

            windows = analyze_audio_file_windows(str(audio_path))

        self.assertEqual(len(windows), 2)
        self.assertEqual((windows[0].start_seconds, windows[0].end_seconds), (0.0, 1.0))
        self.assertEqual((windows[1].start_seconds, windows[1].end_seconds), (1.0, 2.0))
        self.assertEqual(windows[0].sample.activity_label, "speech_likely")

    def test_audio_window_selection_tracks_elapsed_time_and_silence(self) -> None:
        first_sample = AudioSample(0.2, -14.0, 0.7, "start", "speech_likely", 0.8)
        second_sample = AudioSample(0.0, -55.0, 0.0, "next", "quiet", 0.0)
        windows = [
            AudioWindow(0.0, 1.0, first_sample),
            AudioWindow(1.0, 2.0, second_sample),
        ]

        self.assertIs(audio_sample_at(windows, 0.5), first_sample)
        self.assertIs(audio_sample_at(windows, 1.5), second_sample)
        self.assertEqual(audio_sample_at(windows, 2.0).activity_label, "quiet")

    def test_symmetric_landmarks_produce_near_centered_head_pose(self) -> None:
        """Swapping a centered face for an asymmetric one must alter focus."""
        pose = estimate_head_pose(
            {
                "left_eye": (0.35, 0.40),
                "right_eye": (0.65, 0.40),
                "nose": (0.50, 0.50),
                "forehead": (0.50, 0.25),
                "chin": (0.50, 0.75),
            }
        )
        self.assertAlmostEqual(pose.yaw_degrees, 0.0, delta=0.01)
        self.assertAlmostEqual(pose.pitch_degrees, 0.0, delta=0.01)
        self.assertGreaterEqual(pose.focus_score, 95.0)

    def test_low_focus_inactivity_and_silence_reduce_engagement(self) -> None:
        """Ignoring poor visual and audio indicators would incorrectly inflate the score."""
        scorer = EngagementScorer(movement_window=3)
        score = scorer.score(
            focus_scores=[10.0, 15.0],
            movement_values=[0.0, 0.0, 0.0],
            audio_level=0.0,
        )
        self.assertLess(score.score, 25.0)
        self.assertTrue(score.sustained_inactivity)

    def test_stillness_alone_does_not_penalize_focused_participants(self) -> None:
        scorer = EngagementScorer(movement_window=3)
        score = scorer.score(
            focus_scores=[80.0],
            movement_values=[0.0, 0.0, 0.0],
            audio_level=0.25,
        )

        self.assertEqual(score.movement_score, 0.0)
        self.assertGreater(score.score, 50.0)
        self.assertFalse(score.sustained_inactivity)

    def test_audio_samples_are_reduced_to_a_bounded_activity_measure(self) -> None:
        """Dropping RMS normalization would make optional audio metrics unusable."""
        sample = analyze_audio_samples(np.array([0.5, -0.5, 0.5, -0.5]), sample_rate=16_000)
        self.assertAlmostEqual(sample.rms, 0.5, delta=0.001)
        self.assertAlmostEqual(sample.decibels_full_scale, -6.02, delta=0.05)
        self.assertGreater(sample.normalized_level, 0.85)

    def test_audio_activity_heuristic_separates_quiet_speech_like_and_noise(self) -> None:
        """Treating every loud signal as speech would make noise misleading."""
        sample_rate = 16_000
        time_axis = np.arange(sample_rate) / sample_rate
        speech_like = 0.30 * np.sin(2.0 * np.pi * 200.0 * time_axis)
        noise_like = np.tile(np.array([0.20, -0.20]), sample_rate // 2)

        self.assertEqual(classify_audio_activity(np.zeros(sample_rate), sample_rate).label, "quiet")
        self.assertEqual(classify_audio_activity(speech_like, sample_rate).label, "speech_likely")
        self.assertEqual(classify_audio_activity(noise_like, sample_rate).label, "ambient_noise")

    def test_session_analytics_summarizes_aggregate_attendance_and_engagement(self) -> None:
        """Ignoring absent profiles would overstate attendance in the session report."""
        attendance = pd.DataFrame(
            [
                {"profile_id": "PROFILE-001", "status": "Present", "cumulative_frames": 20},
                {"profile_id": "PROFILE-002", "status": "Absent", "cumulative_frames": 2},
            ]
        )
        metrics = pd.DataFrame(
            [{"engagement_score": 40.0}, {"engagement_score": 80.0}]
        )

        summary = generate_session_analytics(attendance, metrics)
        self.assertEqual(summary["present_profiles"], 1)
        self.assertEqual(summary["tracked_profiles"], 2)
        self.assertAlmostEqual(float(summary["track_confirmation_rate"]), 50.0)
        self.assertIsNone(summary["attendance_rate"])
        self.assertAlmostEqual(float(summary["mean_engagement"]), 60.0)

        roster_summary = generate_session_analytics(
            attendance, metrics, expected_participants=25
        )
        self.assertAlmostEqual(float(roster_summary["attendance_rate"]), 4.0)

    def test_expected_participants_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            generate_session_analytics(pd.DataFrame(), pd.DataFrame(), expected_participants=0)

    def test_roster_attendance_uses_expected_class_size_not_transient_ids(self) -> None:
        attendance = pd.DataFrame(
            [{"status": "Present"}] + [{"status": "Absent"} for _ in range(11)]
        )

        summary = generate_session_analytics(
            attendance, pd.DataFrame(), expected_participants=25
        )

        self.assertEqual(summary["present_profiles"], 1)
        self.assertEqual(summary["tracked_profiles"], 12)
        self.assertEqual(summary["track_confirmation_rate"], 8.33)
        self.assertEqual(summary["attendance_rate"], 4.0)


class LocalAgentTests(unittest.TestCase):
    def test_local_agent_returns_three_validated_interventions(self) -> None:
        """Returning arbitrary LLM text must not be accepted as actionable advice."""
        response_body = {
            "response": json.dumps(
                {
                    "interventions": [
                        {"title": "Think-pair-share", "action": "Ask a 60-second paired question.", "rationale": "Re-engages quiet learners."},
                        {"title": "Quick poll", "action": "Use a show-of-hands check.", "rationale": "Restores participation."},
                        {"title": "Micro-break", "action": "Invite a 30-second stretch.", "rationale": "Resets attention."},
                    ]
                }
            )
        }

        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return response_body

        agent = LocalTeachingAgent(http_post=lambda *args, **kwargs: Response())
        suggestions = agent.request_suggestions({"engagement_score": 42.0})
        self.assertEqual(len(suggestions), 3)
        self.assertEqual(suggestions[0].title, "Think-pair-share")


class SessionStoreTests(unittest.TestCase):
    def test_sessions_round_trip_and_clear_from_local_database(self) -> None:
        metrics = pd.DataFrame(
            [{"timestamp": "2026-10-24T09:00:00+00:00", "engagement_score": 72.5}]
        )
        attendance = pd.DataFrame(
            [{"profile_id": "PROFILE-001", "status": "Present", "cumulative_frames": 20}]
        )

        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "history.sqlite3"
            session_id = save_session(
                metrics,
                attendance,
                "Uploaded audio",
                started_at="2026-10-24T09:00:00+00:00",
                expected_participants=25,
                database_path=database_path,
            )

            summaries = list_sessions(database_path)
            loaded = load_session(session_id, database_path)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["audio_source"], "Uploaded audio")
            self.assertEqual(summaries[0]["expected_participants"], 25)
            self.assertAlmostEqual(summaries[0]["roster_attendance_rate"], 4.0)
            self.assertAlmostEqual(float(loaded["metrics"].iloc[0]["engagement_score"]), 72.5)
            self.assertEqual(loaded["expected_participants"], 25)
            self.assertEqual(loaded["attendance"].iloc[0]["status"], "Present")
            self.assertTrue(update_session_roster(session_id, 20, database_path))
            self.assertAlmostEqual(list_sessions(database_path)[0]["roster_attendance_rate"], 5.0)
            self.assertEqual(clear_history(database_path), 1)
            self.assertEqual(list_sessions(database_path), [])


class ReportTests(unittest.TestCase):
    def test_session_pdf_is_generated_in_memory(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "timestamp": "2026-10-24T09:00:00+00:00",
                    "engagement_score": 70.0,
                    "focus_score": 80.0,
                    "movement_score": 50.0,
                    "audio_activity": "speech_likely",
                }
            ]
        )
        attendance = pd.DataFrame(
            [{"profile_id": "PROFILE-001", "status": "Present", "cumulative_frames": 20}]
        )
        summary = {
            "mean_engagement": 70.0,
            "minimum_engagement": 70.0,
            "present_profiles": 1,
            "tracked_profiles": 1,
            "track_confirmation_rate": 100.0,
            "expected_participants": None,
            "attendance_rate": None,
        }

        document = build_session_pdf(
            metrics, attendance, summary, "2026-10-24 09:00 UTC", "Uploaded audio"
        )
        self.assertTrue(document.startswith(b"%PDF-"))


class OpenCVRuntimeTests(unittest.TestCase):
    def test_opencv_runtime_exposes_haar_cascade_api(self) -> None:
        """A stripped OpenCV binding would make offline face detection unavailable."""
        self.assertTrue(hasattr(cv2, "CascadeClassifier"))
        classifier = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        self.assertFalse(classifier.empty())


if __name__ == "__main__":
    unittest.main()
