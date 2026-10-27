"""Local Streamlit dashboard for attendance and engagement analysis."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import Deque, Dict, Optional, Sequence, Tuple, Union, cast

import cv2
import pandas as pd

import config
from analytics_engine import (
    AudioSample,
    AudioWindow,
    CascadeFaceDetector,
    EngagementScore,
    EngagementScorer,
    FaceTracker,
    MediaPipeFaceLandmarker,
    MetricsLogger,
    TrackingUpdate,
    analyze_audio_file_windows,
    audio_sample_at,
    generate_session_analytics,
    simulate_audio_amplitude,
)
from app import calculate_frame_engagement, draw_overlay, metrics_payload, should_trigger_agent
from llm_agent import LocalLLMError, LocalTeachingAgent
from reporting import build_session_pdf
from session_store import (
    clear_history,
    list_sessions,
    load_session,
    save_session,
    update_session_roster,
)


def persist_upload(upload, suffix: str) -> Path:
    """Store an uploaded asset only for the duration of one dashboard action."""
    with NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(upload.getbuffer())
        return Path(temporary_file.name)


def build_source(mode: str, video_upload) -> Tuple[Union[int, str], Optional[Path]]:
    """Translate dashboard input into an OpenCV source and temporary file path."""
    if mode == "Webcam":
        return 0, None
    if video_upload is None:
        raise ValueError("Choose a video file before running analysis.")
    temporary_video = persist_upload(video_upload, Path(video_upload.name).suffix or ".mp4")
    return str(temporary_video), temporary_video


def run_analysis(
    source: Union[int, str],
    frame_limit: int,
    audio_windows: Optional[Sequence[AudioWindow]],
    media_pipe_model: Optional[str],
    preview,
    status,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], Optional[object]]:
    """Analyze a bounded webcam or video stream fully on the local machine."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the selected video source.")

    detector = CascadeFaceDetector()
    tracker = FaceTracker()
    scorer = EngagementScorer()
    logger = MetricsLogger(interval_seconds=0.0)
    movement_history: Dict[str, Deque[float]] = defaultdict(
        lambda: deque(maxlen=config.MOVEMENT_WINDOW_FRAMES)
    )
    landmarker = MediaPipeFaceLandmarker(media_pipe_model) if media_pipe_model else None
    latest_payload: Dict[str, object] = {}
    latest_frame = None
    session_clock_start = monotonic()
    previous_video_elapsed = -1.0
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    try:
        for frame_index in range(frame_limit):
            captured, frame = capture.read()
            if not captured:
                break
            update = tracker.update(detector.detect(frame))
            if audio_windows is not None:
                if isinstance(source, str):
                    video_elapsed = capture.get(cv2.CAP_PROP_POS_MSEC) / 1_000.0
                    if frame_index and video_elapsed <= previous_video_elapsed:
                        video_elapsed = frame_index / video_fps
                else:
                    video_elapsed = monotonic() - session_clock_start
                previous_video_elapsed = video_elapsed
                active_audio = audio_sample_at(audio_windows, video_elapsed)
            else:
                active_audio = AudioSample(
                    rms=0.0,
                    decibels_full_scale=config.AUDIO_SILENCE_DBFS,
                    normalized_level=simulate_audio_amplitude(frame_index),
                    captured_at=datetime.now(timezone.utc).isoformat(),
                )
            engagement = calculate_frame_engagement(
                update,
                scorer,
                movement_history,
                active_audio.normalized_level,
            )
            if landmarker is not None:
                poses = landmarker.detect_poses(frame)
                if poses:
                    movements = [
                        movement
                        for values in movement_history.values()
                        for movement in values
                    ]
                    engagement = scorer.score(
                        [pose.focus_score for pose in poses],
                        movements,
                        active_audio.normalized_level,
                    )
            logger.append_if_due(update.tracks, engagement, active_audio)
            latest_payload = metrics_payload(update, engagement, active_audio)

            if frame_index % config.DASHBOARD_FRAME_UPDATE_INTERVAL == 0:
                draw_overlay(frame, update, engagement)
                preview.image(
                    frame,
                    channels="BGR",
                    width="stretch",
                    alt="Most recently analyzed frame with anonymous tracking overlays",
                )
                latest_frame = frame.copy()
                status.caption(
                    f"Frame {frame_index + 1} · engagement {engagement.score:.1f}/100 · "
                    f"active profiles {sum(track.active for track in update.tracks)}"
                )
    finally:
        capture.release()
        if landmarker is not None:
            landmarker.close()

    return logger.dataframe(), tracker.attendance_dataframe(), latest_payload, latest_frame


def main() -> None:
    """Render the local classroom dashboard and its session-scoped reports."""
    import streamlit as st

    st.set_page_config(page_title="EduSense AI", page_icon=":material/school:", layout="wide")

    defaults = {
        "workspace_view": "Live monitor",
        "session_metrics": pd.DataFrame(),
        "session_attendance": pd.DataFrame(),
        "session_payload": {},
        "session_frame": None,
        "session_audio_source": "No uploaded audio",
        "session_suggestions": [],
        "session_id": None,
        "session_started_at": None,
        "session_expected_participants": None,
        "confirm_clear_history": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    with st.sidebar:
        st.markdown("## EduSense AI")
        st.caption("Classroom intelligence")
        st.radio(
            "Workspace",
            ["Live monitor", "Analytics & reports", "Teacher profile"],
            key="workspace_view",
        )
        st.caption("Local processing. Video and raw audio are not saved.")

    view = st.session_state.workspace_view
    page_details = {
        "Live monitor": ("Classroom monitor", ":material/monitoring:", "Local video analysis with anonymous session-only profiles."),
        "Analytics & reports": ("Analytics & reports", ":material/analytics:", "Review engagement trends and roster-based attendance."),
        "Teacher profile": ("Session profile", ":material/person_check:", "Reflect on observed signals; this is not an evaluation of teaching quality."),
    }
    title, title_icon, page_caption = page_details[view]
    st.header(title, icon=title_icon)
    st.caption(page_caption)

    source_mode = st.session_state.get("source_mode", "Webcam")
    video_upload = None
    audio_upload = None
    model_upload = None
    frame_limit = config.DASHBOARD_DEFAULT_MAX_FRAMES
    expected_participants = st.session_state.session_expected_participants
    ask_agent = False
    save_history = True
    run_clicked = False

    if view == "Live monitor":
        st.markdown("#### Session setup")
        source_mode = st.segmented_control(
            "Video source",
            ["Webcam", "Video file"],
            default="Webcam",
            key="source_mode",
            width="stretch",
            required=True,
        )
        with st.form("session_setup", border=True):
            if source_mode == "Video file":
                video_upload = st.file_uploader(
                    "Classroom video", type=["mp4", "avi", "mov", "mkv"]
                )
            else:
                st.caption("Webcam selected. Grant camera access when you start analysis.")

            with st.expander("Optional inputs and analysis settings", icon=":material/tune:"):
                audio_col, model_col = st.columns(2)
                audio_upload = audio_col.file_uploader(
                    "Optional audio", type=["wav", "mp3", "flac", "ogg"]
                )
                model_upload = model_col.file_uploader(
                    "Optional face landmark model", type=["task"]
                )

                frame_col, roster_col = st.columns(2)
                frame_limit = frame_col.slider(
                    "Frames to analyze",
                    min_value=30,
                    max_value=1_200,
                    value=config.DASHBOARD_DEFAULT_MAX_FRAMES,
                    step=30,
                )
                if "expected_participants" not in st.session_state:
                    st.session_state.expected_participants = expected_participants
                expected_participants = roster_col.number_input(
                    "Expected class size (optional)",
                    min_value=1,
                    max_value=300,
                    step=1,
                    key="expected_participants",
                )

                options_col, history_col = st.columns(2)
                ask_agent = options_col.checkbox("Request local teaching suggestions", value=False)
                save_history = history_col.checkbox("Save report to local history", value=True)
            run_clicked = st.form_submit_button(
                "Start local analysis",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
            )

    if run_clicked:
        temporary_paths = []
        try:
            session_started_at = datetime.now(timezone.utc).isoformat()
            source, temporary_video = build_source(source_mode, video_upload)
            if temporary_video is not None:
                temporary_paths.append(temporary_video)
            audio_windows = None
            if audio_upload is not None:
                temporary_audio = persist_upload(audio_upload, Path(audio_upload.name).suffix or ".wav")
                temporary_paths.append(temporary_audio)
                audio_windows = analyze_audio_file_windows(str(temporary_audio))
                if not audio_windows:
                    raise ValueError("The uploaded audio file contains no samples.")
            model_path = None
            if model_upload is not None:
                temporary_model = persist_upload(model_upload, ".task")
                temporary_paths.append(temporary_model)
                model_path = str(temporary_model)

            preview = st.skeleton(height=420)
            status = st.empty()
            with st.status("Analyzing the session locally", expanded=True) as analysis_status:
                metrics, attendance, payload, last_frame = run_analysis(
                    source=source,
                    frame_limit=frame_limit,
                    audio_windows=audio_windows,
                    media_pipe_model=model_path,
                    preview=preview,
                    status=status,
                )
                analysis_status.update(
                    label="Analysis complete",
                    state="complete",
                    expanded=False,
                )
            if metrics.empty:
                st.warning("No frames were read from the selected source.")
            else:
                st.session_state.session_metrics = metrics
                st.session_state.session_attendance = attendance
                st.session_state.session_payload = payload
                st.session_state.session_frame = last_frame
                audio_source = "Uploaded audio (1-second windows)" if audio_windows is not None else "Demo signal"
                st.session_state.session_audio_source = audio_source
                st.session_state.session_suggestions = []
                st.session_state.session_started_at = session_started_at
                st.session_state.session_id = None
                st.session_state.session_expected_participants = expected_participants
                if save_history:
                    try:
                        st.session_state.session_id = save_session(
                            metrics,
                            attendance,
                            audio_source,
                            started_at=session_started_at,
                            expected_participants=expected_participants,
                        )
                    except (OSError, sqlite3.Error) as error:
                        st.warning(f"Session completed, but local history could not be saved: {error}")
                if ask_agent:
                    current_score = EngagementScore(
                        score=float(payload["engagement_score"]),
                        focus_score=float(payload["focus_score"]),
                        movement_score=float(payload["movement_score"]),
                        audio_score=float(payload["audio_score"]),
                        sustained_inactivity=bool(payload["sustained_inactivity"]),
                    )
                    if should_trigger_agent(current_score, int(payload["active_profiles"])):
                        try:
                            st.session_state.session_suggestions = LocalTeachingAgent().request_suggestions(payload)
                        except LocalLLMError as error:
                            st.warning(f"Local Ollama is unavailable: {error}")
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            st.error(str(error))
        finally:
            for temporary_path in temporary_paths:
                temporary_path.unlink(missing_ok=True)

    metrics = st.session_state.session_metrics
    attendance = st.session_state.session_attendance
    has_session = not metrics.empty
    saved_sessions = list_sessions()

    if view == "Live monitor":
        st.caption("Local session · Anonymous tracking · No cloud upload")
        live_col, insight_col = st.columns([2.35, 1], gap="medium")
        with live_col:
            with st.container(border=True):
                live_header, live_status = st.columns([4, 1])
                live_header.markdown("#### Latest frame")
                live_status.badge(
                    "Run complete" if has_session else "Ready",
                    color="green" if has_session else "gray",
                )
                if st.session_state.session_frame is not None:
                    st.image(
                        st.session_state.session_frame,
                        channels="BGR",
                        width="stretch",
                        alt="Latest analyzed frame with anonymous face tracking overlays",
                    )
                else:
                    st.caption(":material/videocam_off: Choose a webcam or video file in Session setup to begin.")
            profile_col, suggestion_col = st.columns(2, gap="medium")
            with profile_col:
                with st.container(border=True):
                    st.markdown("#### Tracked profiles")
                    if has_session and not attendance.empty:
                        st.dataframe(
                            attendance[["profile_id", "status", "cumulative_frames", "active"]],
                            width="stretch",
                            hide_index=True,
                            alt="Anonymous profile tracks and their attendance confirmation status",
                        )
                    else:
                        st.caption("No profiles tracked yet.")
            with suggestion_col:
                with st.container(border=True):
                    st.markdown("#### AI suggestions")
                    if st.session_state.session_suggestions:
                        for suggestion in st.session_state.session_suggestions:
                            st.markdown(f"**{suggestion.title}**  \n{suggestion.action} {suggestion.rationale}")
                    else:
                        st.caption("No local suggestions were generated for this session.")
        with insight_col:
            if has_session:
                current = metrics.iloc[-1]
                summary = generate_session_analytics(
                    attendance,
                    metrics,
                    expected_participants=expected_participants,
                )
                st.metric("Profiles confirmed", int(current["present_profiles"]))
                st.metric("Engagement", f"{current['engagement_score']:.1f} / 100")
                roster_attendance = (
                    f"{summary['present_profiles']}/{expected_participants} "
                    f"({summary['attendance_rate']:.1f}%)"
                    if summary["attendance_rate"] is not None
                    else "Not set"
                )
                st.metric("Roster attendance", roster_attendance)
                st.caption(
                    f"{summary['present_profiles']} of {summary['tracked_profiles']} detected IDs "
                    "met the presence threshold."
                )
                if summary["attendance_rate"] is None:
                    st.caption("Set expected class size in Session setup for a roster percentage.")
                st.caption("Engagement is a heuristic signal, not an assessment of learning.")
                with st.container(border=True):
                    st.markdown("#### Audio activity")
                    st.caption(st.session_state.session_audio_source)
                    st.markdown(f"**{current['audio_activity']}**")
                    st.caption(f"{current['audio_dbfs']:.1f} dBFS")
            else:
                st.metric("Profiles confirmed", "--")
                st.metric("Engagement", "--")
                st.metric("Roster attendance", "--")
                st.caption("Set expected class size in Session setup before analysis.")
                with st.container(border=True):
                    st.markdown("#### Audio activity")
                    st.caption("Upload an audio file to estimate room activity.")

    elif view == "Analytics & reports":
        st.markdown("## Session analytics")
        st.caption("Roster attendance uses expected class size; detected IDs are reported separately.")
        session_options = {}
        if has_session:
            current_started_at = st.session_state.session_started_at or "Current session"
            session_options[f"Current run · {current_started_at}"] = None
        for record in saved_sessions:
            if has_session and record["session_id"] == st.session_state.session_id:
                continue
            label = (
                f"{record['started_at']} · {float(record['mean_engagement']):.1f} avg · "
                f"{str(record['session_id'])[:8]}"
            )
            session_options[label] = str(record["session_id"])

        if session_options:
            selected_label = st.selectbox("Session report", list(session_options))
            selected_id = session_options[selected_label]
            if selected_id is None:
                report_metrics = metrics
                report_attendance = attendance
                report_audio_source = st.session_state.session_audio_source
                report_started_at = st.session_state.session_started_at or "Current session"
                report_expected_participants = expected_participants
            else:
                saved_record = load_session(selected_id)
                if saved_record is None:
                    st.warning("That saved session is no longer available.")
                    st.stop()
                report_metrics = cast(pd.DataFrame, saved_record["metrics"])
                report_attendance = cast(pd.DataFrame, saved_record["attendance"])
                report_audio_source = str(saved_record["audio_source"])
                report_started_at = str(saved_record["started_at"])
                saved_expected_participants = saved_record["expected_participants"]
                report_expected_participants = (
                    int(saved_expected_participants)
                    if saved_expected_participants is not None
                    else None
                )

            roster_widget_key = f"report_roster_{selected_id or 'current'}"
            if roster_widget_key not in st.session_state:
                st.session_state[roster_widget_key] = report_expected_participants
            roster_col, save_roster_col = st.columns([3, 1])
            with roster_col:
                report_expected_participants = st.number_input(
                    "Expected class size",
                    min_value=1,
                    max_value=300,
                    step=1,
                    key=roster_widget_key,
                )
            report_session_id = selected_id or st.session_state.session_id
            with save_roster_col:
                save_roster = st.button(
                    "Save class size",
                    disabled=report_expected_participants is None or report_session_id is None,
                    width="stretch",
                )
            if save_roster and report_session_id is not None:
                if update_session_roster(report_session_id, report_expected_participants):
                    if selected_id is None:
                        st.session_state.session_expected_participants = report_expected_participants
                    st.success("Class size saved with this report.")
                    st.rerun()
                else:
                    st.warning("The saved session could not be updated.")

            summary = generate_session_analytics(
                report_attendance,
                report_metrics,
                expected_participants=report_expected_participants,
            )
            top_a, top_b, top_c = st.columns(3)
            top_a.metric("Mean engagement", f"{summary['mean_engagement']:.1f} / 100")
            top_b.metric(
                "Confirmed detected IDs",
                f"{summary['present_profiles']} / {summary['tracked_profiles']}",
            )
            top_c.metric(
                "Roster attendance",
                (
                    f"{summary['present_profiles']}/{report_expected_participants} "
                    f"({summary['attendance_rate']:.1f}%)"
                    if summary["attendance_rate"] is not None
                    else "Not set"
                ),
            )
            st.caption(
                f"Detected-ID confirmation: {summary['track_confirmation_rate']:.1f}%. "
                "Short-lived detections can create extra anonymous IDs."
            )
            if (
                report_expected_participants is not None
                and summary["present_profiles"] > report_expected_participants
            ):
                st.warning("More profiles were marked present than the expected class size; review possible duplicate tracks.")
            download_col, pdf_col, clear_col = st.columns([1.3, 1.4, 1])
            download_col.download_button(
                "Download CSV data",
                data=report_metrics.to_csv(index=False).encode("utf-8"),
                file_name="session_metrics.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )
            pdf_col.download_button(
                "Download PDF report",
                data=build_session_pdf(
                    report_metrics,
                    report_attendance,
                    summary,
                    report_started_at,
                    report_audio_source,
                ),
                file_name="session_report.pdf",
                mime="application/pdf",
                icon=":material/picture_as_pdf:",
                width="stretch",
            )
            if clear_col.button("Clear saved history", icon=":material/delete:"):
                st.session_state.confirm_clear_history = True
            if st.session_state.confirm_clear_history:
                st.warning("This deletes all saved local reports. The current unsaved run remains open.")
                confirm_col, cancel_col, _ = st.columns([1.5, 1, 4])
                if confirm_col.button("Delete saved reports", type="primary"):
                    removed_count = clear_history()
                    st.session_state.confirm_clear_history = False
                    st.success(f"Deleted {removed_count} saved session(s).")
                    st.rerun()
                if cancel_col.button("Cancel"):
                    st.session_state.confirm_clear_history = False
                    st.rerun()
            trend_col, audio_col = st.columns(2, gap="medium")
            with trend_col:
                with st.container(border=True):
                    st.markdown("#### Engagement trend")
                    st.line_chart(
                        report_metrics.set_index("timestamp")[["engagement_score", "focus_score", "movement_score"]],
                        height=280,
                        alt="Engagement and visual signal trends over the selected session",
                    )
            with audio_col:
                with st.container(border=True):
                    st.markdown("#### Audio activity over time")
                    activity_counts = report_metrics["audio_activity"].value_counts().rename_axis("activity").to_frame("samples")
                    st.bar_chart(
                        activity_counts,
                        height=280,
                        alt="Counts of audio activity classifications in the selected session",
                    )
            st.markdown("#### Attendance")
            st.dataframe(
                report_attendance,
                width="stretch",
                hide_index=True,
                alt="Attendance status and observed frames for anonymous tracked profiles",
            )
            st.caption(
                f"Minimum engagement {summary['minimum_engagement']:.1f}/100. Audio source: "
                f"{report_audio_source.lower()}."
            )
        else:
            st.info("Run an analysis session to see its report here and save it for later comparison.")

    else:
        st.markdown("## Session profile")
        st.caption("A summary of observed engagement signals, not a measure of teaching quality")
        profile_metrics = metrics
        profile_attendance = attendance
        profile_expected_participants = expected_participants
        if profile_metrics.empty and saved_sessions:
            latest_record = load_session(str(saved_sessions[0]["session_id"]))
            if latest_record is not None:
                profile_metrics = cast(pd.DataFrame, latest_record["metrics"])
                profile_attendance = cast(pd.DataFrame, latest_record["attendance"])
                saved_expected_participants = latest_record["expected_participants"]
                profile_expected_participants = (
                    int(saved_expected_participants)
                    if saved_expected_participants is not None
                    else None
                )
        if not profile_metrics.empty:
            summary = generate_session_analytics(
                profile_attendance,
                profile_metrics,
                expected_participants=profile_expected_participants,
            )
            focus_average = float(profile_metrics["focus_score"].mean())
            movement_average = float(profile_metrics["movement_score"].mean())
            audio_average = float(profile_metrics["audio_score"].mean())
            score_col, strengths_col, review_col = st.columns([1, 1, 1], gap="medium")
            with score_col:
                with st.container(border=True):
                    st.markdown("#### Session engagement")
                    st.metric("Average score", f"{summary['mean_engagement']:.1f} / 100")
                    if summary["attendance_rate"] is not None:
                        st.caption(f"Roster attendance {summary['attendance_rate']:.1f}%")
                    else:
                        st.caption(
                            f"No class size set. {summary['present_profiles']} of "
                            f"{summary['tracked_profiles']} detected IDs were confirmed."
                        )
            with strengths_col:
                with st.container(border=True):
                    st.markdown("#### Strongest signals")
                    components = {
                        "Focus": focus_average,
                        "Movement": movement_average,
                        "Audio activity": audio_average,
                    }
                    for name, value in sorted(components.items(), key=lambda item: item[1], reverse=True):
                        st.markdown(f"**{name}**  ·  {value:.1f}/100")
            with review_col:
                with st.container(border=True):
                    st.markdown("#### Suggestions")
                    if st.session_state.session_suggestions:
                        for suggestion in st.session_state.session_suggestions:
                            st.markdown(f"**{suggestion.title}**  \n{suggestion.action}")
                    else:
                        lowest_signal = min(components, key=components.get)
                        st.write(f"Review the {lowest_signal.lower()} trend alongside the class context.")
                        st.caption("Optional suggestions can be requested from a local Ollama model.")
            st.markdown("#### Recent session comparison")
            if saved_sessions:
                history_cols = st.columns(min(3, len(saved_sessions)))
                for column, entry in zip(history_cols, saved_sessions[:3]):
                    with column:
                        with st.container(border=True):
                            st.markdown(f"**{entry['started_at']}**")
                            st.markdown(f"{entry['mean_engagement']:.1f} / 100 engagement")
                            if entry["expected_participants"]:
                                rate = float(entry["roster_attendance_rate"] or 0.0)
                                st.caption(
                                    f"Roster attendance {entry['present_profiles']}/"
                                    f"{entry['expected_participants']} ({rate:.1f}%)"
                                )
                            else:
                                st.caption(
                                    f"{entry['present_profiles']} of {entry['tracked_profiles']} "
                                    "detected IDs confirmed; no roster size saved"
                                )
            else:
                st.caption("No saved sessions available for comparison.")
        else:
            st.info("Run a session to build a profile from observed engagement and attendance signals.")

if __name__ == "__main__":
    main()
