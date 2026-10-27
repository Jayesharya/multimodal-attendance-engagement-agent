# Multimodel Attendance

Offline attendance and engagement analysis for a webcam or local video file.

## Requirements

- Python 3.11+
- Webcam or local video file
- Optional microphone or audio file
- Optional local Ollama instance for teaching suggestions

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade --force-reinstall -r requirements.txt
```

## Run

Terminal runner:

```powershell
python app.py --source 0
```

Dashboard:

```powershell
python -m streamlit run dashboard.py
```

The dashboard accepts webcam/video input, optional audio files processed locally with Librosa, and an optional local MediaPipe Face Landmarker `.task` model. Without a MediaPipe model, it uses OpenCV Haar cascades.

## Session reports

- Uploaded audio is analyzed in one-second windows and aligned from the start of the webcam run or video timeline. It is not extracted from the video file.
- Completed sessions can be saved locally using the sidebar option. Saved history is limited to the 50 most recent sessions.
- The SQLite database is stored outside the project folder: `%LOCALAPPDATA%\MultimodelAttendance\session_history.sqlite3` on Windows, or `$XDG_DATA_HOME/MultimodelAttendance/session_history.sqlite3` on Linux/macOS (defaulting to `~/.local/share`).
- Saved reports contain aggregate metrics and temporary profile labels only. Video frames and raw audio are not stored. Use **Clear saved history** on the Analytics page to delete saved sessions.
- Analytics can export CSV metrics and a PDF session report.
- Set **Expected class size** in the sidebar to calculate roster attendance. Without a roster count, the dashboard reports how many anonymous detected IDs met the presence threshold; it does not label that detection-confirmation ratio as class attendance.

Engagement is a heuristic built from approximate visual focus, movement, and audio activity. The application does not identify people, classify emotions, transcribe speech, or measure teaching quality. Profile IDs are anonymous, session-local tracking labels.

## Validation

Run the behavioral test suite with:

```powershell
python -m unittest discover -s tests -v
```

The tests cover core analytics, audio-window alignment, local session storage, PDF generation, and the local Ollama response contract. They do not substitute for validation in real classroom conditions.

## Included modules

- Vision: face detection, centroid tracking, and sustained-presence attendance
- Audio: RMS/dBFS extraction and quiet, speech-like, or ambient-noise activity labels
- Fusion: rolling engagement score from focus, movement, and audio activity
- Recommendations: optional loopback-only Ollama teaching suggestions
- Analytics: session-level attendance and engagement summaries
