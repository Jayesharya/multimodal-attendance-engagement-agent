"""PDF exports for aggregate session analytics."""

from __future__ import annotations

from io import BytesIO
from typing import Mapping

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_session_pdf(
    metrics: pd.DataFrame,
    attendance: pd.DataFrame,
    summary: Mapping[str, object],
    started_at: str,
    audio_source: str,
) -> bytes:
    """Build a downloadable report without writing session data to disk."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Classroom session analytics",
        author="Multimodel Attendance",
    )
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], alignment=TA_LEFT, leading=14
    )
    story = [
        Paragraph("Classroom Session Analytics", styles["Title"]),
        Paragraph(f"Started: {started_at}", body_style),
        Paragraph(f"Audio source: {audio_source}", body_style),
        Spacer(1, 0.18 * inch),
    ]

    kpi_rows = [
        ["Mean engagement", "Minimum engagement", "Confirmed IDs", "Roster attendance"],
        [
            f"{float(summary['mean_engagement']):.1f}/100",
            f"{float(summary['minimum_engagement']):.1f}/100",
            f"{summary['present_profiles']}/{summary['tracked_profiles']}",
            (
                f"{summary['present_profiles']}/{summary['expected_participants']} "
                f"({float(summary['attendance_rate']):.1f}%)"
                if summary.get("attendance_rate") is not None
                else "Not configured"
            ),
        ],
    ]
    story.append(
        Table(
            kpi_rows,
            colWidths=[1.8 * inch, 1.8 * inch, 1.6 * inch, 1.6 * inch],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14233b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#edf3fa")),
                    ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#10203a")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c6d2e1")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        )
    )
    story.extend([Spacer(1, 0.2 * inch), Paragraph("Engagement samples", styles["Heading2"])])

    metric_columns = [
        column
        for column in ("timestamp", "engagement_score", "focus_score", "movement_score", "audio_activity")
        if column in metrics.columns
    ]
    sample_rows = metrics[metric_columns]
    if len(sample_rows) > 24:
        sample_rows = sample_rows.iloc[:: max(1, len(sample_rows) // 24)].head(24)
    timeline_rows = [metric_columns]
    timeline_rows.extend(
        [str(value)[:22] for value in row]
        for row in sample_rows.itertuples(index=False, name=None)
    )
    if len(timeline_rows) == 1:
        timeline_rows.append(["No session samples"] + [""] * (len(metric_columns) - 1))
    timeline = Table(timeline_rows, repeatRows=1, hAlign="LEFT")
    timeline.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14233b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c6d2e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
            ]
        )
    )
    story.extend([timeline, Spacer(1, 0.2 * inch), Paragraph("Attendance", styles["Heading2"])])

    attendance_columns = [
        column
        for column in ("profile_id", "status", "cumulative_frames", "first_seen_at", "last_seen_at")
        if column in attendance.columns
    ]
    attendance_rows = [attendance_columns]
    attendance_rows.extend(
        [str(value)[:24] for value in row]
        for row in attendance[attendance_columns].itertuples(index=False, name=None)
    )
    if len(attendance_rows) == 1:
        attendance_rows.append(["No tracked profiles"] + [""] * (len(attendance_columns) - 1))
    attendance_table = Table(attendance_rows, repeatRows=1, hAlign="LEFT")
    attendance_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14233b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c6d2e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
            ]
        )
    )
    story.append(attendance_table)
    document.build(story)
    return output.getvalue()