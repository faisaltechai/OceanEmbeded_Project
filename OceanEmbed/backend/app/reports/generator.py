"""
PDF technical report generator.

Uses reportlab (a real, mature PDF library) to assemble a report purely
from data already returned by other endpoints in this run -- it takes
already-fetched dicts as arguments rather than re-deriving numbers, so the
PDF can never drift from what the UI showed the user. No chart image
generation library is bundled here by default (matplotlib is a heavy,
optional add); the profile/map numbers are rendered as tables, which is
both honest about what this sandbox could verify runs offline and perfectly
normal for a technical report. Swap in matplotlib-rendered PNGs under
`_build_charts` if you want figures -- the seam is marked below.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

REPORTS_DIR = os.getenv("OCEANEMBED_REPORTS_DIR", os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "reports"))

DEEP_OCEAN = colors.HexColor("#021B2D")
SKY_BLUE = colors.HexColor("#4FC3F7")
SAND = colors.HexColor("#E8D8A8")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("OEHeading", parent=ss["Heading1"], textColor=DEEP_OCEAN, spaceAfter=6))
    ss.add(ParagraphStyle("OESub", parent=ss["Heading2"], textColor=SKY_BLUE, spaceBefore=12, spaceAfter=4))
    ss.add(ParagraphStyle("OEBody", parent=ss["BodyText"], spaceAfter=4))
    return ss


def _source_table(rows: list[tuple[str, str, str, str]], header: list[str] | None = None,
                   col_widths: list[float] | None = None) -> Table:
    header = header or ["Field", "Value", "Source", "Status"]
    styles = _styles()
    wrap_style = ParagraphStyle("cell", parent=styles["OEBody"], fontSize=8, leading=10, spaceAfter=0)
    header_cells = [Paragraph(f"<b>{h}</b>", ParagraphStyle("hcell", parent=wrap_style, textColor=colors.white)) for h in header]
    data = [header_cells] + [[Paragraph(str(c), wrap_style) for c in row] for row in rows]
    widths = col_widths or [40 * mm, 35 * mm, 55 * mm, 30 * mm]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DEEP_OCEAN),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F8FC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def build_report(
    region: str,
    region_name: str,
    lat: float | None,
    lon: float | None,
    overview: dict,
    profile: dict | None,
    events: list[dict],
) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"oceanembed_report_{region}_{ts}.pdf"
    path = os.path.join(REPORTS_DIR, filename)

    styles = _styles()
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    story = []

    story.append(Paragraph("OceanEmbed — Ocean Intelligence Report", styles["OEHeading"]))
    story.append(Paragraph(
        f"Region: {region_name} &nbsp;|&nbsp; Generated: {datetime.now(timezone.utc).isoformat()} UTC",
        styles["OEBody"],
    ))
    if lat is not None and lon is not None:
        story.append(Paragraph(f"Point of interest: {lat:.3f}, {lon:.3f}", styles["OEBody"]))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Overview", styles["OESub"]))
    rows = []
    for key, card in overview.get("cards", {}).items():
        rows.append((key.replace("_", " ").title(), str(card.get("value")), str(card.get("source", "")), str(card.get("status", ""))))
    story.append(_source_table(rows))

    if profile:
        story.append(Paragraph("Digital Ocean Profile", styles["OESub"]))
        story.append(Paragraph(
            f"Matched grid cell: {profile.get('matched_grid_cell')} · Date: {profile.get('date')} · "
            f"Confidence: {profile.get('confidence')}",
            styles["OEBody"],
        ))
        depths = profile.get("depths_m", [])
        temps = profile.get("predicted_temperature_c", [])
        uncert = profile.get("uncertainty_c", [])
        prof_rows = [(str(d), f"{t:.2f}", f"±{u:.2f}") for d, t, u in zip(depths, temps, uncert)]
        story.append(_source_table(prof_rows, header=["Depth (m)", "Temp (°C)", "Uncertainty"],
                                    col_widths=[50 * mm, 50 * mm, 50 * mm]))

    if events:
        story.append(Paragraph("Detected Events", styles["OESub"]))
        ev_rows = [(e.get("type", ""), e.get("severity", ""), e.get("date", ""), e.get("description", "")) for e in events]
        story.append(_source_table(ev_rows, header=["Type", "Severity", "Date", "Description"]))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        "Sources & methodology: values labeled status=\"live\" were fetched from the named "
        "provider during report generation. Values labeled status=\"demo\" come from the "
        "documented synthetic prototype pipeline (see docs/requirements.md) and are never "
        "presented as live satellite observations. This report is a Smart India Hackathon "
        "SIH26066 prototype technical document.",
        styles["OEBody"],
    ))

    doc.build(story)
    return path
