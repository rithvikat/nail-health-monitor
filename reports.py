import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from recommendations import get_full_recommendation, MEDICAL_DISCLAIMER_TEXT, MODEL_INFO


def generate_pdf_report(screening, user, previous_screening=None, output_path=None, top_2=None) -> str:
    """
    Generate a modern, professional PDF screening report strictly adhering
    to non-diagnostic healthcare reporting standards.

    Args:
        screening: Screening database object (id, predicted_label, confidence, image_path, created_at)
        user: User database object (name, email)
        previous_screening: Optional previous Screening database object
        output_path: Optional explicit file path for PDF
        top_2: Optional list of top 2 predictions [{class_id, label, confidence, percentage}]

    Returns:
        str: Relative/accessible path to the generated PDF file
    """
    os.makedirs("uploads/reports", exist_ok=True)

    if output_path is None:
        filename = f"report_screening_{screening.id}_{int(datetime.utcnow().timestamp())}.pdf"
        output_path = os.path.join("uploads", "reports", filename).replace("\\", "/")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f766e"),  # Teal
        alignment=1,  # Center
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#4b5563"),
        alignment=1,
        spaceAfter=14
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )

    warning_style = ParagraphStyle(
        "WarningText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#92400e")
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#7f1d1d")
    )

    # Retrieve full recommendation and interpretation bundle
    rec_bundle = get_full_recommendation(screening, previous_screening, top_2=top_2)
    top_2_list = rec_bundle["top_2"]
    conf_pct = f"{screening.confidence * 100:.1f}%"

    story = []

    # 1. Header (Title & Subtitle)
    story.append(Paragraph("Nail Health Monitor", title_style))
    story.append(Paragraph("Automated AI Screening & Monitoring Report", subtitle_style))
    story.append(Spacer(1, 6))

    # 2. Patient Information
    screening_date = (
        screening.created_at.strftime("%B %d, %Y - %I:%M %p")
        if hasattr(screening.created_at, "strftime")
        else str(screening.created_at)
    )

    patient_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_style),
            Paragraph(str(user.name), body_style),
            Paragraph("<b>Screening ID:</b>", body_style),
            Paragraph(f"#{screening.id}", body_style)
        ],
        [
            Paragraph("<b>Email:</b>", body_style),
            Paragraph(str(user.email), body_style),
            Paragraph("<b>Screening Date:</b>", body_style),
            Paragraph(screening_date, body_style)
        ]
    ]

    patient_table = Table(patient_data, colWidths=[100, 170, 100, 160])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 10))

    # 3. AI Screening Assessment Section
    story.append(Paragraph("AI Screening Assessment", section_heading))

    assessment_data = [
        [
            Paragraph("<b>AI Screening Finding:</b>", body_style),
            Paragraph(f"<b><font color='#0f766e' size='10'>{screening.predicted_label}</font></b>", body_style)
        ],
        [
            Paragraph("<b>AI Prediction Confidence:</b>", body_style),
            Paragraph(f"<b>{conf_pct}</b>", body_style)
        ]
    ]

    if top_2_list and len(top_2_list) > 1:
        assessment_data.append([
            Paragraph("<b>Top 2 AI Predictions:</b>", body_style),
            Paragraph(
                "<br/>".join([f"{i+1}. {p['label']} ({p['percentage']}%)" for i, p in enumerate(top_2_list[:2])]),
                body_style
            )
        ])

    assessment_table = Table(assessment_data, colWidths=[160, 370])
    assessment_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdfa")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#99f6e4")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccfbf1")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(assessment_table)
    story.append(Spacer(1, 8))

    # Dynamic Low-Confidence Warning (ONLY if confidence < 80%)
    if rec_bundle["show_low_confidence_warning"]:
        warning_data = [
            [
                Paragraph(
                    "<b>⚠ IMPORTANT:</b> The AI prediction has relatively low confidence. "
                    "Professional evaluation is recommended.",
                    warning_style
                )
            ]
        ]
        warning_table = Table(warning_data, colWidths=[530])
        warning_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#fcd34d")),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(warning_table)
        story.append(Spacer(1, 8))

    # AI Screening Interpretation Section (Requirement 5)
    interp_data = [
        [
            Paragraph("<b>AI Screening Interpretation:</b>", body_style),
            Paragraph(rec_bundle["interpretation"], body_style)
        ]
    ]
    interp_table = Table(interp_data, colWidths=[160, 370])
    interp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(interp_table)
    story.append(Spacer(1, 10))

    # 4. Captured Nail Image Section
    story.append(Paragraph("Captured Nail Image", section_heading))
    img_inserted = False
    if screening.image_path and os.path.exists(screening.image_path):
        try:
            nail_img = RLImage(screening.image_path, width=2.2 * inch, height=2.2 * inch)
            story.append(nail_img)
            img_inserted = True
        except Exception:
            pass

    if not img_inserted:
        story.append(Paragraph("<i>Nail image preview not available in report document.</i>", body_style))

    story.append(Spacer(1, 10))

    # 5. Screening Recommendations & Guidance Section (Requirement 6)
    story.append(Paragraph("Screening Recommendations & Guidance", section_heading))
    safe_rec = rec_bundle["recommendations"]

    guidance_data = [
        [
            Paragraph("<b>Priority / Status:</b>", body_style),
            Paragraph(f"<b>{safe_rec.get('priority_status', 'Routine Nail Care')}</b>", body_style)
        ],
        [
            Paragraph("<b>Recommended Action:</b>", body_style),
            Paragraph(safe_rec["recommended_action"], body_style)
        ],
        [
            Paragraph("<b>Clinical Guidance:</b>", body_style),
            Paragraph(safe_rec["clinical_guidance"], body_style)
        ],
        [
            Paragraph("<b>Daily Care & Hygiene:</b>", body_style),
            Paragraph(safe_rec["daily_care"], body_style)
        ],
        [
            Paragraph("<b>What to Avoid:</b>", body_style),
            Paragraph(safe_rec["what_to_avoid"], body_style)
        ],
        [
            Paragraph("<b>Suggested Monitoring:</b>", body_style),
            Paragraph(safe_rec.get("suggested_monitoring", "Continue periodic screening to monitor nail appearance over time."), body_style)
        ]
    ]

    guidance_table = Table(guidance_data, colWidths=[150, 380])
    guidance_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdfa")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#99f6e4")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccfbf1")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(guidance_table)
    story.append(Spacer(1, 10))

    # 6. Monitoring Timeline & Progression Section (Requirement 7)
    story.append(Paragraph("Monitoring Timeline & Progression", section_heading))
    mon = rec_bundle["monitoring"]

    monitoring_data = [
        [
            Paragraph("<b>Current Screening Finding:</b>", body_style),
            Paragraph(mon["current_finding"], body_style)
        ],
        [
            Paragraph("<b>Previous Screening Finding:</b>", body_style),
            Paragraph(mon["previous_finding"], body_style)
        ],
        [
            Paragraph("<b>Trend:</b>", body_style),
            Paragraph(f"<b>{mon['trend']}</b>", body_style)
        ],
        [
            Paragraph("<b>Change Detected:</b>", body_style),
            Paragraph(mon["change_detected"], body_style)
        ]
    ]

    monitoring_table = Table(monitoring_data, colWidths=[170, 360])
    monitoring_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(monitoring_table)
    story.append(Spacer(1, 12))

    # IMPORTANT MEDICAL DISCLAIMER
    disclaimer_box = [
        [
            Paragraph(
                f"<b>IMPORTANT MEDICAL DISCLAIMER:</b><br/>{MEDICAL_DISCLAIMER_TEXT}",
                disclaimer_style
            )
        ]
    ]
    disclaimer_table = Table(disclaimer_box, colWidths=[530])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#fca5a5")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(KeepTogether([disclaimer_table]))

    # Build PDF document
    doc.build(story)

    # Convert to web-safe relative path with forward slashes
    web_path = output_path.replace("\\", "/")
    return web_path
