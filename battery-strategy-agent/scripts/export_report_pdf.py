"""
마크다운 보고서를 스타일이 적용된 PDF로 내보냅니다.

사용 예:
    python scripts/export_report_pdf.py path/to/report.md
    python scripts/export_report_pdf.py path/to/report.md --output path/to/report.pdf
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="마크다운 보고서를 PDF로 변환합니다.")
    parser.add_argument("input", help="입력 마크다운 파일 경로입니다.")
    parser.add_argument(
        "--output",
        help="출력 PDF 경로입니다. 생략하면 입력 파일명에 .pdf를 붙입니다.",
    )
    return parser.parse_args()


def _register_font() -> str:
    for font_path in FONT_CANDIDATES:
        if not font_path.exists():
            continue
        font_name = "AppleGothic"
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
        return font_name
    raise FileNotFoundError("이 시스템에서 사용할 수 있는 한글 글꼴을 찾지 못했습니다.")


def _build_styles(font_name: str) -> dict[str, ParagraphStyle]:
    base_styles = getSampleStyleSheet()

    body = ParagraphStyle(
        "Body",
        parent=base_styles["BodyText"],
        fontName=font_name,
        fontSize=10.2,
        leading=16,
        textColor=colors.HexColor("#1f2937"),
        wordWrap="CJK",
        spaceAfter=7,
    )
    return {
        "kicker": ParagraphStyle(
            "Kicker",
            parent=body,
            fontSize=8.5,
            leading=10,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#475569"),
            spaceAfter=4,
        ),
        "title": ParagraphStyle(
            "Title",
            parent=body,
            fontSize=22,
            leading=30,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=body,
            fontSize=15.5,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=16,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=body,
            fontSize=12.6,
            leading=18,
            textColor=colors.HexColor("#1e3a8a"),
            spaceBefore=10,
            spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=body,
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#111827"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": body,
        "bullet": ParagraphStyle(
            "Bullet",
            parent=body,
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=4,
        ),
        "numbered": ParagraphStyle(
            "Numbered",
            parent=body,
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=4,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=body,
            fontSize=9.6,
            leading=14,
            textColor=colors.HexColor("#334155"),
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=body,
            fontSize=9.3,
            leading=13.5,
        ),
    }


def _format_inline(text: str) -> str:
    formatted = escape(text)
    formatted = formatted.replace("&lt;br/&gt;", "<br/>").replace("&lt;br&gt;", "<br/>")
    formatted = re.sub(r"\*\*(.+?)\*\*", r"\1", formatted)
    formatted = re.sub(r"\*(.+?)\*", r"\1", formatted)
    formatted = re.sub(r"`(.+?)`", r"\1", formatted)
    formatted = re.sub(r"\[\^[^\]]+\]", "", formatted)
    formatted = re.sub(r"\s*\[출처:\s*[^\]]+\]", "", formatted)
    return formatted


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _is_table_divider(line: str) -> bool:
    stripped = (
        line.strip()
        .strip("|")
        .replace("|", "")
        .replace(":", "")
        .replace("-", "")
        .replace(" ", "")
    )
    return stripped == ""


def _parse_table_rows(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        if _is_table_divider(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def _title_block(title: str, styles: dict[str, ParagraphStyle]) -> list:
    return [
        Paragraph("BATTERY STRATEGY REPORT", styles["kicker"]),
        Paragraph(title, styles["title"]),
        HRFlowable(
            width=42 * mm,
            thickness=1.8,
            color=colors.HexColor("#2563eb"),
            spaceAfter=10,
        ),
    ]


def _quote_flowable(text: str, styles: dict[str, ParagraphStyle]) -> Table:
    quote = Table(
        [[Paragraph(_format_inline(text), styles["quote"])]],
        colWidths=[170 * mm],
        hAlign="LEFT",
    )
    quote.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("LINEBEFORE", (0, 0), (0, -1), 2.2, colors.HexColor("#94a3b8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return quote


def _table_flowable(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    table_rows = [
        [
            Paragraph(_format_inline(cell), styles["table"])
            for cell in row
        ]
        for row in rows
    ]
    table = Table(table_rows, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#94a3b8")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("AppleGothic", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def _build_story(markdown_text: str, styles: dict[str, ParagraphStyle]) -> list:
    lines = markdown_text.splitlines()
    story: list = []
    title_rendered = False

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        footnote_match = re.match(r"^\[\^([^\]]+)\]:\s*(.+)$", stripped)
        if footnote_match:
            index += 1
            continue

        if _is_table_line(stripped):
            table_lines: list[str] = []
            while index < len(lines) and _is_table_line(lines[index].strip()):
                table_lines.append(lines[index].strip())
                index += 1
            rows = _parse_table_rows(table_lines)
            if rows:
                story.append(_table_flowable(rows, styles))
                story.append(Spacer(1, 8))
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate.startswith(">"):
                    break
                quote_lines.append(candidate.lstrip(">").strip())
                index += 1
            if quote_lines:
                story.append(_quote_flowable(" ".join(quote_lines), styles))
                story.append(Spacer(1, 8))
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = _format_inline(heading_match.group(2))
            if not title_rendered and level == 1:
                story.extend(_title_block(title, styles))
                title_rendered = True
            else:
                style_key = {1: "h1", 2: "h1", 3: "h2", 4: "h3"}.get(level, "body")
                story.append(Paragraph(title, styles[style_key]))
                if style_key == "h1":
                    story.append(
                        HRFlowable(
                            width="100%",
                            thickness=0.7,
                            color=colors.HexColor("#cbd5e1"),
                            spaceAfter=8,
                        )
                    )
            index += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            bullet_text = re.sub(r"^[-*]\s+", "", stripped)
            story.append(
                Paragraph(
                    f"- {_format_inline(bullet_text)}",
                    styles["bullet"],
                )
            )
            index += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            story.append(
                Paragraph(
                    _format_inline(stripped),
                    styles["numbered"],
                )
            )
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (
                not candidate
                or re.match(r"^(#{1,4})\s+(.+)$", candidate)
                or re.match(r"^\[\^([^\]]+)\]:", candidate)
                or candidate.startswith(">")
                or re.match(r"^[-*]\s+", candidate)
                or re.match(r"^\d+\.\s+", candidate)
                or _is_table_line(candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1

        story.append(
            Paragraph(
                _format_inline(" ".join(paragraph_lines)),
                styles["body"],
            )
        )

    return story


def export_markdown_to_pdf(input_path: Path, output_path: Path) -> None:
    font_name = _register_font()
    styles = _build_styles(font_name)
    markdown_text = input_path.read_text(encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=16 * mm,
        title=input_path.stem,
    )
    story = _build_story(markdown_text, styles)
    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"마크다운 파일을 찾지 못했습니다: {input_path}")
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_suffix(".pdf")
    )

    export_markdown_to_pdf(input_path, output_path)
    print(f"PDF를 저장했습니다: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
