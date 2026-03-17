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
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
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
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#1f2937"),
        wordWrap="CJK",
        spaceAfter=6,
    )
    return {
        "title": ParagraphStyle(
            "Title",
            parent=body,
            fontSize=20,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=body,
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=body,
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#1d4ed8"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=body,
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#111827"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": body,
        "bullet": ParagraphStyle(
            "Bullet",
            parent=body,
            leftIndent=10,
            firstLineIndent=-8,
        ),
        "footnote": ParagraphStyle(
            "Footnote",
            parent=body,
            fontSize=9,
            leading=13,
            leftIndent=10,
            firstLineIndent=-10,
            textColor=colors.HexColor("#374151"),
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=body,
            fontSize=9.5,
            leading=13,
        ),
    }


def _extract_footnote_numbers(lines: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    next_number = 1

    for line in lines:
        stripped = line.strip()
        if re.match(r"^\[\^[^\]]+\]:", stripped):
            continue
        for match in re.finditer(r"\[\^([^\]]+)\]", line):
            footnote_id = match.group(1)
            if footnote_id not in mapping:
                mapping[footnote_id] = next_number
                next_number += 1

    for line in lines:
        match = re.match(r"^\[\^([^\]]+)\]:", line.strip())
        if not match:
            continue
        footnote_id = match.group(1)
        if footnote_id not in mapping:
            mapping[footnote_id] = next_number
            next_number += 1

    return mapping


def _format_inline(text: str, footnote_numbers: dict[str, int]) -> str:
    formatted = escape(text)
    formatted = re.sub(r"\*\*(.+?)\*\*", r"\1", formatted)
    formatted = re.sub(r"`(.+?)`", r"\1", formatted)
    formatted = re.sub(
        r"\[\^([^\]]+)\]",
        lambda match: f"<super>{footnote_numbers.get(match.group(1), '?')}</super>",
        formatted,
    )
    return formatted


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _is_table_divider(line: str) -> bool:
    stripped = line.strip().strip("|").replace(":", "").replace("-", "").replace(" ", "")
    return stripped == ""


def _parse_table_rows(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        if _is_table_divider(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def _table_flowable(rows: list[list[str]], styles: dict[str, ParagraphStyle], footnote_numbers: dict[str, int]) -> Table:
    table_rows = [
        [
            Paragraph(_format_inline(cell, footnote_numbers), styles["table"])
            for cell in row
        ]
        for row in rows
    ]
    table = Table(table_rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("AppleGothic", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def _build_story(markdown_text: str, styles: dict[str, ParagraphStyle]) -> list:
    lines = markdown_text.splitlines()
    footnote_numbers = _extract_footnote_numbers(lines)
    story: list = []

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        footnote_match = re.match(r"^\[\^([^\]]+)\]:\s*(.+)$", stripped)
        if footnote_match:
            footnote_id, content = footnote_match.groups()
            number = footnote_numbers.get(footnote_id, 0)
            story.append(
                Paragraph(
                    f"{number}. {_format_inline(content, footnote_numbers)}",
                    styles["footnote"],
                )
            )
            index += 1
            continue

        if _is_table_line(stripped):
            table_lines: list[str] = []
            while index < len(lines) and _is_table_line(lines[index].strip()):
                table_lines.append(lines[index].strip())
                index += 1
            rows = _parse_table_rows(table_lines)
            if rows:
                story.append(_table_flowable(rows, styles, footnote_numbers))
                story.append(Spacer(1, 6))
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = _format_inline(heading_match.group(2), footnote_numbers)
            style_key = {1: "title", 2: "h1", 3: "h2", 4: "h3"}.get(level, "body")
            story.append(Paragraph(title, styles[style_key]))
            index += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            bullet_text = re.sub(r"^[-*]\s+", "", stripped)
            story.append(
                Paragraph(
                    f"- {_format_inline(bullet_text, footnote_numbers)}",
                    styles["bullet"],
                )
            )
            index += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            story.append(
                Paragraph(
                    _format_inline(stripped, footnote_numbers),
                    styles["bullet"],
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
                or re.match(r"^[-*]\s+", candidate)
                or re.match(r"^\d+\.\s+", candidate)
                or _is_table_line(candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1

        story.append(
            Paragraph(
                _format_inline(" ".join(paragraph_lines), footnote_numbers),
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
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
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
