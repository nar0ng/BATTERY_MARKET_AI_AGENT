"""
참고문헌 형식 변환기 (규칙 기반)
보고서 초안 Agent에서 사용

REFERENCE 형식 (가이드 요구사항):
  - 기관 보고서: 발행기관(YYYY). 보고서명. URL
  - 학술 논문: 저자(YYYY). 논문제목. 학술지명, 권(호), 페이지.
  - 웹페이지: 기관명 또는 작성자(YYYY-MM-DD). 제목. 사이트명, URL
"""
from __future__ import annotations

import re
from urllib.parse import urlparse


def _safe_text(value: str | None, default: str) -> str:
    cleaned = (value or "").strip()
    return cleaned or default


def _infer_reference_type(source: dict) -> str:
    source_type = source.get("type")
    if source_type in {"report", "paper", "webpage"}:
        return source_type

    if source.get("journal"):
        return "paper"

    url = (source.get("url") or "").lower()
    if url.endswith(".pdf") or source.get("source_kind") == "document":
        return "report"

    return "webpage"


def _extract_year(source: dict) -> str:
    date_value = (source.get("date") or "").strip()
    match = re.search(r"(\d{4})", date_value)
    return match.group(1) if match else "n.d."


def _extract_date(source: dict) -> str:
    date_value = (source.get("date") or "").strip()
    match = re.search(r"\d{4}-\d{2}-\d{2}", date_value)
    if match:
        return match.group(0)

    year = _extract_year(source)
    return f"{year}-01-01" if year != "n.d." else "1970-01-01"


def _fallback_url(source: dict) -> str:
    url = (source.get("url") or "").strip()
    if url:
        return url

    slug_source = _safe_text(source.get("source") or source.get("title"), "document")
    slug = re.sub(r"\s+", "_", slug_source)
    return f"local://{slug}"


def _default_site_name(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "") if parsed.netloc else "Local Archive"


def format_reference(source: dict) -> str:
    """
    단일 출처를 REFERENCE 형식으로 변환합니다.

    입력값:
        source: {type, title, url, date, author, publisher, ...}
        type: "report" | "paper" | "webpage"

    반환값:
        포맷된 참고문헌 문자열
    """
    ref_type = _infer_reference_type(source)
    url = _fallback_url(source)

    if ref_type == "report":
        publisher = _safe_text(
            source.get("publisher") or source.get("author"),
            "발행기관 미상",
        )
        year = _extract_year(source)
        title = _safe_text(source.get("title") or source.get("source"), "제목 미상")
        return f"{publisher}({year}). *{title}*. {url}"

    if ref_type == "paper":
        author = _safe_text(source.get("author"), "저자 미상")
        year = _extract_year(source)
        title = _safe_text(source.get("title"), "제목 미상")
        journal = _safe_text(source.get("journal"), "학술지명 미상")
        volume = _safe_text(source.get("volume"), "0")
        issue = _safe_text(source.get("issue"), "0")
        pages = _safe_text(source.get("pages"), "1-1")
        return f"{author}({year}). {title}. *{journal}*, {volume}({issue}), {pages}."

    author = _safe_text(
        source.get("publisher") or source.get("author"),
        "기관명 미상",
    )
    date = _extract_date(source)
    title = _safe_text(source.get("title"), "제목 미상")
    site_name = _safe_text(source.get("site_name"), _default_site_name(url))
    return f"{author}({date}). *{title}*. {site_name}, {url}"


def format_all_references(sources: list[dict]) -> dict:
    """
    모든 출처를 카테고리별로 묶어서 포맷합니다.

    반환값:
        {
            "기관 보고서": ["발행기관(YYYY). 보고서명. URL", ...],
            "학술 논문": ["저자(YYYY). 제목. 학술지명, 권(호), 페이지.", ...],
            "웹페이지": ["기관명(YYYY-MM-DD). 제목. 사이트명, URL", ...]
        }
    """
    grouped = {
        "기관 보고서": [],
        "학술 논문": [],
        "웹페이지": [],
    }
    seen: set[str] = set()

    for source in sources:
        formatted = format_reference(source)
        if formatted in seen:
            continue
        seen.add(formatted)

        ref_type = _infer_reference_type(source)
        if ref_type == "report":
            grouped["기관 보고서"].append(formatted)
        elif ref_type == "paper":
            grouped["학술 논문"].append(formatted)
        else:
            grouped["웹페이지"].append(formatted)

    return grouped


def validate_reference_format(reference: str) -> bool:
    """
    참고문헌 형식이 정규식 패턴에 맞는지 검증합니다.

    기준:
        참고문헌 형식이 요구된 정규식 패턴을 만족해야 합니다.
    """
    patterns = {
        "report": r"^.+\((\d{4}|n\.d\.)\)\.\s.+\.\s(?:https?|local)://.+$",
        "paper": r"^.+\((\d{4}|n\.d\.)\)\.\s.+\.\s.+,\s[\w-]+\([\w-]+\),\s[\w-]+(?:-[\w-]+)?\.$",
        "webpage": r"^.+\(\d{4}-\d{2}-\d{2}\)\.\s.+\.\s.+,\s(?:https?|local)://.+$",
    }
    return any(re.match(pattern, reference) for pattern in patterns.values())
