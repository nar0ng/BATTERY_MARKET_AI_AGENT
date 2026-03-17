"""
보고서 초안 Agent
- 분석 결과를 보고서 초안으로 작성
- 검색 없이 앞선 3개 Agent의 결과를 조립만 함

Outcome: SUMMARY부터 REFERENCE까지 7개 필수 섹션이 모두 포함되고
         가이드 형식을 갖춘 보고서 초안이 완성된다.

Success Criteria:
  - 구조 완결성: 7개 섹션 전부 포함
  - SUMMARY 분량: 1/2 페이지(약 300단어) 이내
  - 참고문헌 형식: 기관보고서/학술논문/웹페이지 각 형식 준수
  - 근거 연결: 본문의 모든 주장이 출처에 연결
  - 내러티브 흐름:과거→현재→미래

Control Strategy:
  - Linear + Retry: 누락 섹션만 재생성 (max 1회)
"""
from __future__ import annotations

import re
from itertools import chain

from config.settings import REQUIRED_SECTIONS, SUMMARY_MAX_WORDS
from src.state import ReportState
from src.tools.reference_formatter import format_all_references


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _trim_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def _build_reference_sources(state: ReportState) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()

    for result in state.get("market_rag_results", []):
        key = f"doc:{result.get('source')}:{result.get('page')}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "type": "report",
                "title": result.get("source", "문서"),
                "source": result.get("source", "문서"),
                "publisher": result.get("source", "문서"),
                "date": "",
                "url": f"local://documents/{result.get('source', 'document')}",
                "source_kind": "document",
            }
        )

    for analysis in state.get("company_analyses", {}).values():
        for result in analysis.get("evidence", []):
            key = f"doc:{result.get('source')}:{result.get('page')}"
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "type": "report",
                    "title": result.get("source", "문서"),
                    "source": result.get("source", "문서"),
                    "publisher": result.get("source", "문서"),
                    "date": "",
                    "url": f"local://documents/{result.get('source', 'document')}",
                    "source_kind": "document",
                }
            )

    for result in state.get("market_web_results", []):
        key = f"web:{result.get('url')}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "type": "webpage",
                "title": result.get("title", "제목 미상"),
                "url": result.get("url", ""),
                "date": result.get("date", ""),
                "publisher": result.get("publisher", ""),
                "site_name": result.get("publisher", ""),
            }
        )

    return sources


_STRATEGY_SUMMARY_MAP = {
    "ESS 및 비EV 응용처 확대": "ESS와 비EV 응용처 확장",
    "배터리 케미스트리 포트폴리오 다변화": "배터리 케미스트리 다변화",
    "생산 거점과 지역 포트폴리오 분산": "생산 거점과 지역 포트폴리오 분산",
    "밸류체인 내재화 및 리사이클링 강화": "밸류체인 내재화와 리사이클링 강화",
    "고객 포트폴리오 다변화": "고객 포트폴리오 다변화",
    "생산능력과 재무 체력 관리": "증설 속도와 수익성의 균형 관리",
}

_RISK_SIGNAL_MAP = [
    (("원재료", "광물", "리튬", "니켈", "코발트"), "원재료 가격과 핵심광물 조달 변동성"),
    (("ira", "규제", "정책", "feoc", "유럽", "관세"), "정책·규제 변화에 따른 공급망 재편 리스크"),
    (("수요", "둔화", "가격 경쟁", "oversupply", "점유율"), "EV 수요 둔화와 가격 경쟁 심화"),
    (("투자", "capex", "가동률", "유휴", "수익성"), "증설과 가동률 조정 과정에서의 수익성 부담"),
]


def _join_phrases(phrases: list[str]) -> str:
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return "포트폴리오 다각화"
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} 및 {phrases[1]}"
    return f"{', '.join(phrases[:-1])} 및 {phrases[-1]}"


def _company_code(company: str) -> str:
    return "lg" if "LG" in company else "catl"


def _topic_particle(word: str) -> str:
    if not word:
        return "은"
    last_char = word[-1]
    if "가" <= last_char <= "힣":
        return "은" if (ord(last_char) - ord("가")) % 28 else "는"
    return "은"


def _footnote_key(result: dict) -> str:
    return str(result.get("chunk_id") or f"{result.get('source')}:{result.get('page')}")


def _build_company_footnotes(company: str, evidence: list[dict]) -> tuple[dict[str, str], list[str]]:
    company_code = _company_code(company)
    ref_ids: dict[str, str] = {}
    footnotes: list[str] = []

    for result in evidence:
        key = _footnote_key(result)
        if key in ref_ids:
            continue
        footnote_id = f"{company_code}-{len(ref_ids) + 1}"
        ref_ids[key] = footnote_id
        footnotes.append(
            f"[^{footnote_id}]: {result.get('source', '문서')}, p.{result.get('page', '?')}"
        )

    return ref_ids, footnotes


def _build_swot_footnotes(company: str, swot_data: dict) -> tuple[dict[str, str], list[str]]:
    company_code = _company_code(company)
    ref_ids: dict[str, str] = {}
    footnotes: list[str] = []

    for category in ("S", "W", "O", "T"):
        for item in swot_data.get(category, []):
            source = _sanitize_footnote_source(item.get("source") or item.get("evidence") or "분석 요약")
            factor = item.get("factor", "")
            key = f"{category}:{factor}:{source}"
            if key in ref_ids:
                continue
            footnote_id = f"swot-{company_code}-{len(ref_ids) + 1}"
            ref_ids[key] = footnote_id
            footnotes.append(f"[^{footnote_id}]: {source}")

    return ref_ids, footnotes


def _sanitize_footnote_source(source: str) -> str:
    source = " ".join((source or "").split()).strip()
    if not source:
        return "분석 요약"

    pdf_match = re.search(r"([A-Za-z0-9_-]+\.pdf,\s*p\.\d+)", source, flags=re.IGNORECASE)
    if pdf_match:
        return pdf_match.group(1)

    url_match = re.search(r"https?://[^\s\]]+", source)
    if url_match:
        return url_match.group(0)

    bracket_match = re.search(r"\[출처:\s*([^\]]+)\]", source)
    if bracket_match:
        return bracket_match.group(1).strip()

    if source in {"시장 배경 요약", "분석 요약"}:
        return source

    return source[:120]


def _footnote_marks(results: list[dict], ref_ids: dict[str, str]) -> str:
    marks: list[str] = []
    for result in results:
        footnote_id = ref_ids.get(_footnote_key(result))
        if footnote_id:
            marks.append(f"[^{footnote_id}]")
    return "".join(marks)


def _swot_footnote_mark(category: str, item: dict, ref_ids: dict[str, str]) -> str:
    key = f"{category}:{item.get('factor', '')}:{item.get('source') or '분석 요약'}"
    footnote_id = ref_ids.get(key)
    if not footnote_id:
        return ""
    return f"[^{footnote_id}]"


def _evidence_groups(evidence: list[dict]) -> dict[str, list[dict]]:
    return {
        "past": evidence[:2],
        "present": evidence[2:4],
        "future": evidence[4:6],
        "risk": evidence[6:8] or evidence[:2],
    }


def _strategy_phrases(analysis: dict, limit: int = 4) -> list[str]:
    mapped = [
        _STRATEGY_SUMMARY_MAP.get(strategy, strategy)
        for strategy in analysis.get("key_strategy", [])
    ]
    deduped: list[str] = []
    for item in mapped:
        if item in deduped:
            continue
        deduped.append(item)
    return deduped[:limit]


def _company_context_phrases(company: str, evidence: list[dict]) -> list[str]:
    text_blob = " ".join(result.get("chunk", "").lower() for result in evidence)
    phrases: list[str] = []

    if company == "LG에너지솔루션":
        if any(keyword in text_blob for keyword in ("북미", "유럽", "poland", "arizona")):
            phrases.append("북미·유럽 거점 재편")
        if any(keyword in text_blob for keyword in ("ess", "energy storage", "전환")):
            phrases.append("EV 라인의 ESS 전환")
        if any(keyword in text_blob for keyword in ("lmr", "46시리즈", "원통형")):
            phrases.append("차세대 제품 로드맵 확대")
    else:
        if any(keyword in text_blob for keyword in ("동력전지", "动力电池")):
            phrases.append("동력전지 중심 생산 기반")
        if any(keyword in text_blob for keyword in ("저장전지", "ess", "energy storage")):
            phrases.append("저장전지 사업 확대")
        if any(keyword in text_blob for keyword in ("헝가리", "독일", "스페인", "인도네시아")):
            phrases.append("해외 생산거점 확장")

    return phrases


def _time_sentences(company: str, analysis: dict) -> dict[str, str]:
    evidence = analysis.get("evidence", [])
    strategy_phrases = _strategy_phrases(analysis)
    context_phrases = _company_context_phrases(company, evidence)

    past_focus = _join_phrases((context_phrases[:1] + strategy_phrases[:2])[:3])
    present_focus = _join_phrases((context_phrases[1:2] + strategy_phrases[1:4])[:3])
    future_focus = _join_phrases((context_phrases[2:3] + strategy_phrases[2:])[:3])

    topic_particle = _topic_particle(company)
    return {
        "past": f"{company}{topic_particle} 과거에 {past_focus} 중심으로 사업 기반을 구축했습니다.",
        "present": f"현재는 {present_focus}를 병행하면서 제품 포트폴리오와 운영 효율을 함께 조정하고 있습니다.",
        "future": f"앞으로는 {future_focus}를 축으로 시장 대응력과 수익성 방어력을 높이려는 방향이 두드러집니다.",
    }


def _risk_bullets(company: str, analysis: dict) -> list[str]:
    evidence_text = " ".join(result.get("chunk", "") for result in analysis.get("evidence", []))
    risk_text = " ".join(analysis.get("risk_factors", []))
    text_blob = f"{evidence_text} {risk_text}".lower()

    bullets: list[str] = []
    for keywords, label in _RISK_SIGNAL_MAP:
        if any(keyword in text_blob for keyword in keywords):
            bullets.append(label)

    if not bullets:
        if company == "LG에너지솔루션":
            bullets = [
                "북미·유럽 증설과 라인 전환 과정에서의 수익성 부담",
                "정책 변화에 따른 지역별 수요 변동성",
            ]
        else:
            bullets = [
                "가격 경쟁 심화에 따른 마진 압박",
                "해외 확장과 공급망 통합 과정의 운영 리스크",
            ]

    deduped: list[str] = []
    for bullet in bullets:
        if bullet in deduped:
            continue
        deduped.append(bullet)
    return deduped[:3]


def _render_company_section(state: ReportState) -> tuple[str, list[str]]:
    sections: list[str] = []
    all_footnotes: list[str] = []
    for company in ("LG에너지솔루션", "CATL"):
        analysis = state.get("company_analyses", {}).get(company, {})
        evidence = analysis.get("evidence", [])
        grouped_evidence = _evidence_groups(evidence)
        ref_ids, footnotes = _build_company_footnotes(
            company,
            list(chain.from_iterable(grouped_evidence.values())),
        )
        time_sentences = _time_sentences(company, analysis)
        risk_bullets = _risk_bullets(company, analysis)

        risk_lines = [
            f"- {bullet}{_footnote_marks(grouped_evidence['risk'], ref_ids)}"
            for bullet in risk_bullets
        ]
        all_footnotes.extend(footnotes)
        sections.append(
            "\n".join(
                [
                    f"### {company}",
                    f"과거: {time_sentences['past']}{_footnote_marks(grouped_evidence['past'], ref_ids)}",
                    f"현재: {time_sentences['present']}{_footnote_marks(grouped_evidence['present'], ref_ids)}",
                    f"미래: {time_sentences['future']}{_footnote_marks(grouped_evidence['future'], ref_ids)}",
                    "핵심 전략:",
                    *[f"- {item}" for item in analysis.get("key_strategy", [])],
                    "주요 리스크:",
                    *risk_lines,
                ]
            ).strip()
        )
    return "\n\n".join(sections), all_footnotes


def _render_swot_items(category: str, items: list[dict], ref_ids: dict[str, str]) -> str:
    return "\n".join(
        f"- {item['factor']}{_swot_footnote_mark(category, item, ref_ids)}"
        for item in items
    )


def _render_comparison_table(comparison_data: dict | None) -> str:
    if not comparison_data:
        return "비교 데이터가 아직 충분하지 않아 정성 비교만 제공합니다."

    lines = [
        "| 차원 | LG에너지솔루션 | CATL |",
        "|---|---|---|",
    ]
    for row in comparison_data.get("dimensions", []):
        lines.append(
            f"| {row['dimension']} | {row['LG에너지솔루션']} | {row['CATL']} |"
        )
    verdict = comparison_data.get("verdict")
    if verdict:
        lines.extend(["", verdict])
    return "\n".join(lines)


def _render_swot_section(state: ReportState) -> tuple[str, list[str]]:
    swot_lg = state.get("swot_lg") or {"S": [], "W": [], "O": [], "T": []}
    swot_catl = state.get("swot_catl") or {"S": [], "W": [], "O": [], "T": []}
    lg_ref_ids, lg_footnotes = _build_swot_footnotes("LG에너지솔루션", swot_lg)
    catl_ref_ids, catl_footnotes = _build_swot_footnotes("CATL", swot_catl)

    section = "\n\n".join(
        [
            "### 핵심 전략 비교",
            _render_comparison_table(state.get("comparison_data")),
            "### LG에너지솔루션 SWOT",
            "#### Strength",
            _render_swot_items("S", swot_lg.get("S", []), lg_ref_ids),
            "#### Weakness",
            _render_swot_items("W", swot_lg.get("W", []), lg_ref_ids),
            "#### Opportunity",
            _render_swot_items("O", swot_lg.get("O", []), lg_ref_ids),
            "#### Threat",
            _render_swot_items("T", swot_lg.get("T", []), lg_ref_ids),
            "### CATL SWOT",
            "#### Strength",
            _render_swot_items("S", swot_catl.get("S", []), catl_ref_ids),
            "#### Weakness",
            _render_swot_items("W", swot_catl.get("W", []), catl_ref_ids),
            "#### Opportunity",
            _render_swot_items("O", swot_catl.get("O", []), catl_ref_ids),
            "#### Threat",
            _render_swot_items("T", swot_catl.get("T", []), catl_ref_ids),
            "### 전략 차이 요약",
            state.get("strategy_diff_summary", "전략 차이 요약 확보 필요"),
        ]
    )
    return section, [*lg_footnotes, *catl_footnotes]


def _render_implications(state: ReportState) -> str:
    comparison_verdict = (state.get("comparison_data") or {}).get(
        "verdict",
        "현재 비교 데이터가 제한적이므로 보수적 해석이 필요합니다.",
    )
    return "\n".join(
        [
            "1. 의사결정자는 EV 수요 회복 속도보다 제품·지역 다각화의 실행력 차이를 먼저 봐야 합니다.",
            f"2. {comparison_verdict}",
            "3. 향후 핵심 변수는 정책 지원 지속성, 가격 경쟁 압력, 차세대 케미스트리 전환 속도입니다.",
        ]
    )


def _render_reference_section(reference_groups: dict[str, list[str]]) -> tuple[str, list[dict]]:
    lines: list[str] = []
    flattened: list[dict] = []

    for category, references in reference_groups.items():
        if not references:
            continue
        lines.append(f"## {category}")
        for reference in references:
            lines.append(f"- {reference}")
            flattened.append({"category": category, "text": reference})
        lines.append("")

    return "\n".join(lines).strip(), flattened


def report_writer_node(state: ReportState) -> dict:
    """
    보고서 초안 Agent 메인 로직

    입력: 시장 분석 결과, 기업 분석 결과, SWOT 결과, 목차
    출력: 보고서 초안, Summary, Reference 초안

    작성 순서 (SUMMARY는 가장 마지막에 작성, 보고서에서는 맨 앞 배치):
    1. 시장 배경 섹션 (시장 분석 결과 기반, 과거→현재→미래)
    2. 기업별 전략 섹션 (기업 분석 결과)
    3. SWOT 섹션 (SWOT 결과, 내부/외부 레이블)
    4. 전략 비교 섹션 (전략 차이 요약)
    5. 시사점 섹션
    6. REFERENCE 편집 (형식 변환)
       - 기관 보고서: 발행기관(YYYY). 보고서명. URL
       - 학술 논문: 저자(YYYY). 논문제목. 학술지명, 권(호), 페이지.
       - 웹페이지: 기관명(YYYY-MM-DD). 제목. 사이트명, URL
    7. SUMMARY (맨 마지막 작성 → 맨 앞 배치)

    Note: RAG/Web 미사용. 앞선 결과를 조립만 함.
    """
    market_section = state.get("market_summary") or "시장 배경 요약 확보 필요"
    company_section, company_footnotes = _render_company_section(state)
    swot_section, swot_footnotes = _render_swot_section(state)
    implications_section = _render_implications(state)

    reference_groups = format_all_references(_build_reference_sources(state))
    reference_section, flattened_references = _render_reference_section(reference_groups)
    all_footnotes = [*company_footnotes, *swot_footnotes]
    if all_footnotes:
        footnote_block = "\n".join(all_footnotes)
        reference_section = f"{reference_section}\n\n## 각주\n{footnote_block}".strip()

    summary_text = _trim_words(
        " ".join(
            [
                "배터리 시장은 수요 변동성과 정책 재편이 동시에 전개되는 국면에 있습니다.",
                "LG에너지솔루션과 CATL은 모두 포트폴리오 다각화로 대응하지만 실행 축과 리스크 관리 방식에는 차이가 있습니다.",
                state.get("strategy_diff_summary", ""),
                "의사결정자는 제품·지역·밸류체인 분산과 수익성 방어력을 함께 봐야 합니다.",
            ]
        ).strip(),
        SUMMARY_MAX_WORDS,
    )

    section_map = {
        "SUMMARY": summary_text,
        "시장 배경": market_section,
        "기업별 포트폴리오 다각화 전략 및 핵심 경쟁력": company_section,
        "핵심 전략 비교 및 SWOT 분석": swot_section,
        "종합 시사점": implications_section,
        "REFERENCE": reference_section or "참고문헌 데이터 없음",
    }

    report_parts = []
    section_lengths: dict[str, int] = {}
    for heading in REQUIRED_SECTIONS:
        body = section_map.get(heading, "")
        report_parts.append(f"# {heading}\n{body}".strip())
        section_lengths[heading] = _count_words(body)

    return {
        "report_draft": "\n\n".join(report_parts).strip(),
        "final_report": None,
        "references": flattened_references,
        "summary": summary_text,
        "section_lengths": section_lengths,
        "quality_score": None,
        "quality_checked": False,
    }
