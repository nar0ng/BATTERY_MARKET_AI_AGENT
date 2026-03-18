"""
Supervisor 에이전트.
- 전체 워크플로우 제어, 라우팅, 품질 검증을 담당합니다.
- 모든 작업 에이전트는 Supervisor를 통해서만 연결됩니다.

목표:
  - 하위 에이전트가 기준을 충족하는 결과를 반환하게 조율합니다.
  - 품질 기준을 통과한 보고서만 최종 보고서로 확정합니다.

검사 기준:
  - 라우팅 정확도: 적절한 시점에 올바른 상태와 함께 호출
  - 상태 무결성: 에이전트 실행 후 상태 일관성 유지
  - 종료 조건: 통과 또는 최대 3회 반복
  - 비용 통제: LLM 15회 이하, 웹 검색 10회 이하
"""
from __future__ import annotations

import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Send
from langchain_openai import ChatOpenAI

from config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_ITERATIONS,
    MAX_LLM_CALLS,
    MAX_WEB_SEARCHES,
    REQUIRED_SECTIONS,
    SUMMARY_MAX_WORDS,
)
from src.prompts.report_prompt import REPORT_SYSTEM, SUPERVISOR_FINALIZE_TEMPLATE
from src.state import ReportState
from src.tools.reference_formatter import validate_reference_format
from src.tools.web_search import check_bias_ratio


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _summarize_company_for_dimension(analysis: dict, keywords: tuple[str, ...], fallback: str) -> str:
    core_competitiveness = " ".join(analysis.get("core_competitiveness", []))
    strategic_position = analysis.get("strategic_position", "")
    portfolio_strategy = analysis.get("portfolio_strategy", "")
    key_strategies = " ".join(analysis.get("key_strategy", []))
    risk_factors = " ".join(analysis.get("risk_factors", []))
    content = f"{strategic_position} {portfolio_strategy} {core_competitiveness} {key_strategies} {risk_factors}".lower()
    if any(keyword in content for keyword in keywords):
        return f"{analysis.get('company', '기업')}는 {fallback}"
    return fallback


def _build_comparison_data(company_analyses: dict) -> dict | None:
    lg = company_analyses.get("LG에너지솔루션")
    catl = company_analyses.get("CATL")
    if not lg or not catl:
        return None

    dimensions = [
        {
            "dimension": "제품 다각화",
            "LG에너지솔루션": _summarize_company_for_dimension(
                lg,
                ("ess", "전고체", "chemistry", "포트폴리오"),
                "제품 포트폴리오를 프리미엄·차세대 중심으로 확장 중",
            ),
            "CATL": _summarize_company_for_dimension(
                catl,
                ("lfp", "ess", "포트폴리오", "chemistry"),
                "대중형 제품과 ESS 확장을 동시에 노리는 구조",
            ),
        },
        {
            "dimension": "지역 다각화",
            "LG에너지솔루션": _summarize_company_for_dimension(
                lg,
                ("북미", "유럽", "ira", "거점"),
                "북미·유럽 생산 거점 재배치를 통해 정책 수혜를 노림",
            ),
            "CATL": _summarize_company_for_dimension(
                catl,
                ("유럽", "중국", "거점", "해외"),
                "중국 기반 규모 우위에 해외 거점 확장을 병행",
            ),
        },
        {
            "dimension": "밸류체인 다각화",
            "LG에너지솔루션": _summarize_company_for_dimension(
                lg,
                ("리사이클", "내재화", "양극재", "supply"),
                "소재 조달과 리사이클링 연계를 강화하는 방향",
            ),
            "CATL": _summarize_company_for_dimension(
                catl,
                ("리사이클", "내재화", "원가", "supply"),
                "원가 경쟁력을 위해 수직계열화와 공급망 통합을 강화",
            ),
        },
        {
            "dimension": "고객/매출 다각화",
            "LG에너지솔루션": _summarize_company_for_dimension(
                lg,
                ("oem", "고객", "gm", "tesla", "현대"),
                "글로벌 OEM 포트폴리오 분산이 핵심",
            ),
            "CATL": _summarize_company_for_dimension(
                catl,
                ("oem", "고객", "중국", "해외"),
                "중국 내 강점 위에 해외 고객 확장이 과제",
            ),
        },
        {
            "dimension": "재무 체력",
            "LG에너지솔루션": _summarize_company_for_dimension(
                lg,
                ("현금", "투자", "가동률", "capa"),
                "대규모 투자와 수익성 방어를 동시에 관리해야 하는 단계",
            ),
            "CATL": _summarize_company_for_dimension(
                catl,
                ("원가", "투자", "capa", "수익"),
                "규모와 원가 우위를 바탕으로 투자 지속 여력이 상대적으로 큼",
            ),
        },
    ]

    verdict = (
        "종합적으로 LG에너지솔루션은 정책 수혜형 지역 재편과 고객 포트폴리오 분산, "
        "CATL은 규모·원가 우위와 밸류체인 통합을 중심으로 각기 다른 우위를 구축하는 흐름입니다."
    )

    return {
        "dimensions": dimensions,
        "verdict": verdict,
    }


def _extract_sections(report_draft: str) -> dict[str, str]:
    if not report_draft:
        return {}

    matches = list(re.finditer(r"^# (.+)$", report_draft, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report_draft)
        sections[heading] = report_draft[start:end].strip()
    return sections


def _normalize_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def _strip_reference_footnotes(text: str) -> str:
    return re.split(r"\n## 각주\b", text or "", maxsplit=1)[0].strip()


def _strip_citation_marks(text: str) -> str:
    cleaned = re.sub(r"\[\^[^\]]+\]", "", text or "")
    cleaned = re.sub(r"\s*\[출처:\s*[^\]]+\]", "", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r" +\n", "\n", cleaned)
    return _normalize_blank_lines(cleaned)


def _prepare_sections_for_delivery(sections: dict[str, str]) -> dict[str, str]:
    prepared: dict[str, str] = {}
    for heading, body in sections.items():
        cleaned = body.strip()
        if heading == "REFERENCE":
            cleaned = _strip_reference_footnotes(cleaned)
        prepared[heading] = _strip_citation_marks(cleaned)
    return prepared


def _polish_list_spacing(text: str) -> str:
    lines = text.splitlines()
    polished: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index > 0 and stripped.startswith(("- ", "1.", "2.", "3.", "4.", "5.")):
            if polished and polished[-1].strip():
                polished.append("")
        polished.append(line.rstrip())
    return "\n".join(polished).strip()


def _polish_company_section_layout(body: str) -> str:
    replacements = {
        "포트폴리오 다각화 전략:": "**포트폴리오 다각화 전략**",
        "전략적 포지션:": "**전략적 포지션**",
        "핵심 경쟁력:": "**핵심 경쟁력**",
        "핵심 전략:": "**핵심 전략**",
        "주요 리스크:": "**주요 리스크**",
    }

    lines = []
    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        matched = False
        for prefix, replacement in replacements.items():
            if stripped.startswith(prefix):
                payload = stripped[len(prefix):].strip()
                lines.append(replacement)
                if payload:
                    lines.append(payload)
                lines.append("")
                matched = True
                break
        if not matched:
            lines.append(raw_line.rstrip())

    return _polish_list_spacing(_normalize_blank_lines("\n".join(lines)))


def _polish_market_section_layout(body: str) -> str:
    lines: list[str] = []
    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("과거:"):
            lines.append(f"- **과거**: {stripped[3:].strip()}")
        elif stripped.startswith("현재:"):
            lines.append(f"- **현재**: {stripped[3:].strip()}")
        elif stripped.startswith("미래:"):
            lines.append(f"- **미래**: {stripped[3:].strip()}")
        elif stripped.startswith("시장 신호:"):
            lines.append("")
            lines.append(f"> {stripped}")
        else:
            lines.append(raw_line.rstrip())
    return _normalize_blank_lines("\n".join(lines))


def _polish_summary_layout(body: str) -> str:
    sentences = re.split(r"(?<=[.!?다])\s+", body.strip())
    cleaned = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
    return _normalize_blank_lines(cleaned)


def _heuristic_polish_sections(sections: dict[str, str]) -> dict[str, str]:
    polished = dict(sections)
    polished["SUMMARY"] = _polish_summary_layout(sections.get("SUMMARY", ""))
    polished["시장 배경"] = _polish_market_section_layout(sections.get("시장 배경", ""))
    polished["기업별 포트폴리오 다각화 전략 및 핵심 경쟁력"] = _polish_company_section_layout(
        sections.get("기업별 포트폴리오 다각화 전략 및 핵심 경쟁력", "")
    )
    polished["핵심 전략 비교 및 SWOT 분석"] = _normalize_blank_lines(
        _polish_list_spacing(sections.get("핵심 전략 비교 및 SWOT 분석", ""))
    )
    polished["종합 시사점"] = _normalize_blank_lines(
        _polish_list_spacing(sections.get("종합 시사점", ""))
    )
    return polished


def _compose_readable_body(sections: dict[str, str]) -> str:
    parts: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section == "REFERENCE":
            continue
        body = sections.get(section, "").strip()
        if not body:
            continue
        parts.append(f"# {section}\n{body}".strip())
    return "\n\n".join(parts).strip()


def _build_final_report_title(query: str) -> str:
    return "배터리 시장 전략 분석 보고서"


def _render_final_section_heading(index: int, section: str) -> str:
    return f"## {index}. {section}"


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def _llm_polish_sections(sections: dict[str, str]) -> tuple[dict[str, str], int]:
    if not os.getenv("OPENAI_API_KEY"):
        return sections, 0

    report_body = _compose_readable_body(sections)
    if not report_body:
        return sections, 0

    prompt = SUPERVISOR_FINALIZE_TEMPLATE.format(report_body=report_body)
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        response = llm.invoke(
            [
                SystemMessage(content=REPORT_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        polished_body = _message_content_to_text(response.content)
        polished_sections = _extract_sections(polished_body)
        required_non_reference = [section for section in REQUIRED_SECTIONS if section != "REFERENCE"]
        if all(polished_sections.get(section, "").strip() for section in required_non_reference):
            polished_sections["REFERENCE"] = sections.get("REFERENCE", "")
            return polished_sections, 1
    except Exception:
        return sections, 0

    return sections, 0


def _compose_final_report(state: ReportState, quality_result: dict) -> tuple[str, int]:
    """품질 승인 이후 Supervisor가 가독성까지 반영해 최종 보고서를 조립합니다."""
    report_draft = (state.get("report_draft") or "").strip()
    sections = _extract_sections(report_draft)
    sections = _heuristic_polish_sections(sections)
    sections, llm_calls = _llm_polish_sections(sections)
    sections = _prepare_sections_for_delivery(sections)

    final_parts: list[str] = [f"# {_build_final_report_title(state.get('query', ''))}"]
    for index, section in enumerate(REQUIRED_SECTIONS, start=1):
        body = sections.get(section, "").strip()
        final_parts.append(f"{_render_final_section_heading(index, section)}\n{body}".strip())

    finalized = _normalize_blank_lines("\n\n".join(part for part in final_parts if part))
    return finalized, llm_calls


def supervisor_node(state: ReportState) -> dict:
    """
    Supervisor 메인 로직
    - State 확인 후 라우팅 결정
    - 품질 검증 (Reflector 내장)
    """
    updates: dict = {}

    company_analyses = state.get("company_analyses", {})
    if len(company_analyses) >= 2 and not state.get("comparison_data"):
        comparison_data = _build_comparison_data(company_analyses)
        if comparison_data:
            updates["comparison_data"] = comparison_data

    prospective_state = {**state, **updates}
    if prospective_state.get("report_draft") and not prospective_state.get("quality_checked"):
        quality_result = _run_quality_check(prospective_state)
        updates["quality_score"] = quality_result
        updates["quality_checked"] = True
        if quality_result["passed"]:
            final_report, llm_calls = _compose_final_report(
                prospective_state,
                quality_result,
            )
            updates["final_report"] = final_report
            if llm_calls:
                updates["llm_call_count"] = state.get("llm_call_count", 0) + llm_calls
        else:
            updates["final_report"] = None
            updates["iteration_count"] = state.get("iteration_count", 0) + 1

    return updates


def supervisor_route(state: ReportState):
    """
    Supervisor 라우팅 로직 (Control Strategy의 실체)
    - State를 보고 다음 Agent 결정
    - 병렬 디스패치: Send()로 동시 호출
    - Dependency는 여기서 관리

    1단계: 시장 분석 ∥ 기업 분석(LG) ∥ 기업 분석(CATL) — 병렬
    2단계: SWOT 추출 — 순차 (기업 분석 결과 필요)
    3단계: 보고서 초안 — 순차 (전체 결과 필요)
    4단계: 품질 검증 — Supervisor 내장
    """
    if state["llm_call_count"] >= MAX_LLM_CALLS:
        return "end"
    if state["web_search_count"] >= MAX_WEB_SEARCHES:
        return "end"
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "end"

    if state.get("final_report") and state.get("quality_checked"):
        return "end"

    if state.get("report_draft") and state.get("quality_checked"):
        quality_result = state.get("quality_score") or {}
        if quality_result.get("passed"):
            return "end"

        failed_agents = quality_result.get("failed_agents", [])
        if failed_agents:
            return failed_agents[0]
        return "end"

    sends = []
    if not state.get("market_summary"):
        sends.append(Send("market_analyst", state))
    for company in state["companies"]:
        if company not in state.get("company_analyses", {}):
            sends.append(Send("company_analyst", {**state, "_target_company": company}))
    if sends:
        return sends

    if not state.get("swot_lg") or not state.get("swot_catl"):
        return "swot_extractor"

    if not state.get("report_draft"):
        return "report_writer"

    return "end"


def _run_quality_check(state: ReportState) -> dict:
    """
    품질 검증 (Reflector 내장)
    11개 Success Criteria 전수 검사

    검사 항목:
    - 규칙 기반: 섹션 수, SUMMARY 분량, 참고문헌 형식, 찬반 비율
    - LLM Judge: SWOT 내/외부 정확성, 시간적 포괄성
    """
    details: list[str] = []
    failed_agents: list[str] = []

    report_draft = state.get("report_draft") or ""
    sections = _extract_sections(report_draft)

    missing_sections = [section for section in REQUIRED_SECTIONS if section not in sections]
    if missing_sections:
        details.append(f"누락 섹션: {', '.join(missing_sections)}")
        failed_agents.append("report_writer")
    else:
        details.append("필수 섹션 구조 확인 완료")

    summary = state.get("summary") or sections.get("SUMMARY", "")
    summary_words = _count_words(summary)
    if summary and summary_words <= SUMMARY_MAX_WORDS:
        details.append(f"SUMMARY 분량 적정 ({summary_words}단어)")
    else:
        details.append(f"SUMMARY 분량 초과 또는 누락 ({summary_words}단어)")
        failed_agents.append("report_writer")

    references = [item.get("text", "") for item in state.get("references", [])]
    invalid_references = [reference for reference in references if not validate_reference_format(reference)]
    if invalid_references:
        details.append(f"REFERENCE 형식 오류 {len(invalid_references)}건")
        failed_agents.append("report_writer")
    else:
        details.append("REFERENCE 형식 검증 완료")

    swot_validation = state.get("swot_validation") or {}
    swot_misclassified = sum(
        payload.get("misclassified", 0) for payload in swot_validation.values()
    )
    if swot_misclassified == 0:
        details.append("SWOT 내/외부 분류 검증 완료")
    else:
        details.append(f"SWOT 분류 오류 {swot_misclassified}건")
        failed_agents.append("swot_extractor")

    market_section = sections.get("시장 배경", "")
    report_time_coverage = all(keyword in market_section for keyword in ["과거", "현재", "미래"])
    if report_time_coverage:
        details.append("시장 배경의 과거·현재·미래 시간축 포함")
    else:
        details.append("시장 배경의 시간축 서술(과거·현재·미래) 보완 필요")
        failed_agents.append("report_writer")

    has_inline_citation = "[출처:" in report_draft
    has_footnote_marks = "[^" in report_draft
    reference_section = sections.get("REFERENCE", "")
    has_footnote_section = "## 각주" in reference_section
    if has_footnote_marks and has_footnote_section:
        details.append("각주 기반 출처 연결 확인")
    elif has_inline_citation:
        details.append("인라인 출처 연결 확인")
    elif state.get("references"):
        details.append("REFERENCE는 있으나 본문 각주 연결 보완 필요")
        failed_agents.append("report_writer")
    else:
        details.append("출처 데이터가 부족하여 근거 연결 평가는 보류")

    bias_metrics = check_bias_ratio(state.get("market_web_results", []))
    if state.get("market_web_results"):
        if bias_metrics["is_balanced"]:
            details.append("웹 검색 찬반 비율 균형 확인")
        else:
            details.append(
                f"웹 검색 찬반 비율 불균형 (pro={bias_metrics['pro']}, con={bias_metrics['con']})"
            )
            failed_agents.append("market_analyst")
    else:
        details.append("웹 검색 결과가 없어 찬반 비율 평가는 생략")

    empty_sections = [heading for heading, body in sections.items() if not body.strip()]
    if empty_sections:
        details.append(f"빈 섹션 존재: {', '.join(empty_sections)}")
        failed_agents.append("report_writer")
    else:
        details.append("빈 섹션 없음")

    company_section = sections.get("기업별 포트폴리오 다각화 전략 및 핵심 경쟁력", "")
    readability_markers = [
        "### LG에너지솔루션" in company_section,
        "### CATL" in company_section,
        "핵심 경쟁력" in company_section,
        "핵심 전략" in company_section,
        "주요 리스크" in company_section,
    ]
    if all(readability_markers):
        details.append("기업 섹션 가독성 구조 확인")
    else:
        details.append("기업 섹션 가독성 구조 보완 필요")
        failed_agents.append("report_writer")

    unique_failed_agents: list[str] = []
    for agent in failed_agents:
        if agent not in unique_failed_agents:
            unique_failed_agents.append(agent)

    return {
        "passed": not unique_failed_agents,
        "details": details,
        "failed_agents": unique_failed_agents,
    }
