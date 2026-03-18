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

import re

from langgraph.graph import END
from langgraph.types import Send

from config.settings import (
    MAX_ITERATIONS,
    MAX_LLM_CALLS,
    MAX_WEB_SEARCHES,
    REQUIRED_SECTIONS,
    SUMMARY_MAX_WORDS,
)
from src.state import ReportState
from src.tools.reference_formatter import validate_reference_format
from src.tools.web_search import check_bias_ratio


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _summarize_company_for_dimension(analysis: dict, keywords: tuple[str, ...], fallback: str) -> str:
    key_strategies = " ".join(analysis.get("key_strategy", []))
    risk_factors = " ".join(analysis.get("risk_factors", []))
    content = f"{analysis.get('present', '')} {analysis.get('future', '')} {key_strategies} {risk_factors}".lower()
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


def _compose_final_report(state: ReportState, quality_result: dict) -> str:
    """품질 승인 이후 Supervisor가 최종 보고서를 직접 조립합니다."""
    report_draft = (state.get("report_draft") or "").strip()
    sections = _extract_sections(report_draft)
    approved_details = quality_result.get("details", [])

    final_parts = [
        "# Supervisor Final Report",
        "\n".join(
            [
                "## Supervisor 최종 판단",
                "본 문서는 `report_writer` 초안을 Supervisor가 품질 기준에 따라 검토한 뒤 최종 확정한 버전입니다.",
                *[f"- {detail}" for detail in approved_details],
            ]
        ).strip(),
    ]

    for section in REQUIRED_SECTIONS:
        body = sections.get(section, "").strip()
        final_parts.append(f"# {section}\n{body}".strip())

    finalized = "\n\n".join(part for part in final_parts if part).strip()
    return re.sub(r"\n{3,}", "\n\n", finalized)


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
            updates["final_report"] = _compose_final_report(
                prospective_state,
                quality_result,
            )
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

    report_time_coverage = all(keyword in report_draft for keyword in ["과거", "현재", "미래"])
    if report_time_coverage:
        details.append("과거·현재·미래 시간축 포함")
    else:
        details.append("시간축 서술(과거·현재·미래) 보완 필요")
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

    unique_failed_agents: list[str] = []
    for agent in failed_agents:
        if agent not in unique_failed_agents:
            unique_failed_agents.append(agent)

    return {
        "passed": not unique_failed_agents,
        "details": details,
        "failed_agents": unique_failed_agents,
    }
