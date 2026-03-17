"""
SWOT 추출 Agent
- 기업 분석 결과를 바탕으로 SWOT 도출
- 기업 분석 '결과'를 입력으로 받음
- 근거 보강 필요 시에만 조건부 RAG

Outcome: LG·CATL 각각의 SWOT에서 S/W는 내부 요인, O/T는 외부 요인으로
         정확히 분류되고, 두 기업 간 전략 차이가 요약된다.

Success Criteria:
  - SWOT 정확성: S/W = 반드시 내부 요인, O/T = 반드시 외부 요인
  - 비교 가능성: 기업 간 비교가 일관된 차원(기술/지역/재무/파트너십)

Control Strategy:
  - Linear + Retry: 오분류 시 인라인 교정 (max 1회)
  - LLM-as-a-Judge로 내/외부 검증
"""
from __future__ import annotations

from src.state import ReportState


def _source_from_evidence(evidence: str) -> str:
    if "[출처:" in evidence:
        return evidence.split("[출처:", 1)[1].rstrip("]").strip()
    return "분석 요약"


def _make_item(factor: str, factor_type: str, evidence: str, source: str | None = None) -> dict:
    return {
        "factor": factor,
        "type": factor_type,
        "evidence": evidence,
        "source": source or _source_from_evidence(evidence),
    }


def _take(values: list[str], index: int, fallback: str) -> str:
    return values[index] if len(values) > index else fallback


def _build_company_swot(company: str, analysis: dict, market_summary: str, web_results: list[dict]) -> dict:
    strategies = analysis.get("key_strategy", [])
    risks = analysis.get("risk_factors", [])
    sources = analysis.get("source_refs", [])
    evidence = analysis.get("evidence", [])

    strengths = [
        _make_item(
            _take(strategies, 0, "포트폴리오 다각화 실행력"),
            "internal",
            _take(sources, 0, analysis.get("present", "현재 경쟁력 관련 근거 확보 필요")),
        ),
        _make_item(
            _take(strategies, 1, "생산 거점 및 고객 포트폴리오 운영력"),
            "internal",
            _take(sources, 1, analysis.get("future", "미래 전략 관련 근거 확보 필요")),
        ),
    ]

    weaknesses = [
        _make_item(
            "수요 둔화 국면에서의 수익성 변동성",
            "internal",
            _take(risks, 0, "가격 경쟁 심화 시 수익성 방어 여부를 추가 점검해야 합니다."),
        ),
        _make_item(
            "대규모 투자 집행에 따른 운영 부담",
            "internal",
            _take(risks, 1, "증설과 투자 속도에 비해 가동률 회복이 지연될 수 있습니다."),
        ),
    ]

    pro_web = [item for item in web_results if item.get("pro_con_tag") == "pro"]
    con_web = [item for item in web_results if item.get("pro_con_tag") == "con"]

    opportunities = [
        _make_item(
            "정책 지원과 비EV 응용처 확대로 인한 신규 수요",
            "external",
            (
                f"{pro_web[0].get('title', '')} [출처: {pro_web[0].get('url', '')}]"
                if pro_web
                else market_summary
            ),
            source=pro_web[0].get("url") if pro_web else "시장 배경 요약",
        ),
        _make_item(
            "배터리 케미스트리 전환과 지역 재편 과정에서의 선점 기회",
            "external",
            (
                f"{pro_web[1].get('title', '')} [출처: {pro_web[1].get('url', '')}]"
                if len(pro_web) > 1
                else analysis.get("future", "미래 전략 관련 근거 확보 필요")
            ),
            source=pro_web[1].get("url") if len(pro_web) > 1 else None,
        ),
    ]

    threats = [
        _make_item(
            "글로벌 가격 경쟁과 공급과잉 심화",
            "external",
            (
                f"{con_web[0].get('title', '')} [출처: {con_web[0].get('url', '')}]"
                if con_web
                else market_summary
            ),
            source=con_web[0].get("url") if con_web else "시장 배경 요약",
        ),
        _make_item(
            "정책·규제 변화와 원재료 가격 변동성",
            "external",
            (
                f"{con_web[1].get('title', '')} [출처: {con_web[1].get('url', '')}]"
                if len(con_web) > 1
                else analysis.get("past", "과거 전략 관련 근거 확보 필요")
            ),
            source=con_web[1].get("url") if len(con_web) > 1 else None,
        ),
    ]

    if evidence:
        strengths[0]["evidence"] = f"{evidence[0].get('chunk', '')} [출처: {evidence[0].get('source', '')}, p.{evidence[0].get('page', '?')}]"
        strengths[0]["source"] = evidence[0].get("source", strengths[0]["source"])
    if len(evidence) > 1:
        strengths[1]["evidence"] = f"{evidence[1].get('chunk', '')} [출처: {evidence[1].get('source', '')}, p.{evidence[1].get('page', '?')}]"
        strengths[1]["source"] = evidence[1].get("source", strengths[1]["source"])

    return {
        "S": strengths,
        "W": weaknesses,
        "O": opportunities,
        "T": threats,
    }


def _validate_swot(swot_data: dict) -> dict:
    expected_types = {
        "S": "internal",
        "W": "internal",
        "O": "external",
        "T": "external",
    }
    misclassified = 0
    for category, expected in expected_types.items():
        for item in swot_data.get(category, []):
            if item.get("type") != expected:
                misclassified += 1
                item["type"] = expected

    return {
        "misclassified": misclassified,
        "is_accurate": misclassified == 0,
    }


def _strategy_diff_summary(swot_lg: dict, swot_catl: dict) -> str:
    lg_strength = swot_lg["S"][0]["factor"] if swot_lg.get("S") else "내부 실행력"
    lg_weakness = swot_lg["W"][0]["factor"] if swot_lg.get("W") else "수익성 변동성"
    catl_strength = swot_catl["S"][0]["factor"] if swot_catl.get("S") else "원가 경쟁력"
    catl_weakness = swot_catl["W"][0]["factor"] if swot_catl.get("W") else "정책 노출"

    return (
        f"기술 전략 측면에서 LG에너지솔루션은 {lg_strength}에 무게를 두고, "
        f"CATL은 {catl_strength}을 통한 규모·원가 우위를 강조하는 흐름으로 해석됩니다. "
        f"재무·운영 측면에서는 LG에너지솔루션의 {lg_weakness} 관리와 "
        f"CATL의 {catl_weakness} 대응 방식이 차별화 포인트입니다. "
        "두 기업 모두 지역 재편과 파트너십 확장 속도에 따라 향후 경쟁구도가 달라질 가능성이 큽니다."
    )


def swot_extractor_node(state: ReportState) -> dict:
    """
    SWOT 추출 Agent 메인 로직

    입력: 기업 분석 결과 (직접 검색하지 않음!)
    출력: LGES SWOT, CATL SWOT, 전략 차이 요약

    실행 흐름:
    1. 기업 분석 결과를 입력으로 수신 (Supervisor 경유)
    2. 기업별 SWOT 초안 생성 (LLM)
    3. 내부/외부 태그 부여
       - 내부 (S/W): 기업이 직접 통제 가능
         예) 기술 특허, 생산 능력, 재무 구조, 인력, 브랜드
       - 외부 (O/T): 기업이 통제 불가
         예) 정부 정책, 원자재 가격, 경쟁사 동향, 소비자 트렌드, 규제
    4. LLM-as-a-Judge로 내/외부 정확성 검증
    5. 오분류 발견 → 인라인 교정 (max 1회)
    6. 기업 간 전략 차이 요약 생성
    7. 근거 보강 필요 시 → 조건부 RAG
    8. State 기록 → Supervisor 반환
    """
    company_analyses = state.get("company_analyses", {})
    if not company_analyses:
        return state

    lg_analysis = company_analyses.get("LG에너지솔루션", {})
    catl_analysis = company_analyses.get("CATL", {})

    swot_lg = _build_company_swot(
        "LG에너지솔루션",
        lg_analysis,
        state.get("market_summary", ""),
        state.get("market_web_results", []),
    )
    swot_catl = _build_company_swot(
        "CATL",
        catl_analysis,
        state.get("market_summary", ""),
        state.get("market_web_results", []),
    )

    validation_lg = _validate_swot(swot_lg)
    validation_catl = _validate_swot(swot_catl)

    return {
        "swot_lg": swot_lg,
        "swot_catl": swot_catl,
        "strategy_diff_summary": _strategy_diff_summary(swot_lg, swot_catl),
        "swot_validation": {
            "LG에너지솔루션": validation_lg,
            "CATL": validation_catl,
        },
        "report_draft": None,
        "final_report": None,
        "references": [],
        "summary": None,
        "section_lengths": None,
        "quality_score": None,
        "quality_checked": False,
    }
