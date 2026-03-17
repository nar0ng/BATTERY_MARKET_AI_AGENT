"""
공유 상태 스키마 (LangGraph TypedDict)
모든 에이전트가 읽고 쓰는 단일 State. Supervisor가 중계한다.
"""
from typing import Annotated, Optional, TypedDict


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """병렬 LangGraph 브랜치에서 나온 부분 딕셔너리 갱신값을 합칩니다."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class SWOTData(TypedDict):
    """기업별 SWOT 구조"""
    S: list[dict]  # [{factor, type: "internal", evidence, source}]
    W: list[dict]
    O: list[dict]
    T: list[dict]


class QualityScore(TypedDict):
    """품질 검증 결과"""
    passed: bool
    details: list[str]           # 각 기준별 결과 메시지
    failed_agents: list[str]     # 재호출 필요한 Agent 이름 리스트


class ReportState(TypedDict):
    """전체 시스템 공유 상태"""

    # ── 사용자 입력 ──
    query: str                          # 사용자 원본 질의
    companies: list[str]                # 대상 기업: ["LG에너지솔루션", "CATL"]

    # ── 시장 분석 Agent 산출 (T1) ──
    market_rag_results: list[dict]      # RAG 청크: {chunk, score, page, source}
    market_web_results: list[dict]      # 웹 결과: {title, url, date, snippet, pro_con_tag}
    market_summary: Optional[str]       # 시장 배경 요약

    # ── 기업 분석 Agent 산출 (T2) ──
    company_analyses: Annotated[dict, merge_dicts]  # {LG: {...}, CATL: {...}}
    comparison_data: Optional[dict]     # 기업 간 비교 데이터

    # ── SWOT 추출 Agent 산출 (T3) ──
    swot_lg: Optional[SWOTData]                 # LG SWOT
    swot_catl: Optional[SWOTData]               # CATL SWOT
    strategy_diff_summary: Optional[str]        # 전략 차이 요약
    swot_validation: Optional[dict]             # {LG: {misclassified: int}, CATL: {misclassified: int}}

    # ── 보고서 초안 Agent 산출 (T4) ──
    report_draft: Optional[str]         # 완성된 보고서 초안
    final_report: Optional[str]         # Supervisor가 승인한 최종 보고서
    references: list[dict]              # 카테고리별 포맷된 참고문헌
    summary: Optional[str]              # SUMMARY 섹션
    section_lengths: Optional[dict]     # {섹션명: 단어수}

    # ── Supervisor 관리 ──
    quality_score: Optional[QualityScore]   # 품질 검증 결과
    quality_checked: bool                   # 검증 수행 여부
    iteration_count: int                    # 현재 재시도 횟수 (max 3)
    next_agent: Optional[str]               # 라우팅 결정값
    llm_call_count: int                     # LLM 호출 카운터
    web_search_count: int                   # 웹 검색 카운터
    _target_company: Optional[str]          # Supervisor가 Send() 시 주입


def create_initial_state(query: str) -> ReportState:
    """초기 상태 생성"""
    from config.settings import TARGET_COMPANIES

    return ReportState(
        query=query,
        companies=TARGET_COMPANIES,
        market_rag_results=[],
        market_web_results=[],
        market_summary=None,
        company_analyses={},
        comparison_data=None,
        swot_lg=None,
        swot_catl=None,
        strategy_diff_summary=None,
        swot_validation=None,
        report_draft=None,
        final_report=None,
        references=[],
        summary=None,
        section_lengths=None,
        quality_score=None,
        quality_checked=False,
        iteration_count=0,
        next_agent=None,
        llm_call_count=0,
        web_search_count=0,
    )
