"""
시장 분석 Agent
- 배터리 시장 환경 및 산업 배경 분석
- RAG + Web Search 사용

Outcome: 배터리 시장의 환경 변화와 산업 구조에 대한 문서+웹 근거가
         출처 정보(REFERENCE)와 함께 수집된다.

Success Criteria:
  - 관련성: 검색 청크 유사도 ≥ 0.65
  - 중복 제거: 코사인 유사도 > 0.95 제거
  - 추적 가능성: 페이지 번호 + 문서명 포함
  - 출처 명확성: 웹 자료가 제목/URL/작성일/기관명 포함 → REFERENCE

Control Strategy:
  - Loop: Query Rewrite (max 2회)
  - Conditional Branch: 찬반 비율 > 7:3 시 보충 검색
"""
from __future__ import annotations

from src.state import ReportState
from src.tools.rag import deduplicate, rewrite_query, search as rag_search
from src.tools.web_search import (
    check_bias_ratio,
    classify_pro_con,
    generate_balanced_queries,
    search as web_search,
    supplement_minority_view,
)
from config.settings import MAX_QUERY_REWRITE


def _build_market_queries(query: str) -> list[str]:
    base_query = query.strip()
    return [
        f"{base_query} 배터리 시장 환경 변화",
        "배터리 산업 구조 공급망 정책 변화",
        "전기차 수요 둔화 배터리 시장 대응 전략",
        "배터리 시장 미래 전망 기술 트렌드 규제",
    ]


def _safe_rag_search(query: str) -> list[dict]:
    try:
        return rag_search(query, top_k=5)
    except Exception:
        return []


def _run_rag_loop(query: str) -> list[dict]:
    current_query = query
    results: list[dict] = []

    for _ in range(MAX_QUERY_REWRITE + 1):
        results = _safe_rag_search(current_query)
        if results:
            average_score = sum(item["score"] for item in results) / len(results)
            if average_score >= 0.65:
                break
        current_query = rewrite_query(current_query, results)

    return results


def _rag_citation(result: dict) -> str:
    source = result.get("source", "문서")
    page = result.get("page", "?")
    return f"[출처: {source}, p.{page}]"


def _web_citation(result: dict) -> str:
    return f"[출처: {result.get('url', 'URL 없음')}]"


def _snippet(text: str, limit: int = 130) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _compose_market_summary(
    query: str,
    rag_results: list[dict],
    web_results: list[dict],
) -> str:
    top_rag = rag_results[:3]
    pro_results = [item for item in web_results if item.get("pro_con_tag") == "pro"][:2]
    con_results = [item for item in web_results if item.get("pro_con_tag") == "con"][:2]

    past_line = (
        "과거: "
        + " / ".join(
            f"{_snippet(item['chunk'])} {_rag_citation(item)}" for item in top_rag[:2]
        )
        if top_rag
        else f"과거: {query}와 관련된 문서 근거가 아직 충분히 적재되지 않아 산업 형성 초기 맥락을 일반론 수준에서만 정리할 수 있습니다."
    )
    present_line = (
        "현재: "
        + " / ".join(
            f"{_snippet(item.get('snippet') or item.get('title', ''))} {_web_citation(item)}"
            for item in (pro_results + con_results)[:2]
        )
        if web_results
        else "현재: 웹 검색 결과가 없어 현재 시장 신호는 문서 기반 근거 중심으로 해석해야 합니다."
    )
    future_inputs = top_rag[2:3] + pro_results[:1] + con_results[:1]
    future_line = (
        "미래: "
        + " / ".join(
            f"{_snippet(item.get('chunk') or item.get('snippet') or item.get('title', ''))} "
            f"{_rag_citation(item) if item.get('chunk') else _web_citation(item)}"
            for item in future_inputs
        )
        if future_inputs
        else "미래: 기술 전환과 정책 변수에 따라 성장 기회와 수익성 리스크가 동시에 확대될 가능성이 있습니다."
    )

    balance = check_bias_ratio(web_results) if web_results else {"pro": 0, "con": 0}
    balance_line = (
        f"긍정 관점 {balance.get('pro', 0)}건과 부정 관점 {balance.get('con', 0)}건을 함께 반영했습니다."
        if web_results
        else "웹 찬반 데이터는 확보하지 못했으며, 후속 검색으로 균형 보완이 필요합니다."
    )

    return "\n\n".join([past_line, present_line, future_line, balance_line])


def market_analyst_node(state: ReportState) -> dict:
    """
    시장 분석 Agent 메인 로직

    입력: 사용자 질의, 시장 관련 문서, 웹 검색 질의
    출력: 시장 배경 요약, 핵심 포인트, 근거 자료

    실행 흐름:
    1. 시장 배경 관련 쿼리 생성
    2. RAG: pgvector에서 문서 검색 (관련도 < 0.65 → Rewrite, max 2회)
    3. Web: Tavily로 최신 시장 동향 검색 (찬/반 쌍 쿼리)
    4. 편향 비율 > 7:3 → 소수 관점 보충 검색 (Conditional Branch)
    5. 결과 통합 → 시장 배경 요약 생성
    6. State 기록 → Supervisor 반환
    """
    query = state["query"]

    rag_results: list[dict] = []
    for candidate_query in _build_market_queries(query):
        rag_results.extend(_run_rag_loop(candidate_query))
    rag_results = sorted(
        deduplicate(rag_results),
        key=lambda item: item.get("score", 0),
        reverse=True,
    )[:8]

    balanced_queries = generate_balanced_queries(query)
    web_results: list[dict] = []
    web_search_count = state.get("web_search_count", 0)
    for side, candidate_query in balanced_queries.items():
        search_results = web_search(candidate_query, max_results=4)
        if search_results:
            web_search_count += 1
        web_results.extend(
            [{**result, "query_side": side} for result in search_results]
        )

    web_results = classify_pro_con(web_results)
    if web_results and not check_bias_ratio(web_results)["is_balanced"]:
        supplemented = supplement_minority_view(query, web_results)
        if len(supplemented) > len(web_results):
            web_search_count += 1
        web_results = supplemented

    market_summary = _compose_market_summary(query, rag_results, web_results)

    return {
        "market_rag_results": rag_results,
        "market_web_results": web_results,
        "market_summary": market_summary,
        "swot_lg": None,
        "swot_catl": None,
        "strategy_diff_summary": None,
        "swot_validation": None,
        "report_draft": None,
        "final_report": None,
        "references": [],
        "summary": None,
        "section_lengths": None,
        "quality_score": None,
        "quality_checked": False,
        "web_search_count": web_search_count,
    }
