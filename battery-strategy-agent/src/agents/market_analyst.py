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

import re

from src.state import ReportState
from src.tools.rag import (
    build_market_filter,
    deduplicate,
    rewrite_query,
    search as rag_search,
)
from src.tools.web_search import (
    check_bias_ratio,
    classify_pro_con,
    search as web_search,
    supplement_minority_view,
)
from config.settings import MAX_QUERY_REWRITE

_MARKET_WEB_SIGNAL_KEYWORDS = (
    "battery market", "배터리 시장", "시장", "industry", "산업", "demand", "수요",
    "supply chain", "공급망", "policy", "정책", "regulation", "규제", "ira", "feoc",
    "price", "가격", "raw material", "원재료", "ess", "energy storage", "ev",
)
_MARKET_WEB_NOISE_KEYWORDS = (
    "annual results", "earnings", "investor confidence", "buyer", "supply agreement",
    "deal", "control technologies", "실적", "계약", "공급 계약",
)
_COMPANY_WEB_KEYWORDS = (
    "lg energy solution", "lg에너지솔루션", "catl", "tesla", "gm", "ford",
    "현대", "hyundai", "volkswagen", "byd",
)
_MARKET_QUERY_NOISE_PATTERNS = (
    r"lg에너지솔루션",
    r"\bcatl\b",
    r"\blg\b",
    r"비교\s*평가",
    r"비교\s*분석",
    r"비교해줘",
    r"평가해줘",
    r"분석해줘",
    r"어느\s*기업",
    r"유리한\s*포지션",
    r"포지션",
    r"핵심\s*경쟁력",
)


def _normalize_market_topic(query: str) -> str:
    lowered = query.lower()
    cleaned = query.strip()
    for pattern in _MARKET_QUERY_NOISE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?")

    topic_parts: list[str] = []
    if "캐즘" in query:
        topic_parts.append("전기차 캐즘")
    if "배터리" in query or "battery" in lowered or "캐즘" in query or "전기차" in query or "ev" in lowered:
        topic_parts.append("배터리 시장")
    if "ev" in lowered or "전기차" in query:
        topic_parts.append("EV 수요")
    if "ess" in lowered or "에너지저장" in query or "저장" in query:
        topic_parts.append("ESS")
    if any(keyword in lowered for keyword in ("ira", "feoc")) or "정책" in query or "규제" in query:
        topic_parts.append("정책")
    if "공급망" in query or "supply chain" in lowered:
        topic_parts.append("공급망")

    if not topic_parts:
        topic_parts.append(cleaned or "글로벌 배터리 시장")

    return " ".join(dict.fromkeys(topic_parts))


def _build_market_queries(query: str) -> list[str]:
    base_query = _normalize_market_topic(query)
    return [
        f"{base_query} 배터리 시장 환경 변화",
        f"{base_query} 배터리 산업 구조 공급망 정책 변화",
        f"{base_query} 전기차 수요 둔화 배터리 시장 대응 전략",
        f"{base_query} 배터리 시장 미래 전망 기술 트렌드 규제",
    ]


def _build_default_market_web_query(query: str) -> str:
    base_query = _normalize_market_topic(query)
    return f"{base_query} 글로벌 최신 동향 수요 둔화 ESS 정책 공급망"


def _build_balanced_market_web_queries(query: str) -> dict[str, str]:
    base_query = _normalize_market_topic(query)
    return {
        "pro": f"{base_query} 배터리 시장 회복 ESS 성장 정책 지원",
        "con": f"{base_query} 배터리 시장 수요 둔화 공급과잉 가격 경쟁",
    }


def _build_fallback_market_web_queries(query: str) -> list[tuple[str, str]]:
    base_query = _normalize_market_topic(query)
    return [
        ("neutral", f"{base_query} 글로벌 배터리 시장 전망 ESS 공급망"),
        ("neutral", "글로벌 배터리 시장 수요 둔화 ESS 정책 공급망"),
        ("pro", "global battery market ESS growth localization policy"),
        ("con", "global battery market EV demand slowdown oversupply policy"),
    ]


def _safe_rag_search(query: str) -> list[dict]:
    try:
        return rag_search(
            query,
            top_k=5,
            metadata_filter=build_market_filter(),
        )
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


def _deduplicate_web_results(results: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduplicated: list[dict] = []
    for result in results:
        key = (result.get("title", ""), result.get("url", ""))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(result)
    return deduplicated


def _market_web_priority(result: dict) -> tuple[float, int]:
    text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
    score = 0.0

    signal_hits = sum(1 for keyword in _MARKET_WEB_SIGNAL_KEYWORDS if keyword in text)
    noise_hits = sum(1 for keyword in _MARKET_WEB_NOISE_KEYWORDS if keyword in text)

    score += signal_hits * 1.2
    score -= noise_hits * 1.5

    if result.get("pro_con_tag") in {"pro", "con"}:
        score += 0.4

    snippet_length = len(result.get("snippet", ""))
    return score, snippet_length


def _rank_market_web_results(results: list[dict], limit: int = 8) -> list[dict]:
    ranked = sorted(
        _deduplicate_web_results(results),
        key=_market_web_priority,
        reverse=True,
    )
    return ranked[:limit]


def _extend_web_results(
    web_results: list[dict],
    queries: list[tuple[str, str]],
    web_search_count: int,
) -> tuple[list[dict], int]:
    collected = list(web_results)
    updated_count = web_search_count

    for side, candidate_query in queries:
        search_results = web_search(candidate_query, max_results=4)
        if search_results:
            updated_count += 1
        collected.extend(
            [{**result, "query_side": side} for result in search_results]
        )

    return collected, updated_count


def _finalize_market_web_results(
    query: str,
    market_topic: str,
    web_results: list[dict],
    web_search_count: int,
) -> tuple[list[dict], int]:
    prepared = classify_pro_con(_deduplicate_web_results(web_results))
    if prepared and not check_bias_ratio(prepared)["is_balanced"]:
        supplemented = supplement_minority_view(market_topic, prepared)
        if len(supplemented) > len(prepared):
            web_search_count += 1
        prepared = supplemented

    ranked = _rank_market_web_results(classify_pro_con(prepared))
    market_level_results = [item for item in ranked if _is_market_level_web_result(item)]
    if market_level_results:
        return market_level_results[:6], web_search_count

    fallback_queries = _build_fallback_market_web_queries(query)
    fallback_results, web_search_count = _extend_web_results(
        prepared,
        fallback_queries,
        web_search_count,
    )
    reranked = _rank_market_web_results(classify_pro_con(fallback_results))
    filtered = [item for item in reranked if _is_market_level_web_result(item)]
    if filtered:
        return filtered[:6], web_search_count

    return [], web_search_count


def _is_market_level_web_result(result: dict) -> bool:
    text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
    signal_hits = sum(1 for keyword in _MARKET_WEB_SIGNAL_KEYWORDS if keyword in text)
    noise_hits = sum(1 for keyword in _MARKET_WEB_NOISE_KEYWORDS if keyword in text)
    company_hits = sum(1 for keyword in _COMPANY_WEB_KEYWORDS if keyword in text)

    if signal_hits == 0 and company_hits:
        return False
    if noise_hits >= 2:
        return False
    if company_hits and signal_hits < 2:
        return False
    return True


def _compose_market_summary(
    query: str,
    rag_results: list[dict],
    web_results: list[dict],
) -> str:
    top_rag = rag_results[:3]
    pro_results = [item for item in web_results if item.get("pro_con_tag") == "pro"][:2]
    con_results = [item for item in web_results if item.get("pro_con_tag") == "con"][:2]
    neutral_results = [item for item in web_results if item.get("pro_con_tag") == "neutral"][:2]
    present_candidates = (pro_results + con_results)[:2] or neutral_results[:2]

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
            for item in present_candidates
        )
        if present_candidates
        else "현재: 웹 검색 결과가 없어 현재 시장 신호는 문서 기반 근거 중심으로 해석해야 합니다."
    )
    future_web_inputs = pro_results[:1] + con_results[:1]
    if len(future_web_inputs) < 2:
        future_web_inputs.extend(neutral_results[: 2 - len(future_web_inputs)])
    future_inputs = top_rag[2:3] + future_web_inputs
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
    2. RAG: 시장 분석용 문서 범위에서만 pgvector 검색 (관련도 < 0.65 → Rewrite, max 2회)
    3. Web: Tavily로 최신 시장 동향 기본 검색 1회 수행
    4. Web: 찬/반 쌍 쿼리로 균형 관점 검색
    5. 편향 비율 > 7:3 → 소수 관점 보충 검색 (Conditional Branch)
    6. 결과 통합 → 시장 배경 요약 생성
    7. State 기록 → Supervisor 반환
    """
    query = state["query"]
    market_topic = _normalize_market_topic(query)

    rag_results: list[dict] = []
    for candidate_query in _build_market_queries(query):
        rag_results.extend(_run_rag_loop(candidate_query))
    rag_results = sorted(
        deduplicate(rag_results),
        key=lambda item: item.get("score", 0),
        reverse=True,
    )[:8]

    web_results: list[dict] = []
    web_search_count = state.get("web_search_count", 0)

    initial_queries: list[tuple[str, str]] = [("neutral", _build_default_market_web_query(query))]
    initial_queries.extend(_build_balanced_market_web_queries(query).items())
    web_results, web_search_count = _extend_web_results(
        web_results,
        initial_queries,
        web_search_count,
    )
    web_results, web_search_count = _finalize_market_web_results(
        query,
        market_topic,
        web_results,
        web_search_count,
    )

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
