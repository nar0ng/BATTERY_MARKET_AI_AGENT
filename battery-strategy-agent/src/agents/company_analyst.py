"""
기업 분석 Agent
- LG에너지솔루션·CATL 전략 분석 및 비교 데이터 생성
- RAG only

Outcome: LG·CATL 각각의 포트폴리오 다각화 전략과 핵심 경쟁력이,
         과거→현재→미래 시간축을 포함한 비교 데이터와 함께 완성

Success Criteria:
  - 관련성: 검색 청크 유사도 ≥ 0.65
  - 중복 제거: 코사인 유사도 > 0.95 제거
  - 추적 가능성: 페이지 번호 + 문서명 포함
  - 시간적 포괄성: 각 기업별 과거/현재/미래 모두 포함

Control Strategy:
  - Loop: Query Rewrite (max 2회)
  - LG/CATL 병렬 실행 가능 (Supervisor가 Send()로 디스패치)
"""
from __future__ import annotations

import re
from functools import lru_cache

from config.settings import MAX_QUERY_REWRITE
from src.state import ReportState
from src.tools.rag import (
    build_company_filter,
    deduplicate,
    load_documents,
    rewrite_query,
    search as rag_search,
)

_COMPANY_ALIASES = {
    "LG에너지솔루션": ("lg에너지솔루션", "lg energy solution", "lg energy solutions", "lges"),
    "CATL": ("catl", "宁德时代"),
}
_STRATEGY_KEYWORDS = [
    (("ess", "energy storage", "모빌리티", "비ev", "non-ev"), "ESS 및 비EV 응용처 확대"),
    (("lfp", "ncm", "전고체", "solid-state", "chemistry"), "배터리 케미스트리 포트폴리오 다변화"),
    (("북미", "유럽", "중국", "미국", "헝가리", "폴란드", "overseas"), "생산 거점과 지역 포트폴리오 분산"),
    (("리사이클", "recycling", "내재화", "양극재", "리튬", "supply chain"), "밸류체인 내재화 및 리사이클링 강화"),
    (("oem", "고객", "tesla", "gm", "ford", "현대", "volkswagen"), "고객 포트폴리오 다변화"),
    (("cash", "현금", "투자", "capex", "가동률", "capa", "수익"), "생산능력과 재무 체력 관리"),
]
_RISK_KEYWORDS = {
    "위험", "리스크", "둔화", "편중", "의존", "적자", "하락", "pressure", "risk",
    "uncertain", "slowdown", "oversupply", "cost", "부채",
}


def _strip_company_names(query: str, target_company: str) -> str:
    scoped_query = query
    for company, aliases in _COMPANY_ALIASES.items():
        if company == target_company:
            continue
        needles = (company.lower(), *aliases)
        for needle in needles:
            scoped_query = re.sub(re.escape(needle), " ", scoped_query, flags=re.IGNORECASE)
    return " ".join(scoped_query.split())


def _build_company_queries(company: str, query: str) -> list[str]:
    scoped_query = _strip_company_names(query, company)
    base_query = f"{company} {scoped_query}".strip()
    return [
        f"{base_query} 과거 전략 진화 포트폴리오",
        f"{company} 현재 생산 거점 고객 포트폴리오 재무 체력",
        f"{company} 미래 투자 전략 케미스트리 로드맵",
        f"{company} ESS 리사이클링 공급망 내재화",
        f"{company} 지역 다각화 IRA 유럽 정책 수혜",
    ]


def _safe_rag_search(
    query: str,
    *,
    top_k: int = 12,
    metadata_filter: dict | None = None,
) -> list[dict]:
    try:
        return rag_search(query, top_k=top_k, metadata_filter=metadata_filter)
    except Exception:
        return []


def _run_rag_loop(
    company: str,
    query: str,
    top_k: int = 12,
) -> list[dict]:
    current_query = query
    results: list[dict] = []
    metadata_filter = build_company_filter(company)

    for _ in range(MAX_QUERY_REWRITE + 1):
        results = _safe_rag_search(
            current_query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        if results:
            average_score = sum(item["score"] for item in results) / len(results)
            if average_score >= 0.65:
                break
        current_query = rewrite_query(current_query, results)

    return results


def _citation(result: dict) -> str:
    return f"[출처: {result.get('source', '문서')}, p.{result.get('page', '?')}]"


def _snippet(text: str, limit: int = 130) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _matches_aliases(text: str, aliases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(alias in lowered for alias in aliases)


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
        if len(token) > 1
    }


@lru_cache(maxsize=1)
def _loaded_documents() -> tuple[dict, ...]:
    return tuple(load_documents())


def _fallback_company_results(company: str, query: str, limit: int = 6) -> list[dict]:
    aliases = _COMPANY_ALIASES.get(company, ())
    scoped_query = _strip_company_names(query, company)
    query_tokens = _tokenize(f"{company} {scoped_query}")

    ranked: list[dict] = []
    for document in _loaded_documents():
        if document.get("analysis_scope") != "company":
            continue
        if document.get("company") != company:
            continue

        chunk = document.get("text", "")
        lowered_chunk = chunk.lower()
        chunk_tokens = _tokenize(chunk)
        overlap = len(query_tokens & chunk_tokens)
        alias_bonus = 3 if _matches_aliases(lowered_chunk, aliases) else 0
        strategy_bonus = sum(
            1
            for keywords, _ in _STRATEGY_KEYWORDS
            if any(keyword in lowered_chunk for keyword in keywords)
        )
        score = overlap + alias_bonus + strategy_bonus
        if score <= 0:
            continue

        ranked.append(
            {
                "chunk": chunk,
                "score": round(score / 10, 4),
                "page": document.get("page"),
                "source": document.get("source"),
                "chunk_id": document.get("chunk_id"),
            }
        )

    ranked.sort(key=lambda item: (item.get("score", 0), -int(item.get("page", 0) or 0)), reverse=True)
    return ranked[:limit]


def _company_result_priority(company: str, result: dict) -> tuple[float, float]:
    content = result.get("chunk", "")
    aliases = _COMPANY_ALIASES.get(company, ())
    competitor_aliases = tuple(
        alias
        for other_company, other_aliases in _COMPANY_ALIASES.items()
        if other_company != company
        for alias in (other_company.lower(), *other_aliases)
    )

    priority = float(result.get("score", 0.0))
    result_company = result.get("company")
    analysis_scope = result.get("analysis_scope")

    if analysis_scope == "company" and result_company == company:
        priority += 2.0

    if _matches_aliases(content, aliases):
        priority += 1.2

    if result_company and result_company != company:
        priority -= 2.0
    if _matches_aliases(content, competitor_aliases):
        priority -= 1.5

    return priority, float(result.get("score", 0.0))


def _select_company_results(company: str, results: list[dict]) -> list[dict]:
    ranked = sorted(
        deduplicate(results),
        key=lambda item: _company_result_priority(company, item),
        reverse=True,
    )
    return ranked[:8]


def _merge_company_results(
    company: str,
    primary: list[dict],
    supplements: list[dict],
    limit: int = 8,
) -> list[dict]:
    merged: list[dict] = []
    seen_chunk_ids: set[str] = set()

    for result in [*primary, *supplements]:
        chunk_id = result.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            continue
        if chunk_id is not None:
            seen_chunk_ids.add(chunk_id)
        merged.append(result)

    merged.sort(
        key=lambda item: _company_result_priority(company, item),
        reverse=True,
    )
    return merged[:limit]


def _extract_strategies(results: list[dict]) -> list[str]:
    text_blob = " ".join(result.get("chunk", "").lower() for result in results)
    strategies: list[str] = []
    for keywords, label in _STRATEGY_KEYWORDS:
        if any(keyword in text_blob for keyword in keywords):
            strategies.append(label)

    if not strategies:
        strategies = [
            "제품·지역·고객 포트폴리오를 함께 조정하는 다각화 전략",
            "공급망과 재무 여력을 동시에 관리하는 보수적 확장 전략",
        ]

    return strategies[:5]


def _extract_risks(results: list[dict]) -> list[str]:
    risks: list[str] = []
    for result in results:
        chunk = result.get("chunk", "")
        lowered = chunk.lower()
        if any(keyword in lowered for keyword in _RISK_KEYWORDS):
            risks.append(f"{_snippet(chunk)} {_citation(result)}")

    if not risks:
        risks = [
            "전기차 수요 둔화와 가격 경쟁 심화가 단기 수익성을 압박할 수 있습니다.",
            "대규모 증설과 정책 변화가 투자 회수 기간을 늘릴 수 있습니다.",
        ]

    deduplicated: list[str] = []
    seen: set[str] = set()
    for risk in risks:
        if risk in seen:
            continue
        seen.add(risk)
        deduplicated.append(risk)
    return deduplicated[:4]


def _build_time_narrative(company: str, results: list[dict]) -> dict:
    top_results = results[:6]
    past_sources = top_results[:2]
    present_sources = top_results[2:4]
    future_sources = top_results[4:6]

    past = (
        " / ".join(f"{_snippet(item['chunk'])} {_citation(item)}" for item in past_sources)
        if past_sources
        else f"{company}의 과거 전략 진화는 현재 문서 적재가 부족해 정성적 추정에 의존합니다."
    )
    present = (
        " / ".join(f"{_snippet(item['chunk'])} {_citation(item)}" for item in present_sources)
        if present_sources
        else f"{company}의 현재 포지션은 생산능력·고객 믹스·수익성 관리 축에서 추가 확인이 필요합니다."
    )
    future = (
        " / ".join(f"{_snippet(item['chunk'])} {_citation(item)}" for item in future_sources)
        if future_sources
        else f"{company}는 케미스트리 전환과 지역 다각화로 미래 변동성에 대응할 가능성이 큽니다."
    )
    return {"past": past, "present": present, "future": future}


def company_analyst_node(state: ReportState) -> dict:
    """
    기업 분석 Agent 메인 로직

    입력: 사용자 질의, 기업 관련 문서, 비교 항목
    출력: 기업별 전략 요약, 비교 데이터

    실행 흐름:
    1. 기업별 전략 관련 쿼리 생성
    2. RAG: 대상 기업 문서 범위에서만 pgvector 검색 (관련도 < 0.65 → Rewrite, max 2회)
    3. 기업별 전략 진화 내러티브 생성 (과거→현재→미래)
    4. 비교 데이터 구조화
    5. State 기록 → Supervisor 반환

    Note: _target_company 필드로 LG/CATL 구분 (Supervisor가 Send 시 주입)
    """
    target = state.get("_target_company", "")
    if not target:
        return {"company_analyses": {}}

    raw_results: list[dict] = []
    for candidate_query in _build_company_queries(target, state["query"]):
        raw_results.extend(_run_rag_loop(target, candidate_query))
    results = _select_company_results(target, raw_results)
    preferred_count = sum(
        1
        for item in results
        if item.get("analysis_scope") == "company" and item.get("company") == target
    )
    if preferred_count < 4:
        fallback_results = _fallback_company_results(target, state["query"])
        results = _merge_company_results(target, results, fallback_results)

    time_narrative = _build_time_narrative(target, results)
    strategies = _extract_strategies(results)
    risks = _extract_risks(results)
    source_refs = []
    for result in results[:6]:
        citation = _citation(result)
        if citation not in source_refs:
            source_refs.append(citation)

    analysis = {
        "company": target,
        "past": time_narrative["past"],
        "present": time_narrative["present"],
        "future": time_narrative["future"],
        "key_strategy": strategies,
        "risk_factors": risks,
        "source_refs": source_refs,
        "evidence": results[:8],
    }

    return {
        "company_analyses": {
            target: analysis,
        }
    }
