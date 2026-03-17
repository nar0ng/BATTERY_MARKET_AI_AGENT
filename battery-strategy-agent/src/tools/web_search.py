"""
웹 검색 도구 — Tavily API
시장 분석 Agent에서만 사용 (다른 Agent는 웹 검색하지 않음)

확증 편향 대비:
  - 찬/반 쌍 쿼리 생성 필수
  - 비율 > 7:3 시 소수 관점 보충 검색
"""
from __future__ import annotations

import os
from datetime import datetime
from urllib.parse import urlparse

from tavily import TavilyClient

from config.settings import (
    BIAS_RATIO_MIN,
    BIAS_RATIO_MAX,
    BIAS_TRIGGER,
    MAX_WEB_SUPPLEMENT,
)

_PRO_KEYWORDS = {
    "growth", "expand", "opportunity", "support", "incentive", "improve",
    "growth", "positive", "recovery", "확대", "성장", "기회", "지원", "호조",
    "회복", "개선", "증가", "유리", "수혜",
}
_CON_KEYWORDS = {
    "risk", "slowdown", "decline", "oversupply", "pressure", "uncertainty",
    "regulation", "conflict", "둔화", "위험", "리스크", "과잉", "압박", "불확실",
    "규제", "침체", "감소", "하락", "분쟁", "적자",
}


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "") if parsed.netloc else "unknown"


def _deduplicate_results(results: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduplicated: list[dict] = []
    for result in results:
        key = (result.get("title", ""), result.get("url", ""))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(result)
    return deduplicated


def generate_balanced_queries(topic: str) -> dict:
    """
    확증 편향 방지를 위한 찬/반 쌍 쿼리 생성

    Args:
        topic: 검색 주제 (예: "배터리 시장 성장")

    Returns:
        {"pro": "배터리 시장 성장 전망 긍정", "con": "배터리 시장 리스크 둔화 과제"}
    """
    base_topic = topic.strip() or "배터리 시장"
    return {
        "pro": f"{base_topic} 성장 수요 확대 투자 기회",
        "con": f"{base_topic} 리스크 둔화 공급과잉 가격경쟁",
    }


def search(query: str, max_results: int = 5) -> list[dict]:
    """
    Tavily API로 웹 검색을 수행합니다.

    반환값:
        [{title, url, date, snippet}] 형태의 목록
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        client = TavilyClient(api_key=api_key)
        payload = client.search(
            query=query,
            search_depth="advanced",
            topic="news",
            max_results=max_results,
            include_raw_content=False,
        )
    except Exception:
        return []

    results: list[dict] = []
    for item in payload.get("results", []):
        url = item.get("url") or ""
        if not url:
            continue

        date = (
            item.get("published_date")
            or item.get("published_at")
            or item.get("date")
            or _today()
        )
        if len(date) >= 10:
            date = date[:10]

        title = item.get("title") or query
        snippet = item.get("content") or item.get("snippet") or ""
        publisher = item.get("site_name") or _domain(url)
        results.append(
            {
                "title": title.strip(),
                "url": url,
                "date": date,
                "snippet": snippet.strip(),
                "publisher": publisher,
            }
        )

    return _deduplicate_results(results)


def classify_pro_con(results: list[dict]) -> list[dict]:
    """
    검색 결과를 찬성/반대/중립으로 분류합니다.

    반환값:
        각 결과에 `pro_con_tag`를 추가한 목록
    """
    classified: list[dict] = []
    for result in results:
        text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        pro_score = sum(1 for keyword in _PRO_KEYWORDS if keyword in text)
        con_score = sum(1 for keyword in _CON_KEYWORDS if keyword in text)
        query_side = result.get("query_side")

        if pro_score > con_score:
            tag = "pro"
        elif con_score > pro_score:
            tag = "con"
        elif query_side in {"pro", "con"}:
            tag = query_side
        else:
            tag = "neutral"

        classified.append({**result, "pro_con_tag": tag})

    return classified


def check_bias_ratio(results: list[dict]) -> dict:
    """
    찬반 비율을 점검합니다.

    반환값:
        {"pro": int, "con": int, "neutral": int, "is_balanced": bool}
        `is_balanced`는 비율이 4:6~6:4 이내일 때 참입니다.
    """
    counts = {"pro": 0, "con": 0, "neutral": 0}
    for result in results:
        tag = result.get("pro_con_tag", "neutral")
        counts[tag if tag in counts else "neutral"] += 1

    pro_con_total = counts["pro"] + counts["con"]
    if pro_con_total == 0:
        is_balanced = True
        ratio = 0.5
    else:
        ratio = counts["pro"] / pro_con_total
        is_balanced = BIAS_RATIO_MIN <= ratio <= BIAS_RATIO_MAX

    return {
        **counts,
        "ratio": round(ratio, 4),
        "is_balanced": is_balanced,
        "needs_supplement": (
            pro_con_total > 0
            and (ratio > BIAS_TRIGGER or ratio < (1 - BIAS_TRIGGER))
        ),
    }


def supplement_minority_view(topic: str, current_results: list[dict]) -> list[dict]:
    """
    편향 비율이 7:3을 넘으면 소수 관점을 보충 검색합니다.

    제어 전략: 조건부 분기, 최대 `MAX_WEB_SUPPLEMENT`회까지 보강
    """
    results = classify_pro_con(current_results)
    bias = check_bias_ratio(results)
    if not bias["needs_supplement"]:
        return results

    minority_tag = "con" if bias["pro"] > bias["con"] else "pro"
    suffix = (
        "리스크 둔화 공급과잉 규제"
        if minority_tag == "con"
        else "회복 성장 지원 수요 확대"
    )

    combined = list(results)
    for _ in range(MAX_WEB_SUPPLEMENT):
        extra_results = search(f"{topic} {suffix}", max_results=3)
        if not extra_results:
            break
        tagged = classify_pro_con(
            [{**result, "query_side": minority_tag} for result in extra_results]
        )
        combined = _deduplicate_results(combined + tagged)
        if check_bias_ratio(combined)["is_balanced"]:
            break

    return combined
