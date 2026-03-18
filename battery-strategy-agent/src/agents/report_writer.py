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
  - 내러티브 흐름: 시장은 과거→현재→미래, 기업은 전략 축별 비교

Control Strategy:
  - Linear + Retry: 누락 섹션만 재생성 (max 1회)
"""
from __future__ import annotations

import json
import os
import re
from itertools import chain

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    REQUIRED_SECTIONS,
    SUMMARY_MAX_WORDS,
)
from src.prompts.report_prompt import (
    REPORT_SYSTEM,
    SECTION_IMPLICATIONS_TEMPLATE,
    SECTION_SUMMARY_TEMPLATE,
)
from src.state import ReportState
from src.tools.reference_formatter import format_all_references, format_reference


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _trim_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text or "")
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def _humanize_source_name(source_name: str) -> str:
    stem = re.sub(r"\.pdf$", "", source_name, flags=re.IGNORECASE)
    normalized = stem.replace("_", " ").replace("-", " ").strip()
    return normalized or source_name


def _extract_document_title_from_chunk(text: str, source_name: str) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return _humanize_source_name(source_name)

    separators = [
        " 문서 목적",
        " TOC_ID",
        " 작성 기준일",
        " 본 문서는",
        " 1.",
        " Ⅰ.",
    ]
    cutoff = len(cleaned)
    for separator in separators:
        index = cleaned.find(separator)
        if index > 0:
            cutoff = min(cutoff, index)

    title = cleaned[:cutoff].strip(" :.-")
    if len(title) < 6:
        return _humanize_source_name(source_name)
    return title


def _document_title_lookup(state: ReportState) -> dict[str, str]:
    lookup: dict[str, str] = {}
    candidates: list[dict] = []
    candidates.extend(state.get("market_rag_results", []))
    for analysis in state.get("company_analyses", {}).values():
        candidates.extend(analysis.get("evidence_pool", []))
        candidates.extend(analysis.get("evidence", []))

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            item.get("source", ""),
            int(item.get("page", 9999) or 9999),
            -float(item.get("score", 0) or 0),
        ),
    )
    for item in sorted_candidates:
        source = item.get("source")
        if not source or source in lookup:
            continue
        lookup[source] = _extract_document_title_from_chunk(item.get("chunk", ""), source)
    return lookup


def _build_reference_sources(state: ReportState) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()
    title_lookup = _document_title_lookup(state)

    for result in state.get("market_rag_results", []):
        key = f"doc:{result.get('source')}:{result.get('page')}"
        if key in seen:
            continue
        seen.add(key)
        source_name = result.get("source", "문서")
        sources.append(
            {
                "type": "report",
                "title": title_lookup.get(source_name, _humanize_source_name(source_name)),
                "source": source_name,
                "publisher": "내부 정리 자료",
                "date": "",
                "url": f"local://documents/{source_name}",
                "source_kind": "document",
            }
        )

    for analysis in state.get("company_analyses", {}).values():
        for result in analysis.get("evidence", []):
            key = f"doc:{result.get('source')}:{result.get('page')}"
            if key in seen:
                continue
            seen.add(key)
            source_name = result.get("source", "문서")
            sources.append(
                {
                    "type": "report",
                    "title": title_lookup.get(source_name, _humanize_source_name(source_name)),
                    "source": source_name,
                    "publisher": "내부 정리 자료",
                    "date": "",
                    "url": f"local://documents/{source_name}",
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


def _build_reference_lookup(sources: list[dict]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for source in sources:
        formatted = format_reference(source)
        if source.get("source_kind") == "document":
            lookup[f"doc:{source.get('source', '')}"] = formatted
        elif source.get("url"):
            lookup[f"web:{source.get('url')}"] = formatted
    return lookup


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
_MARKET_THEME_RULES = [
    (("ev", "전기차", "deployment", "사용량", "수요 구조", "gwh"), "EV 중심 수요 확대"),
    (("ess", "energy storage", "저장전지", "저장"), "ESS 수요 확대"),
    (("ira", "feoc", "eu", "규제", "탄소", "net-zero", "policy"), "정책·규제 기반 공급망 재편"),
    (("리튬", "니켈", "광물", "원재료", "가격", "supply chain", "공급망"), "원재료·공급망 변동성"),
    (("lfp", "ncm", "전고체", "기술"), "케미스트리 전환과 기술 경쟁"),
]
_MARKET_PHASE_KEYWORDS = {
    "past": ("수요 구조", "deployment", "사용량", "gwh", "+27", "+31", "전기차가 70%", "ess", "energy storage"),
    "current": ("ira", "feoc", "규제", "관세", "원재료", "광물", "가격", "경쟁", "oversupply", "둔화", "공급망"),
    "future": ("전고체", "나트륨", "차세대", "기술", "증설", "현지화", "net-zero", "규제", "ess", "리사이클"),
}


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


def _build_company_footnotes(
    company: str,
    evidence: list[dict],
    reference_lookup: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    company_code = _company_code(company)
    ref_ids: dict[str, str] = {}
    footnotes: list[str] = []

    for result in evidence:
        key = _footnote_key(result)
        if key in ref_ids:
            continue
        footnote_id = f"{company_code}-{len(ref_ids) + 1}"
        ref_ids[key] = footnote_id
        source_name = result.get("source", "문서")
        reference_text = reference_lookup.get(
            f"doc:{source_name}",
            _humanize_source_name(source_name),
        )
        footnotes.append(
            f"[^{footnote_id}]: {reference_text}, p.{result.get('page', '?')}"
        )

    return ref_ids, footnotes


def _market_result_key(result: dict) -> str:
    if result.get("url"):
        return f"web:{result.get('url')}"
    return _footnote_key(result)


def _build_market_footnotes(
    rag_results: list[dict],
    web_results: list[dict],
    reference_lookup: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    ref_ids: dict[str, str] = {}
    footnotes: list[str] = []

    for result in [*rag_results[:6], *web_results[:6]]:
        key = _market_result_key(result)
        if key in ref_ids:
            continue
        footnote_id = f"market-{len(ref_ids) + 1}"
        ref_ids[key] = footnote_id
        if result.get("url"):
            reference_text = reference_lookup.get(
                f"web:{result.get('url', '')}",
                result.get("url", "URL 없음"),
            )
            footnotes.append(f"[^{footnote_id}]: {reference_text}")
        else:
            source_name = result.get("source", "문서")
            reference_text = reference_lookup.get(
                f"doc:{source_name}",
                _humanize_source_name(source_name),
            )
            footnotes.append(
                f"[^{footnote_id}]: {reference_text}, p.{result.get('page', '?')}"
            )

    return ref_ids, footnotes


def _format_swot_footnote_source(source: str, reference_lookup: dict[str, str]) -> str:
    source = _sanitize_footnote_source(source)
    pdf_match = re.search(r"([A-Za-z0-9_-]+\.pdf)(?:,\s*p\.(\d+))?", source, flags=re.IGNORECASE)
    if pdf_match:
        source_name, page = pdf_match.groups()
        reference_text = reference_lookup.get(
            f"doc:{source_name}",
            _humanize_source_name(source_name),
        )
        return f"{reference_text}, p.{page}" if page else reference_text

    url_match = re.search(r"https?://[^\s]+", source)
    if url_match:
        url = url_match.group(0)
        return reference_lookup.get(f"web:{url}", url)

    if source in {"시장 배경 요약", "분석 요약"}:
        return "시장 배경 요약(세부 원문 근거는 본문 각주와 REFERENCE 참조)"

    return source


def _build_swot_footnotes(
    company: str,
    swot_data: dict,
    reference_lookup: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    company_code = _company_code(company)
    ref_ids: dict[str, str] = {}
    footnotes: list[str] = []

    for category in ("S", "W", "O", "T"):
        for item in swot_data.get(category, []):
            source = _format_swot_footnote_source(
                item.get("source") or item.get("evidence") or "분석 요약",
                reference_lookup,
            )
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


def _market_footnote_marks(results: list[dict], ref_ids: dict[str, str]) -> str:
    marks: list[str] = []
    for result in results:
        footnote_id = ref_ids.get(_market_result_key(result))
        if footnote_id:
            marks.append(f"[^{footnote_id}]")
    return "".join(marks)


def _swot_footnote_mark(
    category: str,
    item: dict,
    ref_ids: dict[str, str],
    reference_lookup: dict[str, str],
) -> str:
    source = _format_swot_footnote_source(
        item.get("source") or item.get("evidence") or "분석 요약",
        reference_lookup,
    )
    key = f"{category}:{item.get('factor', '')}:{source}"
    footnote_id = ref_ids.get(key)
    if not footnote_id:
        return ""
    return f"[^{footnote_id}]"


def _company_evidence_groups(evidence: list[dict]) -> dict[str, list[dict]]:
    return {
        "overview": evidence[:4],
        "strategy": evidence[:6] or evidence[:4],
        "competitiveness": evidence[2:7] or evidence[:3],
        "risk": evidence[7:10] or evidence[:3],
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


def _company_position_sentence(company: str, analysis: dict) -> str:
    strategic_position = analysis.get("strategic_position", "").strip()
    if strategic_position:
        return strategic_position

    strategy_phrases = _strategy_phrases(analysis)
    context_phrases = _company_context_phrases(company, analysis.get("evidence", []))
    focus = _join_phrases((strategy_phrases[:2] + context_phrases[:2])[:4])
    topic_particle = _topic_particle(company)
    return f"{company}{topic_particle} {focus}를 축으로 포트폴리오 다각화와 운영 안정성을 함께 확보하려는 포지션입니다."


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


def _competitiveness_bullets(company: str, analysis: dict) -> list[str]:
    bullets = list(analysis.get("core_competitiveness", []))
    if bullets:
        return bullets[:4]

    if company == "LG에너지솔루션":
        return [
            "북미·유럽 현지화와 정책 대응력",
            "ESS 및 비EV 확장 실행력",
            "차세대 제품 로드맵과 기술 개발력",
        ]

    return [
        "동력전지 대량생산과 규모의 경제",
        "저장전지 사업 확장과 시스템 통합 역량",
        "원가 경쟁력 기반의 공급망 통합력",
    ]


def _render_company_section(
    state: ReportState,
    reference_lookup: dict[str, str],
) -> tuple[str, list[str]]:
    sections: list[str] = []
    all_footnotes: list[str] = []
    for company in ("LG에너지솔루션", "CATL"):
        analysis = state.get("company_analyses", {}).get(company, {})
        evidence = analysis.get("evidence", [])
        grouped_evidence = _company_evidence_groups(evidence)
        ref_ids, footnotes = _build_company_footnotes(
            company,
            list(chain.from_iterable(grouped_evidence.values())),
            reference_lookup,
        )
        portfolio_strategy = analysis.get("portfolio_strategy", "").strip()
        if not portfolio_strategy:
            strategy_phrases = _strategy_phrases(analysis)
            portfolio_strategy = (
                f"{company}은(는) {_join_phrases(strategy_phrases[:3])}를 중심으로 포트폴리오 다각화를 추진하고 있습니다."
                if strategy_phrases
                else f"{company}은(는) 제품·지역·고객 축의 포트폴리오 다각화를 추진하고 있습니다."
            )
        strategic_position = _company_position_sentence(company, analysis)
        competitiveness_bullets = _competitiveness_bullets(company, analysis)
        risk_bullets = _risk_bullets(company, analysis)

        competitiveness_lines = [
            f"- {bullet}{_footnote_marks(grouped_evidence['competitiveness'], ref_ids)}"
            for bullet in competitiveness_bullets
        ]
        strategy_lines = [
            f"- {item}{_footnote_marks(grouped_evidence['strategy'], ref_ids)}"
            for item in analysis.get("key_strategy", [])
        ]
        risk_lines = [
            f"- {bullet}{_footnote_marks(grouped_evidence['risk'], ref_ids)}"
            for bullet in risk_bullets
        ]
        all_footnotes.extend(footnotes)
        sections.append(
            "\n".join(
                [
                    f"### {company}",
                    f"포트폴리오 다각화 전략: {portfolio_strategy}{_footnote_marks(grouped_evidence['overview'], ref_ids)}",
                    f"전략적 포지션: {strategic_position}{_footnote_marks(grouped_evidence['overview'], ref_ids)}",
                    "핵심 경쟁력:",
                    *competitiveness_lines,
                    "핵심 전략:",
                    *strategy_lines,
                    "주요 리스크:",
                    *risk_lines,
                ]
            ).strip()
        )
    return "\n\n".join(sections), all_footnotes


def _extract_market_themes(texts: list[str]) -> list[str]:
    text_blob = " ".join(texts).lower()
    themes: list[str] = []
    for keywords, label in _MARKET_THEME_RULES:
        if any(keyword in text_blob for keyword in keywords):
            themes.append(label)

    deduped: list[str] = []
    for theme in themes:
        if theme in deduped:
            continue
        deduped.append(theme)
    return deduped[:3]


def _matches_market_phase(result: dict, phase: str) -> bool:
    keywords = _MARKET_PHASE_KEYWORDS[phase]
    text = f"{result.get('chunk', '')} {result.get('snippet', '')} {result.get('title', '')}".lower()
    return any(keyword.lower() in text for keyword in keywords)


def _select_market_evidence(
    rag_results: list[dict],
    web_results: list[dict],
    phase: str,
) -> list[dict]:
    selected: list[dict] = []

    for result in rag_results:
        if _matches_market_phase(result, phase):
            selected.append(result)
        if len(selected) >= 2:
            break

    for result in web_results:
        if _matches_market_phase(result, phase):
            selected.append(result)
        if len(selected) >= 3:
            break

    if selected:
        return selected

    fallback_rag = rag_results[:2]
    fallback_web = web_results[:1]
    return [*fallback_rag, *fallback_web]


def _market_sentence_for_phase(phase: str, themes: list[str]) -> str:
    theme_set = set(themes)

    if phase == "past":
        sentences: list[str] = []
        if "EV 중심 수요 확대" in theme_set:
            sentences.append("EV 수요 확대가 배터리 시장의 외형 성장을 주도했습니다.")
        if "ESS 수요 확대" in theme_set:
            sentences.append("ESS는 보조 수요처를 넘어 별도 성장축으로 부상했습니다.")
        if "정책·규제 기반 공급망 재편" in theme_set and not sentences:
            sentences.append("주요 권역은 배터리 산업 육성을 위한 정책 기반을 마련해 왔습니다.")
        return " ".join(sentences) or "EV 수요 확대가 시장 성장의 주축을 이뤘고 ESS가 점진적으로 보조 성장축으로 자리 잡았습니다."

    if phase == "current":
        sentences: list[str] = []
        if "정책·규제 기반 공급망 재편" in theme_set:
            sentences.append("IRA·FEOC·EU 규제로 지역별 현지화 압력이 강화되고 있습니다.")
        if "원재료·공급망 변동성" in theme_set:
            sentences.append("원재료 가격 변동과 공급망 재편 부담도 동시에 커지고 있습니다.")
        if "EV 중심 수요 확대" in theme_set and len(sentences) < 2:
            sentences.append("EV 수요는 성장세를 유지하되 속도 조정 압력이 커지고 있습니다.")
        elif "ESS 수요 확대" in theme_set and len(sentences) < 2:
            sentences.append("ESS는 EV 둔화를 일부 상쇄할 수 있는 보완 수요처로 주목받고 있습니다.")
        return " ".join(sentences) or "전기차 수요 조정, 가격 경쟁, 정책 기반 현지화 압력이 동시에 작동하면서 시장 변동성이 커지고 있습니다."

    sentences: list[str] = []
    if "정책·규제 기반 공급망 재편" in theme_set:
        sentences.append("정책 기반 현지화 충족 여부가 지역별 점유율을 좌우할 가능성이 큽니다.")
    if "ESS 수요 확대" in theme_set:
        sentences.append("ESS 비중 확대는 EV 외 추가 성장 여력을 만들어낼 가능성이 큽니다.")
    if "케미스트리 전환과 기술 경쟁" in theme_set:
        sentences.append("LFP·차세대 배터리 전환 속도도 경쟁 구도를 다시 짤 핵심 변수입니다.")
    if "원재료·공급망 변동성" in theme_set and len(sentences) < 2:
        sentences.append("원재료와 공급망 안정성 확보는 수익성 방어의 핵심 변수로 남을 전망입니다.")
    return " ".join(sentences) or "정책 기반 현지화, ESS 비중 확대, 차세대 배터리 전환 속도가 향후 경쟁 구도를 좌우할 가능성이 큽니다."


def _render_market_section(
    state: ReportState,
    reference_lookup: dict[str, str],
) -> tuple[str, list[str]]:
    rag_results = state.get("market_rag_results", [])
    web_results = state.get("market_web_results", [])
    ref_ids, footnotes = _build_market_footnotes(rag_results, web_results, reference_lookup)

    past_sources = _select_market_evidence(rag_results, web_results, "past")
    current_sources = _select_market_evidence(rag_results, web_results, "current")
    future_sources = _select_market_evidence(rag_results, web_results, "future")

    past_themes = _extract_market_themes(
        [item.get("chunk", "") or item.get("snippet", "") or item.get("title", "") for item in past_sources]
    )
    current_themes = _extract_market_themes(
        [item.get("chunk", "") or item.get("snippet", "") or item.get("title", "") for item in current_sources]
    )
    future_themes = _extract_market_themes(
        [item.get("chunk", "") or item.get("snippet", "") or item.get("title", "") for item in future_sources]
    )

    past_sentence = _market_sentence_for_phase("past", past_themes)
    current_sentence = _market_sentence_for_phase("current", current_themes)
    future_sentence = _market_sentence_for_phase("future", future_themes)

    pro_count = sum(1 for item in web_results if item.get("pro_con_tag") == "pro")
    con_count = sum(1 for item in web_results if item.get("pro_con_tag") == "con")
    balance_sources = [item for item in web_results if item.get("pro_con_tag") in {"pro", "con"}][:2]

    if pro_count and con_count:
        signal_line = (
            f"시장 신호: 웹 검색에서는 회복·성장 관점 {pro_count}건과 둔화·리스크 관점 {con_count}건이 병존합니다."
            f"{_market_footnote_marks(balance_sources, ref_ids)}"
        )
    elif pro_count and not con_count:
        signal_line = (
            f"시장 신호: 최근 웹 검색은 회복·성장 관점 기사 비중이 높아 단기 낙관에 치우칠 수 있으므로, 정책·원가 리스크는 문서 근거와 함께 해석할 필요가 있습니다."
            f"{_market_footnote_marks(balance_sources, ref_ids)}"
        )
    elif con_count and not pro_count:
        signal_line = (
            f"시장 신호: 최근 웹 검색은 둔화·리스크 관점 기사 비중이 높아 보수적으로 읽힐 수 있으므로, 수요 회복 가능성도 함께 점검할 필요가 있습니다."
            f"{_market_footnote_marks(balance_sources, ref_ids)}"
        )
    else:
        signal_line = "시장 신호: 최신 웹 기사 확보가 제한적이어서 시장 문서 근거 중심으로 해석했습니다."

    section = "\n".join(
        [
            f"과거: {past_sentence}{_market_footnote_marks(past_sources, ref_ids)}",
            f"현재: {current_sentence}{_market_footnote_marks(current_sources, ref_ids)}",
            f"미래: {future_sentence}{_market_footnote_marks(future_sources, ref_ids)}",
            signal_line,
        ]
    ).strip()

    return section, footnotes


def _render_swot_items(
    category: str,
    items: list[dict],
    ref_ids: dict[str, str],
    reference_lookup: dict[str, str],
) -> str:
    return "\n".join(
        f"- {item['factor']}{_swot_footnote_mark(category, item, ref_ids, reference_lookup)}"
        for item in items
    )


def _render_swot_table_cell(
    category: str,
    items: list[dict],
    ref_ids: dict[str, str],
    reference_lookup: dict[str, str],
) -> str:
    if not items:
        return "-"
    return "<br/>".join(
        f"• {item['factor']}{_swot_footnote_mark(category, item, ref_ids, reference_lookup)}"
        for item in items
    )


def _render_company_swot_table(
    swot_data: dict,
    ref_ids: dict[str, str],
    reference_lookup: dict[str, str],
) -> str:
    return "\n".join(
        [
            "| Strength | Weakness |",
            "|---|---|",
            f"| {_render_swot_table_cell('S', swot_data.get('S', []), ref_ids, reference_lookup)} | {_render_swot_table_cell('W', swot_data.get('W', []), ref_ids, reference_lookup)} |",
            "| Opportunity | Threat |",
            "|---|---|",
            f"| {_render_swot_table_cell('O', swot_data.get('O', []), ref_ids, reference_lookup)} | {_render_swot_table_cell('T', swot_data.get('T', []), ref_ids, reference_lookup)} |",
        ]
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


def _render_swot_section(
    state: ReportState,
    reference_lookup: dict[str, str],
) -> tuple[str, list[str]]:
    swot_lg = state.get("swot_lg") or {"S": [], "W": [], "O": [], "T": []}
    swot_catl = state.get("swot_catl") or {"S": [], "W": [], "O": [], "T": []}
    lg_ref_ids, lg_footnotes = _build_swot_footnotes("LG에너지솔루션", swot_lg, reference_lookup)
    catl_ref_ids, catl_footnotes = _build_swot_footnotes("CATL", swot_catl, reference_lookup)

    section = "\n\n".join(
        [
            "### 핵심 전략 비교",
            _render_comparison_table(state.get("comparison_data")),
            "### LG에너지솔루션 SWOT",
            _render_company_swot_table(swot_lg, lg_ref_ids, reference_lookup),
            "### CATL SWOT",
            _render_company_swot_table(swot_catl, catl_ref_ids, reference_lookup),
            "### 전략 차이 요약",
            state.get("strategy_diff_summary", "전략 차이 요약 확보 필요"),
        ]
    )
    return section, [*lg_footnotes, *catl_footnotes]


def _top_factors(swot_data: dict, category: str, limit: int = 2) -> list[str]:
    return [
        item.get("factor", "")
        for item in swot_data.get(category, [])[:limit]
        if item.get("factor")
    ]


def _monitoring_points(state: ReportState) -> list[str]:
    rag_results = state.get("market_rag_results", [])
    web_results = state.get("market_web_results", [])

    current_themes = _extract_market_themes(
        [
            item.get("chunk", "") or item.get("snippet", "") or item.get("title", "")
            for item in _select_market_evidence(rag_results, web_results, "current")
        ]
    )
    future_themes = _extract_market_themes(
        [
            item.get("chunk", "") or item.get("snippet", "") or item.get("title", "")
            for item in _select_market_evidence(rag_results, web_results, "future")
        ]
    )

    points: list[str] = []
    if "정책·규제 기반 공급망 재편" in current_themes or "정책·규제 기반 공급망 재편" in future_themes:
        points.append("북미·유럽 현지화와 정책 인센티브 지속성")
    if "ESS 수요 확대" in current_themes or "ESS 수요 확대" in future_themes:
        points.append("ESS 수주가 실제 매출과 납품 확대로 이어지는 속도")
    if "원재료·공급망 변동성" in current_themes or "원재료·공급망 변동성" in future_themes:
        points.append("원재료 가격 하락 국면에서의 원가 전가력과 마진 방어")
    if "케미스트리 전환과 기술 경쟁" in future_themes:
        points.append("LFP·차세대 배터리 전환 속도")

    deduped: list[str] = []
    for point in points:
        if point in deduped:
            continue
        deduped.append(point)
    return deduped[:4]


def _render_implications_fallback(state: ReportState) -> str:
    comparison_verdict = (state.get("comparison_data") or {}).get(
        "verdict",
        "현재 비교 데이터가 제한적이므로 보수적 해석이 필요합니다.",
    )
    company_analyses = state.get("company_analyses", {})
    lg_analysis = company_analyses.get("LG에너지솔루션", {})
    catl_analysis = company_analyses.get("CATL", {})
    swot_lg = state.get("swot_lg") or {"S": [], "W": [], "O": [], "T": []}
    swot_catl = state.get("swot_catl") or {"S": [], "W": [], "O": [], "T": []}

    lg_comp = lg_analysis.get("core_competitiveness", [])[:2]
    catl_comp = catl_analysis.get("core_competitiveness", [])[:2]
    lg_comp_text = _join_phrases(lg_comp) if lg_comp else "정책 대응력과 고객 포트폴리오 분산"
    catl_comp_text = _join_phrases(catl_comp) if catl_comp else "규모·원가 경쟁력과 저장전지 확장"

    lg_weaknesses = _top_factors(swot_lg, "W")
    catl_threats = _top_factors(swot_catl, "T")
    lg_weakness_text = _join_phrases(lg_weaknesses) if lg_weaknesses else "수익성 변동성"
    catl_threat_text = _join_phrases(catl_threats) if catl_threats else "정책·규제와 가격 경쟁 심화"

    monitoring_points = _monitoring_points(state)
    monitoring_text = _join_phrases(monitoring_points) if monitoring_points else "정책 지원 지속성, 가격 경쟁 압력 및 차세대 케미스트리 전환 속도"

    return "\n".join(
        [
            f"1. EV 캐즘이 예상보다 길어질수록 단기 방어력은 {catl_comp_text} 중심의 CATL 쪽이 상대적으로 유리할 가능성이 큽니다.",
            f"2. 반대로 북미·유럽 중심의 정책 수혜, 고객 분산, 현지화 대응력이 더 중요해질수록 {lg_comp_text} 중심의 LG에너지솔루션 전략적 위치가 부각될 수 있습니다.",
            f"3. 따라서 의사결정자는 절대적 승자 판단보다 시나리오별 우위를 봐야 하며, 우선 점검할 지표는 {monitoring_text}입니다.",
            f"4. 실행 리스크 측면에서는 LG에너지솔루션의 {lg_weakness_text}, CATL의 {catl_threat_text} 대응력이 실제 경쟁 우위를 가르는 분기점이 될 가능성이 큽니다.",
            f"5. {comparison_verdict}",
        ]
    )


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


def _build_implications_payload(state: ReportState) -> dict:
    company_analyses = state.get("company_analyses", {})
    return {
        "comparison_verdict": (state.get("comparison_data") or {}).get("verdict", ""),
        "market_summary": state.get("market_summary", ""),
        "strategy_diff_summary": state.get("strategy_diff_summary", ""),
        "companies": {
            company: {
                "portfolio_strategy": analysis.get("portfolio_strategy", ""),
                "strategic_position": analysis.get("strategic_position", ""),
                "core_competitiveness": analysis.get("core_competitiveness", []),
                "key_strategy": analysis.get("key_strategy", []),
                "risk_factors": analysis.get("risk_factors", []),
            }
            for company, analysis in company_analyses.items()
        },
        "swot": {
            "LG에너지솔루션": {
                key: _top_factors(state.get("swot_lg") or {"S": [], "W": [], "O": [], "T": []}, key)
                for key in ("S", "W", "O", "T")
            },
            "CATL": {
                key: _top_factors(state.get("swot_catl") or {"S": [], "W": [], "O": [], "T": []}, key)
                for key in ("S", "W", "O", "T")
            },
        },
        "monitoring_points": _monitoring_points(state),
    }


def _render_implications(state: ReportState) -> tuple[str, int]:
    fallback_text = _render_implications_fallback(state)
    if not os.getenv("OPENAI_API_KEY"):
        return fallback_text, 0

    prompt = SECTION_IMPLICATIONS_TEMPLATE.format(
        market_summary=state.get("market_summary", ""),
        company_summary=json.dumps(
            _build_implications_payload(state).get("companies", {}),
            ensure_ascii=False,
            indent=2,
        ),
        swot_summary=json.dumps(
            {
                "comparison_verdict": (state.get("comparison_data") or {}).get("verdict", ""),
                "strategy_diff_summary": state.get("strategy_diff_summary", ""),
                "swot": _build_implications_payload(state).get("swot", {}),
                "monitoring_points": _build_implications_payload(state).get("monitoring_points", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
    ) + """

추가 요구사항:
- 4~5개의 번호 목록으로만 작성할 것
- 누가 '절대적으로 더 낫다'고 단정하지 말고, 어떤 시나리오에서 어느 기업이 상대적으로 유리한지 구분할 것
- 의사결정자가 실제로 점검해야 할 지표 2~4개를 구체적으로 포함할 것
- 단순 반복이나 템플릿 문장을 피하고, 자연스러운 한국어 전략 보고서 문체로 쓸 것
- heading 없이 본문만 반환할 것
"""

    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        response = llm.invoke(
            [
                SystemMessage(content=REPORT_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        text = _message_content_to_text(response.content)
        if text:
            return text, 1
    except Exception:
        return fallback_text, 0

    return fallback_text, 0


def _render_summary_fallback(state: ReportState) -> str:
    return _trim_words(
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


def _render_summary(state: ReportState, report_body: str) -> tuple[str, int]:
    fallback_text = _render_summary_fallback(state)
    if not os.getenv("OPENAI_API_KEY"):
        return fallback_text, 0

    prompt = SECTION_SUMMARY_TEMPLATE.format(report_body=report_body) + f"""

추가 요구사항:
- {SUMMARY_MAX_WORDS}단어 이내를 반드시 지킬 것
- 단순 섹션 나열이 아니라, 시장 맥락, 기업별 상대적 포지션 차이, 핵심 판단 포인트를 한 번에 이해할 수 있게 쓸 것
- "누가 절대적으로 우위"라고 단정하지 말고 어떤 조건에서 어느 기업이 더 유리한지 드러낼 것
- heading 없이 SUMMARY 본문만 반환할 것
"""

    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        response = llm.invoke(
            [
                SystemMessage(content=REPORT_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        text = _trim_words(_message_content_to_text(response.content), SUMMARY_MAX_WORDS)
        if text:
            return text, 1
    except Exception:
        return fallback_text, 0

    return fallback_text, 0


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
    reference_sources = _build_reference_sources(state)
    reference_lookup = _build_reference_lookup(reference_sources)

    market_section, market_footnotes = _render_market_section(state, reference_lookup)
    company_section, company_footnotes = _render_company_section(state, reference_lookup)
    swot_section, swot_footnotes = _render_swot_section(state, reference_lookup)
    implications_section, implication_llm_calls = _render_implications(state)

    reference_groups = format_all_references(reference_sources)
    reference_section, flattened_references = _render_reference_section(reference_groups)
    all_footnotes = [*market_footnotes, *company_footnotes, *swot_footnotes]
    if all_footnotes:
        footnote_block = "\n".join(all_footnotes)
        reference_section = f"{reference_section}\n\n## 각주\n{footnote_block}".strip()

    body_section_map = {
        "시장 배경": market_section,
        "기업별 포트폴리오 다각화 전략 및 핵심 경쟁력": company_section,
        "핵심 전략 비교 및 SWOT 분석": swot_section,
        "종합 시사점": implications_section,
        "REFERENCE": reference_section or "참고문헌 데이터 없음",
    }
    report_body_for_summary = "\n\n".join(
        f"# {heading}\n{body_section_map[heading]}".strip()
        for heading in REQUIRED_SECTIONS
        if heading != "SUMMARY"
    ).strip()
    summary_text, summary_llm_calls = _render_summary(state, report_body_for_summary)

    section_map = {
        "SUMMARY": summary_text,
        **body_section_map,
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
        "llm_call_count": state.get("llm_call_count", 0) + implication_llm_calls + summary_llm_calls,
    }
