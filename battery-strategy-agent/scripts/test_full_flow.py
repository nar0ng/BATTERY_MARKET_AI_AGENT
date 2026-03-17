"""
배터리 전략 에이전트용 전체 플로우 스모크/회귀 테스트.

사용 예:
    python scripts/test_full_flow.py
    python scripts/test_full_flow.py --query "LG와 CATL의 ESS 전략을 비교해줘"
    python scripts/test_full_flow.py --preset regression --reindex
    python scripts/test_full_flow.py --preset regression --strict
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    DATA_DIR,
    OUTPUT_DIR,
    PGVECTOR_CONNECTION,
    PGVECTOR_TABLE,
    mask_connection_string,
)
from src.graph import graph  # noqa: E402
from src.state import create_initial_state  # noqa: E402
from src.tools.rag import (  # noqa: E402
    build_pgvector_index,
    check_pgvector_connection,
    ensure_pgvector_extension,
    load_documents,
)

PRESET_QUERIES = {
    "smoke": [
        "LG에너지솔루션과 CATL의 배터리 포트폴리오 다각화 전략을 과거, 현재, 미래 관점에서 비교 분석해줘",
    ],
    "regression": [
        "LG에너지솔루션과 CATL의 배터리 포트폴리오 다각화 전략을 과거, 현재, 미래 관점에서 비교 분석해줘",
        "CATL과 LG에너지솔루션의 ESS 사업 전략과 리스크를 비교해줘",
        "글로벌 EV 배터리 시장의 범위, 정의, 주요 경쟁 축을 설명하고 두 기업 전략을 연결해줘",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="배터리 전략 에이전트 전체 플로우 테스트를 실행합니다."
    )
    parser.add_argument(
        "--query",
        help="단일 질의를 실행합니다. 생략하면 preset 묶음을 사용합니다.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_QUERIES),
        default="smoke",
        help="--query를 생략했을 때 실행할 preset 질의 묶음입니다.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="그래프 실행 전 pgvector 인덱스를 강제로 다시 구축합니다.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="품질 게이트를 통과하지 못하면 실패로 처리합니다.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="outputs/reports/full_flow_tests 아래에 결과물을 저장하지 않습니다.",
    )
    return parser.parse_args()


def _slugify(text: str, limit: int = 80) -> str:
    slug = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    slug = re.sub(r"[-\s]+", "-", slug.strip()).strip("-").lower()
    return slug[:limit] or "flow-test"


def _collect_top_sources(final_state: dict, limit: int = 5) -> list[dict]:
    counter: Counter[str] = Counter()

    for result in final_state.get("market_rag_results", []):
        source = result.get("source")
        if source:
            counter[source] += 1

    for analysis in final_state.get("company_analyses", {}).values():
        for result in analysis.get("evidence", []):
            source = result.get("source")
            if source:
                counter[source] += 1

    return [
        {"source": source, "count": count}
        for source, count in counter.most_common(limit)
    ]


def _prepare_rag(force_reindex: bool) -> dict:
    pdf_files = sorted(Path(DATA_DIR).rglob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(
            f"{DATA_DIR} 아래에서 PDF를 찾지 못했습니다. 먼저 PDF를 하나 이상 넣어주세요."
        )

    ensure_pgvector_extension()
    diagnostics = check_pgvector_connection()
    documents = load_documents()
    build_pgvector_index(documents, force_reindex=force_reindex)

    return {
        "pdf_count": len(pdf_files),
        "chunk_count": len(documents),
        "diagnostics": diagnostics,
    }


def _run_single_query(
    query: str,
    rag_summary: dict,
    *,
    save_outputs: bool,
) -> dict:
    initial_state = create_initial_state(query)
    started_at = time.perf_counter()
    final_state = graph.invoke(initial_state)
    duration_seconds = round(time.perf_counter() - started_at, 2)

    final_report = final_state.get("final_report") or ""
    report_draft = final_state.get("report_draft") or ""
    report_text = final_report or report_draft
    quality_score = final_state.get("quality_score") or {}
    passed = bool(quality_score.get("passed"))

    result = {
        "query": query,
        "duration_seconds": duration_seconds,
        "pdf_count": rag_summary["pdf_count"],
        "chunk_count": rag_summary["chunk_count"],
        "database": rag_summary["diagnostics"],
        "market_rag_results": len(final_state.get("market_rag_results", [])),
        "market_web_results": len(final_state.get("market_web_results", [])),
        "company_analysis_count": len(final_state.get("company_analyses", {})),
        "swot_ready": bool(final_state.get("swot_lg") and final_state.get("swot_catl")),
        "report_ready": bool(report_text),
        "final_report_ready": bool(final_report),
        "quality_passed": passed,
        "quality_details": quality_score.get("details", []),
        "references_count": len(final_state.get("references", [])),
        "section_lengths": final_state.get("section_lengths") or {},
        "top_sources": _collect_top_sources(final_state),
        "output_report_path": None,
        "output_summary_path": None,
    }

    if save_outputs and report_text:
        flow_output_dir = OUTPUT_DIR / "full_flow_tests"
        flow_output_dir.mkdir(parents=True, exist_ok=True)
        file_stem = _slugify(query)

        report_path = flow_output_dir / f"{file_stem}.md"
        summary_path = flow_output_dir / f"{file_stem}.json"

        report_path.write_text(report_text, encoding="utf-8")
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result["output_report_path"] = str(report_path)
        result["output_summary_path"] = str(summary_path)

    return result


def _print_header(queries: list[str], args: argparse.Namespace) -> None:
    print("전체 플로우 테스트 실행")
    print(f"연결 문자열: {mask_connection_string(PGVECTOR_CONNECTION)}")
    print(f"pgvector 테이블: {PGVECTOR_TABLE}")
    print(f"실행 모드: {'single' if args.query else args.preset}")
    print(f"질의 수: {len(queries)}")
    print(f"강제 재인덱싱: {args.reindex}")
    print(f"엄격 모드: {args.strict}")
    print()


def _print_rag_summary(rag_summary: dict) -> None:
    print("RAG 준비가 완료되었습니다.")
    print(
        json.dumps(
            {
                "pdf_count": rag_summary["pdf_count"],
                "chunk_count": rag_summary["chunk_count"],
                "database": rag_summary["diagnostics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print()


def _print_result(index: int, total: int, result: dict) -> None:
    print(f"[{index}/{total}] {result['query']}")
    print(f"  실행 시간(초): {result['duration_seconds']}")
    print(f"  보고서 생성 여부: {result['report_ready']}")
    print(f"  최종 보고서 생성 여부: {result['final_report_ready']}")
    print(f"  품질 통과 여부: {result['quality_passed']}")
    print(f"  시장 RAG 결과 수: {result['market_rag_results']}")
    print(f"  시장 웹 결과 수: {result['market_web_results']}")
    print(f"  기업 분석 수: {result['company_analysis_count']}")
    print(f"  참고문헌 수: {result['references_count']}")

    if result["top_sources"]:
        sources = ", ".join(
            f"{item['source']}({item['count']})" for item in result["top_sources"]
        )
        print(f"  주요 출처: {sources}")

    if result["quality_details"]:
        print("  품질 상세:")
        for detail in result["quality_details"]:
            print(f"    - {detail}")

    if result["output_report_path"]:
        print(f"  보고서 경로: {result['output_report_path']}")
    if result["output_summary_path"]:
        print(f"  요약 경로: {result['output_summary_path']}")
    print()


def main() -> int:
    args = parse_args()
    queries = [args.query] if args.query else PRESET_QUERIES[args.preset]

    _print_header(queries, args)

    try:
        rag_summary = _prepare_rag(force_reindex=args.reindex)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    _print_rag_summary(rag_summary)

    exit_code = 0
    results: list[dict] = []
    for index, query in enumerate(queries, start=1):
        try:
            result = _run_single_query(
                query,
                rag_summary,
                save_outputs=not args.no_save,
            )
        except Exception as exc:
            print(f"[{index}/{len(queries)}] {query}")
            print(f"  오류: {exc}")
            print()
            exit_code = 1
            continue

        results.append(result)
        _print_result(index, len(queries), result)

        if not result["report_ready"]:
            exit_code = 1
        if args.strict and not result["quality_passed"]:
            exit_code = 1

    if results:
        overall_summary = {
            "query_count": len(queries),
            "completed": len(results),
            "all_reports_ready": all(item["report_ready"] for item in results),
            "all_quality_passed": all(item["quality_passed"] for item in results),
        }
        print("전체 요약:")
        print(json.dumps(overall_summary, ensure_ascii=False, indent=2))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
