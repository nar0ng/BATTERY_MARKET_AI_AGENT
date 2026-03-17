"""
pgvector 기반 RAG 파이프라인용 간단한 스모크 테스트.

사용 예:
    python scripts/test_pgvector_rag.py
    python scripts/test_pgvector_rag.py --query "CATL battery strategy"
    python scripts/test_pgvector_rag.py --reindex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_DIR,
    PGVECTOR_CONNECTION,
    PGVECTOR_TABLE,
    mask_connection_string,
)
from src.tools.rag import (
    build_pgvector_index,
    check_pgvector_connection,
    ensure_pgvector_extension,
    load_documents,
    search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pgvector RAG 구성을 점검합니다.")
    parser.add_argument(
        "--query",
        default="LG에너지솔루션과 CATL의 배터리 포트폴리오 전략",
        help="인덱싱 후 실행할 유사도 검색 질의입니다.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="pgvector 테이블을 비우고 다시 구축합니다.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pdf_files = sorted(Path(DATA_DIR).rglob("*.pdf"))
    if not pdf_files:
        print("data/documents 아래에서 PDF를 찾지 못했습니다. 먼저 PDF를 하나 이상 넣어주세요.")
        print(f"확인한 경로: {DATA_DIR}")
        return 1

    print(f"사용할 pgvector 테이블: {PGVECTOR_TABLE}")
    print(f"연결 문자열: {mask_connection_string(PGVECTOR_CONNECTION)}")
    print(f"PDF 개수: {len(pdf_files)}")

    try:
        ensure_pgvector_extension()
        diagnostics = check_pgvector_connection()
        print("데이터베이스 진단:")
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

        documents = load_documents()
        print(f"불러온 청크 수: {len(documents)}")
        if documents:
            print("첫 번째 청크 미리보기:")
            print(documents[0]["text"][:200])

        build_pgvector_index(documents, force_reindex=args.reindex)
        print("인덱스 구축이 완료되었습니다.")

        results = search(args.query, top_k=5)
        print(f"검색 결과 질의: {args.query}")
        print(json.dumps(results[:3], ensure_ascii=False, indent=2))
    except RuntimeError as exc:
        print(str(exc))
        return 1

    if not results:
        print("관련도 기준을 통과한 결과가 없습니다.")
        return 1

    print("pgvector RAG 스모크 테스트를 통과했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
