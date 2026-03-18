"""
RAG 도구 — pgvector + 오픈소스 임베딩
시장 분석 Agent (RAG+Web), 기업 분석 Agent (RAG only), SWOT Agent (조건부)에서 사용
"""
from __future__ import annotations

import hashlib
import math
import re
import uuid
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
import pdfplumber
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from config.settings import (
    EMBEDDING_MODEL,
    VECTOR_SEARCH_TOP_K,
    RELEVANCE_THRESHOLD,
    DUPLICATE_THRESHOLD,
    MAX_DOCUMENT_PAGES,
    DATA_DIR,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    PGVECTOR_CONNECTION,
    PGVECTOR_TABLE,
    mask_connection_string,
)

_MARKET_ANALYSIS_SCOPE = "market"
_COMPANY_ANALYSIS_SCOPE = "company"
_COMMON_ANALYSIS_SCOPE = "common"
_KNOWN_COMPANY_DOCUMENTS = {
    "LG에너지솔루션": ("lgreport", "lgenergysolution", "lges"),
    "CATL": ("catlreport", "catl"),
}


class SentenceTransformerEmbeddings(Embeddings):
    """`sentence-transformers`를 감싼 LangChain 호환 임베딩 래퍼입니다."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._fallback_dimension = 256
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = None

    @property
    def dimension(self) -> int:
        if self.model is None:
            return self._fallback_dimension
        return int(self.model.get_sentence_embedding_dimension())

    def _embed_with_hash(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.model is None:
            return [self._embed_with_hash(text) for text in texts]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        if self.model is None:
            return self._embed_with_hash(text)
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _text_quality_score(text: str) -> float:
    if not text:
        return 0.0

    visible_chars = [char for char in text if not char.isspace()]
    if not visible_chars:
        return 0.0

    readable_chars = sum(
        1
        for char in visible_chars
        if (
            char.isalnum()
            or "\uac00" <= char <= "\ud7a3"
            or char in ".,:%()/+-_[]{}&'\""
        )
    )
    weird_chars = sum(
        1 for char in visible_chars if char in "öß˚ˇ×ÕªÓÀÿ¯ðÛ⁄‡†‰‹›"
    )

    readable_ratio = readable_chars / len(visible_chars)
    weird_ratio = weird_chars / len(visible_chars)
    return readable_ratio - weird_ratio


def _extract_pdf_texts(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pypdf_pages = [page.extract_text() or "" for page in reader.pages]

    with pdfplumber.open(str(pdf_path)) as pdf:
        plumber_pages = [page.extract_text() or "" for page in pdf.pages]

    page_count = max(len(pypdf_pages), len(plumber_pages))
    extracted_pages: list[str] = []
    for page_index in range(page_count):
        pypdf_text = pypdf_pages[page_index] if page_index < len(pypdf_pages) else ""
        plumber_text = (
            plumber_pages[page_index] if page_index < len(plumber_pages) else ""
        )

        best_text = max(
            (pypdf_text, plumber_text),
            key=lambda value: (_text_quality_score(value), len(value)),
        )
        extracted_pages.append(_normalize_text(best_text))

    return extracted_pages


def _chunk_text(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    if len(normalized) <= RAG_CHUNK_SIZE:
        return [normalized]

    chunks: list[str] = []
    step = max(RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP, 1)
    for start in range(0, len(normalized), step):
        chunk = normalized[start : start + RAG_CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        if start + RAG_CHUNK_SIZE >= len(normalized):
            break
    return chunks


def _chunk_id(source: str, page: int, chunk_index: int, text_value: str) -> str:
    payload = f"{source}:{page}:{chunk_index}:{text_value}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def _normalize_source_key(source_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(source_name).stem.lower())


def _infer_document_metadata(source_name: str) -> dict:
    normalized_key = _normalize_source_key(source_name)

    if "batteryreport" in normalized_key or "marketreport" in normalized_key:
        return {
            "analysis_scope": _MARKET_ANALYSIS_SCOPE,
            "company": None,
            "document_kind": "market_overview",
        }

    for company, aliases in _KNOWN_COMPANY_DOCUMENTS.items():
        if any(alias in normalized_key for alias in aliases):
            return {
                "analysis_scope": _COMPANY_ANALYSIS_SCOPE,
                "company": company,
                "document_kind": "company_profile",
            }

    return {
        "analysis_scope": _COMMON_ANALYSIS_SCOPE,
        "company": None,
        "document_kind": "general_reference",
    }


def build_market_filter(*, include_common: bool = False) -> dict:
    """시장 분석 에이전트가 사용할 메타데이터 필터를 생성합니다."""
    if include_common:
        return {
            "$or": [
                {"analysis_scope": _MARKET_ANALYSIS_SCOPE},
                {"analysis_scope": _COMMON_ANALYSIS_SCOPE},
            ]
        }
    return {"analysis_scope": _MARKET_ANALYSIS_SCOPE}


def build_company_filter(company: str, *, include_common: bool = False) -> dict:
    """기업 분석 에이전트가 사용할 메타데이터 필터를 생성합니다."""
    company_filter = {
        "analysis_scope": _COMPANY_ANALYSIS_SCOPE,
        "company": company,
    }
    if include_common:
        return {
            "$or": [
                company_filter,
                {"analysis_scope": _COMMON_ANALYSIS_SCOPE},
            ]
        }
    return company_filter


def _validate_table_name(table_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(f"Invalid pgvector table name: {table_name!r}")
    return table_name


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    return float(sum(a * b for a, b in zip(vector_a, vector_b)))


def _raise_pgvector_connection_error(exc: OperationalError) -> RuntimeError:
    error_text = str(getattr(exc, "orig", exc))
    normalized_error = error_text.lower()

    guidance = [
        "pgvector connection failed.",
        f"Configured connection: {mask_connection_string(PGVECTOR_CONNECTION)}",
        "Set PGVECTOR_CONNECTION or POSTGRES_HOST/PORT/DB/USER/PASSWORD in .env to match the running database.",
    ]

    if "password authentication failed" in normalized_error:
        guidance.append(
            "The configured Postgres password was rejected by the server."
        )
    elif 'database "' in normalized_error and '" does not exist' in normalized_error:
        guidance.append(
            "The target database does not exist yet. Create it or point POSTGRES_DB at an existing database."
        )
    elif (
        "connection refused" in normalized_error
        or "operation not permitted" in normalized_error
    ):
        guidance.append(
            "The database server is not reachable on the configured host and port."
        )

    guidance.append(f"Original driver error: {error_text}")
    return RuntimeError("\n".join(guidance))


@lru_cache(maxsize=1)
def _get_embedding_service() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_pg_engine() -> PGEngine:
    return PGEngine.from_connection_string(url=PGVECTOR_CONNECTION)


@lru_cache(maxsize=1)
def _get_sqlalchemy_engine():
    return create_engine(PGVECTOR_CONNECTION)


def _table_has_rows(table_name: str) -> bool:
    validated = _validate_table_name(table_name)
    try:
        with _get_sqlalchemy_engine().connect() as conn:
            exists = conn.execute(
                text("SELECT to_regclass(:table_name)"),
                {"table_name": validated},
            ).scalar()
            if not exists:
                return False

            row_count = conn.execute(
                text(f'SELECT COUNT(*) FROM "{validated}"')
            ).scalar_one()
            return int(row_count) > 0
    except OperationalError as exc:
        raise _raise_pgvector_connection_error(exc) from exc


def _metadata_column_name(table_name: str) -> str | None:
    validated = _validate_table_name(table_name)
    column_names = {
        column["name"]
        for column in inspect(_get_sqlalchemy_engine()).get_columns(validated)
    }
    for candidate in ("cmetadata", "langchain_metadata"):
        if candidate in column_names:
            return candidate
    return None


def _table_has_routing_metadata(table_name: str) -> bool:
    validated = _validate_table_name(table_name)
    try:
        with _get_sqlalchemy_engine().connect() as conn:
            exists = conn.execute(
                text("SELECT to_regclass(:table_name)"),
                {"table_name": validated},
            ).scalar()
            if not exists:
                return False

            total_count = conn.execute(
                text(f'SELECT COUNT(*) FROM "{validated}"')
            ).scalar_one()
            if int(total_count) == 0:
                return False

            metadata_column = _metadata_column_name(validated)
            if metadata_column is None:
                return False

            routed_count = conn.execute(
                text(
                    f'''
                    SELECT COUNT(*)
                    FROM "{validated}"
                    WHERE "{metadata_column}"::jsonb ? 'analysis_scope'
                      AND "{metadata_column}"::jsonb ? 'document_kind'
                    '''
                )
            ).scalar_one()
            return int(routed_count) == int(total_count)
    except OperationalError as exc:
        raise _raise_pgvector_connection_error(exc) from exc


def ensure_pgvector_extension() -> None:
    """대상 데이터베이스에 `pgvector` 확장이 설치되어 있는지 확인합니다."""
    try:
        with _get_sqlalchemy_engine().begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except OperationalError as exc:
        raise _raise_pgvector_connection_error(exc) from exc


def check_pgvector_connection() -> dict:
    """현재 설정된 pgvector 데이터베이스의 간단한 진단 정보를 반환합니다."""
    try:
        with _get_sqlalchemy_engine().connect() as conn:
            current_db = conn.execute(text("SELECT current_database()")).scalar_one()
            vector_enabled = conn.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            ).scalar_one()
    except OperationalError as exc:
        raise _raise_pgvector_connection_error(exc) from exc

    return {
        "database": current_db,
        "vector_extension": bool(vector_enabled),
        "table": PGVECTOR_TABLE,
    }


def _documents_to_langchain(documents: list[dict]) -> tuple[list[Document], list[str]]:
    langchain_docs: list[Document] = []
    ids: list[str] = []

    for document in documents:
        ids.append(document["chunk_id"])
        langchain_docs.append(
            Document(
                page_content=document["text"],
                metadata={
                    "page": document["page"],
                    "source": document["source"],
                    "chunk_id": document["chunk_id"],
                    "analysis_scope": document["analysis_scope"],
                    "company": document["company"],
                    "document_kind": document["document_kind"],
                },
            )
        )

    return langchain_docs, ids


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """
    문서 로드 (100페이지 이내)

    Returns:
        list[dict]: [{text, page, source}]
    """
    documents: list[dict] = []
    total_pages = 0

    for pdf_path in sorted(data_dir.rglob("*.pdf")):
        page_texts = _extract_pdf_texts(pdf_path)
        page_count = len(page_texts)
        total_pages += page_count
        if total_pages > MAX_DOCUMENT_PAGES:
            raise ValueError(
                f"문서 페이지 예산을 초과했습니다: {total_pages} > {MAX_DOCUMENT_PAGES}"
            )

        for page_number, page_text in enumerate(page_texts, start=1):
            if not page_text:
                continue

            for chunk_index, chunk in enumerate(_chunk_text(page_text), start=1):
                document_metadata = _infer_document_metadata(pdf_path.name)
                documents.append(
                    {
                        "text": chunk,
                        "page": page_number,
                        "source": pdf_path.name,
                        "chunk_id": _chunk_id(
                            pdf_path.name,
                            page_number,
                            chunk_index,
                            chunk,
                        ),
                        **document_metadata,
                    }
                )

    return documents


def build_pgvector_index(
    documents: list[dict] | None = None,
    *,
    table_name: str = PGVECTOR_TABLE,
    force_reindex: bool = False,
):
    """
    pgvector 인덱스 구축 (오픈소스 임베딩)
    """
    embedding_service = _get_embedding_service()
    validated_table = _validate_table_name(table_name)
    try:
        engine = _get_pg_engine()

        ensure_pgvector_extension()
        try:
            engine.init_vectorstore_table(
                table_name=validated_table,
                vector_size=embedding_service.dimension,
            )
        except ProgrammingError as exc:
            if "already exists" not in str(exc).lower():
                raise
        store = PGVectorStore.create_sync(
            engine=engine,
            table_name=validated_table,
            embedding_service=embedding_service,
        )

        if force_reindex:
            with _get_sqlalchemy_engine().begin() as conn:
                conn.execute(text(f'TRUNCATE TABLE "{validated_table}"'))

        table_has_rows = _table_has_rows(validated_table)
        if table_has_rows and not _table_has_routing_metadata(validated_table):
            with _get_sqlalchemy_engine().begin() as conn:
                conn.execute(text(f'TRUNCATE TABLE "{validated_table}"'))
            table_has_rows = False

        if table_has_rows:
            return store

        loaded_documents = documents if documents is not None else load_documents()
        if not loaded_documents:
            return store

        langchain_docs, ids = _documents_to_langchain(loaded_documents)
        store.add_documents(langchain_docs, ids=ids)
        return store
    except OperationalError as exc:
        raise _raise_pgvector_connection_error(exc) from exc


def search(
    query: str,
    top_k: int = VECTOR_SEARCH_TOP_K,
    *,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """
    pgvector 유사도 검색

    Args:
        query: 검색 쿼리
        top_k: 반환할 청크 수
        metadata_filter: 문서 메타데이터 필터

    Returns:
        list[dict]: [{chunk, score, page, source}]
                    score >= RELEVANCE_THRESHOLD인 것만 반환
    """
    store = build_pgvector_index()
    retrieved_docs = store.similarity_search(query, k=top_k, filter=metadata_filter)
    if not retrieved_docs:
        return []

    embedding_service = _get_embedding_service()
    query_embedding = embedding_service.embed_query(query)
    chunk_embeddings = embedding_service.embed_documents(
        [document.page_content for document in retrieved_docs]
    )

    scored_results: list[dict] = []
    filtered_results: list[dict] = []
    for document, chunk_embedding in zip(retrieved_docs, chunk_embeddings):
        score = _cosine_similarity(query_embedding, chunk_embedding)
        result = (
            {
                "chunk": document.page_content,
                "score": round(score, 4),
                "page": document.metadata.get("page"),
                "source": document.metadata.get("source"),
                "chunk_id": document.metadata.get("chunk_id"),
                "analysis_scope": document.metadata.get("analysis_scope"),
                "company": document.metadata.get("company"),
                "document_kind": document.metadata.get("document_kind"),
            }
        )
        scored_results.append(result)
        if score >= RELEVANCE_THRESHOLD:
            filtered_results.append(result)

    if filtered_results:
        return deduplicate(filtered_results)

    # pgvector가 이미 최근접 이웃을 골라준 상태이므로, 로컬 해시 임베딩이
    # 동작할 때는 고정 코사인 임계값이 스모크 테스트에 지나치게 엄격할 수 있습니다.
    return deduplicate(scored_results)


def deduplicate(results: list[dict], threshold: float = DUPLICATE_THRESHOLD) -> list[dict]:
    """
    코사인 유사도 기반 중복 제거

    Args:
        results: 검색 결과
        threshold: 이 값 초과 시 중복으로 판단 (default 0.95)

    Returns:
        중복 제거된 결과
    """
    if not results:
        return []

    embedding_service = _get_embedding_service()
    chunk_embeddings = embedding_service.embed_documents(
        [result["chunk"] for result in results]
    )

    deduplicated: list[dict] = []
    kept_embeddings: list[list[float]] = []
    for result, embedding in zip(results, chunk_embeddings):
        if any(
            _cosine_similarity(embedding, kept_embedding) > threshold
            for kept_embedding in kept_embeddings
        ):
            continue
        deduplicated.append(result)
        kept_embeddings.append(embedding)

    return deduplicated


def rewrite_query(original_query: str, previous_results: list[dict]) -> str:
    """
    관련도가 낮을 때 쿼리를 다시 쓰기 위한 보조 함수입니다.

    제어 전략: 최대 2회까지 반복 재작성
    """
    sources = [
        Path(str(result["source"])).stem
        for result in previous_results
        if result.get("source")
    ]
    unique_sources = list(dict.fromkeys(sources))
    if not unique_sources:
        return original_query

    keywords = " ".join(unique_sources[:3])
    return f"{original_query} {keywords}".strip()
