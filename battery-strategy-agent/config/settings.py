"""
시스템 설정 — threshold, 비용 한도, 모델 설정
Success Criteria 기준
"""
import os
from pathlib import Path
from urllib.parse import quote_plus, urlsplit, urlunsplit

from dotenv import dotenv_values

# ── 경로 ──
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"


def _load_env_defaults() -> None:
    """루트 저장소와 프로젝트 루트의 `.env` 값을 기본값으로 불러옵니다."""
    inherited_keys = set(os.environ)
    env_candidates = [
        PROJECT_ROOT.parent / ".env",
        PROJECT_ROOT / ".env",
    ]
    for env_path in env_candidates:
        if not env_path.exists():
            continue
        for key, value in dotenv_values(env_path).items():
            if value is None or key in inherited_keys:
                continue
            os.environ[key] = value


_load_env_defaults()

# ── 대상 기업 ──
TARGET_COMPANIES = ["LG에너지솔루션", "CATL"]

# ── 임베딩 모델 (오픈소스 필수) ──
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# ── RAG 설정 ──
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", "10"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.65"))
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.95"))
MAX_DOCUMENT_PAGES = int(os.getenv("MAX_DOCUMENT_PAGES", "100"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# ── pgvector 설정 ──
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "battery_strategy")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "battery_strategy_chunks")
PGVECTOR_CONNECTION = os.getenv("PGVECTOR_CONNECTION") or os.getenv("DATABASE_URL")


def build_pgvector_connection() -> str:
    quoted_user = quote_plus(POSTGRES_USER) if POSTGRES_USER else ""
    quoted_password = quote_plus(POSTGRES_PASSWORD) if POSTGRES_PASSWORD else ""

    auth = ""
    if quoted_user:
        auth = quoted_user
        if quoted_password:
            auth = f"{auth}:{quoted_password}"
        auth = f"{auth}@"

    location = POSTGRES_HOST.strip()
    if location and POSTGRES_PORT:
        location = f"{location}:{POSTGRES_PORT}"

    return f"postgresql+psycopg://{auth}{location}/{POSTGRES_DB}"


def mask_connection_string(connection_string: str) -> str:
    parsed = urlsplit(connection_string)
    if not parsed.netloc or parsed.password is None:
        return connection_string

    userinfo = parsed.username or ""
    if parsed.password is not None:
        userinfo = f"{userinfo}:***" if userinfo else "***"

    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    netloc = hostname
    if userinfo:
        netloc = f"{userinfo}@{netloc}"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


if not PGVECTOR_CONNECTION:
    PGVECTOR_CONNECTION = build_pgvector_connection()

# ── Web Search 설정 (시장 분석 Agent) ──
BIAS_RATIO_MIN = 0.4  # 찬반 비율 최소 (4:6)
BIAS_RATIO_MAX = 0.6  # 찬반 비율 최대 (6:4)
BIAS_TRIGGER = 0.7    # 이 비율 초과 시 보충 검색

# ── Control Strategy ──
MAX_QUERY_REWRITE = 2       # RAG Query Rewrite 최대 횟수
MAX_WEB_SUPPLEMENT = 2      # Web 보충 검색 최대 횟수 (기업당)
MAX_SWOT_CORRECTION = 1     # SWOT 인라인 교정 최대 횟수
MAX_REPORT_RETRY = 1        # 보고서 섹션 재생성 최대 횟수

# ── 종료 정책 (Supervisor) ──
MAX_ITERATIONS = 3          # 전체 품질 루프 최대 반복
MAX_LLM_CALLS = 15          # LLM 호출 한도
MAX_WEB_SEARCHES = 10       # 웹 검색 한도

# ── 보고서 설정 ──
SUMMARY_MAX_WORDS = 300     # SUMMARY 최대 단어 수 (약 1/2 페이지)
REQUIRED_SECTIONS = [
    "SUMMARY",
    "시장 배경",
    "기업별 포트폴리오 다각화 전략 및 핵심 경쟁력",
    "핵심 전략 비교 및 SWOT 분석",
    "종합 시사점",
    "REFERENCE",
]

# ── LLM 설정 ──
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.2       # 분석 보고서이므로 낮은 온도값 사용

# ── SWOT 분석 기준 ──
# 정량 기준
MIN_SWOT_ITEMS_PER_CATEGORY = 2
SWOT_EVIDENCE_REQUIRED = True
MAX_SWOT_MISCLASSIFIED = 0
MAX_SWOT_CORRECTION = 1

# 판정 지표 (2×2 매트릭스 기반)
SWOT_CRITERIA = {
    "S": {
        "axis": ["내부", "긍정"],
        "question": "경쟁사 대비 차별화된 내부 역량인가?",
        "indicators": [
            "핵심 기술력 (특허, 독자 기술)",
            "자본력/재무 건전성",
            "생산 규모 (Capa, 수율)",
            "브랜드/시장 신뢰도",
            "고객 포트폴리오 다양성",
            "공급망 내재화",
            "원가 경쟁력 (내부 효율)",
            "R&D 인력/조직 역량",
        ],
    },
    "W": {
        "axis": ["내부", "부정"],
        "question": "경쟁사 대비 부족하고 개선이 필요한 내부 요소인가?",
        "indicators": [
            "특정 기술 의존도/편중",
            "높은 비용 구조",
            "부족한 자원 (인력, 설비)",
            "재무 부담 (부채, 적자)",
            "고객 집중도 (소수 의존)",
            "기술 전환 지연",
            "지역 편중 (생산 거점)",
            "낮은 수율/품질 이슈",
        ],
    },
    "O": {
        "axis": ["외부", "긍정"],
        "question": "비즈니스에 유리하게 작용하는 외부 환경 변화인가?",
        "indicators": [
            "시장 성장 (EV 수요 확대)",
            "정부 보조금/인센티브 (IRA, EU)",
            "규제 완화",
            "기술 발전 (새 시장 창출)",
            "원자재 가격 하락",
            "신규 시장 개방",
            "경쟁사 약화/철수",
            "소비자 트렌드 (친환경)",
        ],
    },
    "T": {
        "axis": ["외부", "부정"],
        "question": "비즈니스에 부정적 영향을 줄 수 있는 외부 환경인가?",
        "indicators": [
            "경쟁 심화 (증설, 신규 진입)",
            "경기 침체 (수요 둔화)",
            "법적 규제 강화 (관세, 탄소)",
            "원자재 가격 급등",
            "무역 분쟁 (미중 갈등)",
            "대체 기술 등장 (전고체, 나트륨)",
            "환율 변동",
            "지정학적 리스크",
        ],
    },
}
