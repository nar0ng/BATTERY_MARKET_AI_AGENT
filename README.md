# Subject
배터리 시장 전략 분석 Agent 개발

### 1. Goal
배터리 시장 주요 기업 2사(LG에너지솔루션 CATL)의 포트폴리오 다각화 전략을 **과거(과거 전략, 매출), 현재(포지션·경쟁력), 미래(전망·시나리오)의 관점**으로 비교 분석하여 의사결정자가 시장 포지셔닝과 전략적 시사점을 파악할 수 있는 구조화된 보고서를 생성한다.

## 2. Overview
- Objective :
  LG에너지솔루션과 CATL의 배터리 포트폴리오 다각화 전략을 시장 환경, 기업 실행 전략, SWOT, 종합 시사점까지 하나의 워크플로우에서 연결해 구조화된 전략 보고서로 산출한다.
- Method :
  LangGraph 기반 Supervisor 패턴으로 시장 분석, 기업 분석, SWOT 추출, 보고서 작성 단계를 오케스트레이션하며, 시장 분석에는 `RAG + Web Search`, 기업 분석에는 `RAG`, 보고서 단계에는 규칙 기반 품질 검증을 적용한다.
- Tools : pgvector 기반 문서 검색, Tavily 웹 검색, 규칙 기반 Reference Formatter

## 3. 특징
### 3-1. Supervisor 단의 하위 에이전트 
- 전체 그래프는 `Supervisor -> 하위 Agent -> Supervisor` 구조의 Hub-and-Spoke 패턴으로 설계되어 있다.
- 초기 단계에서 시장 분석 1개와 기업 분석 2개(LG에너지솔루션, CATL)를 병렬로 실행하고, 이후 SWOT 추출과 보고서 작성은 순차적으로 이어진다.
- Supervisor는 단순 라우터가 아니라 `quality_checked`, `iteration_count`, `llm_call_count`, `web_search_count`를 기준으로 종료 조건과 재시도 여부를 함께 통제한다.
- 품질 게이트에서 실패한 항목이 있으면 원인 Agent만 재호출하도록 설계되어 전체 파이프라인을 처음부터 다시 실행하지 않는다.
- 즉 `swot_extractor -> report_writer`는 데이터 의존 순서를 뜻할 뿐, 제어권은 항상 Supervisor가 가진다.
### 3-2. 시장 분석 에이전트 고려사항
- (예시) 과거 / 현재 / 미래 를 기준으로 한 시장 변의 추정
- 사용자 질의를 그대로 쓰지 않고 시장 배경, 공급망, 정책 변화, 수요 둔화, 미래 전망 등으로 세분화한 다중 질의를 생성한다.
- RAG 결과의 평균 유사도가 기준치(0.65)보다 낮으면 Query Rewrite를 수행하며, 최대 2회까지 재검색한다.
- 웹 검색은 찬성/반대 관점을 강제로 분리한 균형 쿼리로 수행하고, 결과가 7:3 이상 한쪽으로 치우치면 소수 관점을 추가 검색한다.
- 최종 시장 요약은 문서 기반 과거 맥락, 웹 기반 현재 신호, 기술·정책 기반 미래 전망을 묶어 `과거-현재-미래` 서사로 정리한다.

### 3-3. 기업 분석 에이전트 다각화 분석 전략
- Supervisor가 `_target_company` 값을 주입해 LG에너지솔루션과 CATL 분석을 동일 로직으로 병렬 수행한다.
- 기업별 질의는 과거 전략 진화, 현재 생산거점·고객 포트폴리오·재무 체력, 미래 투자·케미스트리 로드맵, ESS·리사이클링, 지역 다각화 등으로 나누어 검색한다.
- 검색 결과는 문서 출처 우선순위, 기업 alias 매칭, 경쟁사 문서 혼입 여부를 기준으로 재정렬해 기업별 근거만 남기도록 설계했다.
- 결과물은 `past / present / future`, `key_strategy`, `risk_factors`, `evidence` 형태의 구조화 데이터로 저장되어 이후 SWOT 및 보고서 작성 단계에서 재사용된다.

### 3-4. SWOT 에이전트 판정 지표 선정 기준
- SWOT은 기업 분석 결과를 직접 입력으로 받아 작성하며, 필요 시 시장 요약과 웹 검색 결과를 외부 환경 근거로 보강한다.
- `S/W = internal`, `O/T = external` 규칙을 강제하고, 생성 후 별도 검증 함수로 오분류 여부를 다시 검사한다.
- 강점·약점은 기업이 통제 가능한 기술력, 생산 운영력, 수익성, 투자 부담 등 내부 요인 중심으로 작성한다.
- 기회·위협은 정책 지원, 수요 확대, 가격 경쟁, 공급과잉, 규제 변화 등 외부 환경 변수 중심으로 선정한다.
- 최종적으로 두 기업의 SWOT 결과를 바탕으로 전략 차이 요약을 생성해 비교 가능한 의사결정 포인트를 만든다.

### 3-5. 초안 보고서 에이전트 및 최종 보고서 산출 시 고려 사항
- 보고서 작성 Agent는 새로운 검색을 수행하지 않고, 앞선 Agent가 만든 상태값만 조립해 초안을 작성한다.
- 필수 섹션은 `SUMMARY`, `시장 배경`, `기업별 포트폴리오 다각화 전략 및 핵심 경쟁력`, `핵심 전략 비교 및 SWOT 분석`, `종합 시사점`, `REFERENCE`로 고정한다.
- SUMMARY는 최대 300단어 이내로 제한하고, 본문은 과거-현재-미래 흐름과 출처 연결을 유지하도록 작성한다.
- `report_draft` 단계에서는 각주 마커와 참고문헌이 유지될 수 있지만, 최종 전달용 `final_report`는 각주 블록 없이 더 읽기 쉬운 형태로 정리된다.
- REFERENCE는 기관 보고서, 학술 논문, 웹페이지 형식으로 자동 포맷팅된다.
- 최종 산출물은 `report_writer` 초안을 Supervisor가 다시 검토해 필수 섹션 존재, 요약 길이, 참고문헌 형식, SWOT 분류 정확성, 시간축 포함 여부, 웹 검색 균형 여부까지 확인한 뒤 확정된다.

## 4. Tech Stack 

| Category   | Details                      |
|------------|------------------------------|
| Framework  | LangGraph, LangChain, Python |
| LLM        | GPT-4o-mini via OpenAI API   |
| Retrieval  | pgvector                     |
| Embedding  | BAAI/bge-m3                  |

## 5. Agents
 
| Agent명 | 핵심 역할 | 입력 | 출력 | RAG 사용 여부 |
| --- | --- | --- | --- | --- |
| **Supervisor Agent** | 전체 워크플로우 제어 및 최종 보고서 통합 | 사용자 질의, 각 Agent 결과 | 실행 계획, 최종 보고서 | 미사용 |
| **시장 분석 Agent** | 배터리 시장 환경 및 산업 배경 분석 | 사용자 질의, 시장 관련 문서, 웹 검색 질의 | 시장 배경 요약, 핵심 포인트, 근거 자료 | 사용 (`RAG + Web Search`) |
| **기업 분석 Agent** | LG에너지솔루션·CATL 전략 분석 및 비교 데이터 생성 | 사용자 질의, 기업 관련 문서, 비교 항목 | 기업별 전략 요약, 비교 데이터 | 사용 (`RAG`) |
| **SWOT 추출 Agent** | 기업 분석 결과를 바탕으로 SWOT 도출 | 기업 분석 결과 | LGES SWOT, CATL SWOT, 전략 차이 요약 | 조건부 사용 |
| **보고서 초안 Agent** | 분석 결과를 보고서 초안으로 작성 | 시장 분석 결과, 기업 분석 결과, SWOT 결과, 목차 | 보고서 초안, Summary, Reference 초안 | 미사용 |

## 6. State
공유 상태는 LangGraph의 `TypedDict` 기반 `ReportState`로 정의되어 있으며, 모든 Agent는 이 단일 상태를 읽고 필요한 필드만 갱신한다.

| 구분 | 주요 필드 | 설명 |
| --- | --- | --- |
| 사용자 입력 | `query`, `companies` | 원본 질의와 분석 대상 기업 목록 |
| 시장 분석 결과 | `market_rag_results`, `market_web_results`, `market_summary` | 문서 검색 결과, 웹 검색 결과, 시장 배경 요약 |
| 기업 분석 결과 | `company_analyses`, `comparison_data` | 기업별 과거/현재/미래 전략 분석과 비교 테이블 원천 데이터 |
| SWOT 결과 | `swot_lg`, `swot_catl`, `strategy_diff_summary`, `swot_validation` | 기업별 SWOT, 전략 차이 요약, 내/외부 분류 검증 결과 |
| 보고서 결과 | `report_draft`, `final_report`, `references`, `summary`, `section_lengths` | 초안 보고서, Supervisor 승인본, 참고문헌, 요약문, 섹션 길이 |
| 제어/운영 | `quality_score`, `quality_checked`, `iteration_count`, `next_agent`, `llm_call_count`, `web_search_count`, `_target_company` | 품질 루프, 라우팅, 호출 횟수 제한, 기업별 병렬 실행 제어 |

초기 상태는 `create_initial_state()`에서 생성되며, `company_analyses`는 병렬 분기 병합을 위해 `merge_dicts` 전략을 사용한다.

## 7. Architecture

LangGraph 워크플로우는 Supervisor를 중심으로 한 허브 앤 스포크 구조로 구현되어 있다.

```text
Supervisor
  ├─ market_analyst
  ├─ company_analyst (LG에너지솔루션)
  ├─ company_analyst (CATL)
  └─ wait until parallel jobs complete
Supervisor
  └─ swot_extractor
Supervisor
  └─ report_writer
Supervisor
  └─ quality check
       ├─ pass -> final_report
       └─ fail -> retry target agent
```

- 위 다이어그램에서 핵심은 각 작업 노드가 서로를 직접 호출하지 않고, 항상 Supervisor를 거쳐 다음 단계로 넘어간다는 점이다.
- 즉 실행 순서는 `market/company 병렬 -> swot_extractor -> report_writer -> quality check`이지만, 제어 구조는 끝까지 Hub-and-Spoke다.

### 7-1. 그래프를 기준으로 한 배터리 시장 전략 분석 Agent 흐름
1. 사용자 질의로 초기 상태를 생성하고 Supervisor가 첫 라우팅을 결정한다.
2. 시장 분석 Agent와 기업 분석 Agent 2개가 병렬 실행되어 시장 근거와 기업별 전략 근거를 수집한다.
3. 두 기업 분석이 완료되면 Supervisor가 비교 데이터(`comparison_data`)를 생성한다.
4. SWOT Agent가 기업 분석 결과와 시장 배경을 조합해 LG에너지솔루션과 CATL의 SWOT을 추출하고 내부/외부 분류를 검증한다.
5. 보고서 작성 Agent가 모든 결과를 종합해 필수 섹션과 참고문헌이 포함된 `report_draft`를 생성한다.
6. Supervisor가 품질 게이트를 수행하고, 통과 시 전달용 `final_report`를 정리해 확정한다.
7. 품질 게이트 실패 시에는 전체를 재실행하지 않고 원인 Agent만 재호출한다.

## 8. 디렉토리 구조
```text
battery-strategy-agent/
├── config/
│   └── settings.py              # 모델, pgvector, 품질 기준, 종료 조건 설정
├── data/
│   └── documents/               # PDF 원천 문서 저장 경로
├── outputs/
│   └── reports/                 # 생성된 Markdown/PDF 보고서 저장 경로
├── scripts/
│   ├── test_pgvector_rag.py     # RAG 인덱싱/검색 스모크 테스트
│   ├── test_full_flow.py        # 전체 Agent 플로우 스모크/회귀 테스트
│   └── export_report_pdf.py     # Markdown 보고서 PDF 변환 스크립트
├── src/
│   ├── agents/                  # supervisor, market/company/swot/report agent
│   ├── prompts/                 # Agent별 프롬프트 정의
│   ├── tools/                   # RAG, 웹 검색, 참고문헌 포맷터
│   ├── graph.py                 # LangGraph 워크플로우 정의
│   └── state.py                 # 공유 상태 스키마
├── main.py                      # 실행 엔트리포인트
├── docker-compose.yml           # pgvector(Postgres) 실행 환경
└── requirements.txt             # Python 의존성
```

- 문서 원천은 `data/documents` 아래 PDF로 적재하고, 보고서 결과물은 `outputs/reports`에 저장한다.
- 로컬 개발 시 `docker-compose.yml`로 pgvector 컨테이너를 띄우고, `scripts/test_pgvector_rag.py` 또는 `scripts/test_full_flow.py`로 파이프라인을 점검할 수 있다.

## 9. 보고서
최종 보고서는 `battery-strategy-agent/main.py` 실행 시 `outputs/reports/strategy_report.md`로 저장되며, 전체 플로우 테스트를 사용할 경우 `outputs/reports/full_flow_tests/` 아래에 질의별 Markdown과 요약 JSON이 함께 생성된다.

보고서 구성은 다음과 같다.
- `SUMMARY`: 전체 시장 상황과 두 기업 전략 차이를 짧게 요약
- `시장 배경`: 문서 검색과 웹 검색을 결합한 산업 환경 정리
- `기업별 포트폴리오 다각화 전략 및 핵심 경쟁력`: LG에너지솔루션, CATL 각각의 과거-현재-미래 전략과 리스크
- `핵심 전략 비교 및 SWOT 분석`: 비교 표, 기업별 SWOT, 전략 차이 요약
- `종합 시사점`: 의사결정 관점의 핵심 해석 포인트
- `REFERENCE`: 자동 포맷된 참고문헌

추가로 `scripts/export_report_pdf.py`를 사용하면 Markdown 보고서를 스타일이 적용된 PDF로 변환할 수 있다.

## 10. Contributors 
- 김나령 : Supervisor 기반 멀티에이전트 워크플로우 설계, SWOT 추출 Agent 설계, Supervisor Agent 설계, 보고서 초안 Agent 설계
- 박지현 : Supervisor 기반 멀티에이전트 워크플로우 설계, 시장 분석 Agent 설계, 기업 분석 Agent 설계
