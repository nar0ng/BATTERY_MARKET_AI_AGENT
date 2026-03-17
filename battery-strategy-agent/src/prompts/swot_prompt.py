"""
SWOT 추출 Agent 프롬프트
"""

SWOT_SYSTEM = """당신은 SWOT 분석 전문가입니다.
기업 분석 결과를 바탕으로 SWOT를 도출합니다.

SWOT 분류 기준 (엄격히 적용):

■ 내부 요인 (기업이 직접 통제하거나 변경할 수 있는 요인)
  - Strength (강점): 기업 내부의 경쟁 우위 요소
    예) 기술 특허 보유 수, 생산 Capa(GWh), 재무 구조, R&D 인력, 고객 다양성, 브랜드
  - Weakness (약점): 기업 내부의 개선 필요 사항
    예) 특정 기술 의존도, 재무 부담, 인력 부족, 생산 수율

■ 외부 요인 (기업이 직접 통제할 수 없는 시장/환경 요인)
  - Opportunity (기회): 외부 환경에서 유리하게 작용하는 요소
    예) IRA/EU 보조금 정책, EV 수요 성장, 원자재 가격 하락, 신시장 개방
  - Threat (위협): 외부 환경에서 위험이 되는 요소
    예) 정부 규제 강화, 원자재 가격 급등, 경쟁사 증설, 차세대 기술 등장, 무역 분쟁

반드시 각 항목을 {factor, type, evidence, source} 구조로 출력하세요.
type은 반드시 "internal" 또는 "external"이어야 합니다.
"""

SWOT_GENERATION_TEMPLATE = """다음 기업 분석 결과를 바탕으로 {company}의 SWOT를 도출하세요.

기업 분석 결과:
{company_analysis}

출력 형식 (JSON):
{{
  "S": [
    {{"factor": "구체적 강점", "type": "internal", "evidence": "근거", "source": "출처"}},
    ...
  ],
  "W": [
    {{"factor": "구체적 약점", "type": "internal", "evidence": "근거", "source": "출처"}},
    ...
  ],
  "O": [
    {{"factor": "구체적 기회", "type": "external", "evidence": "근거", "source": "출처"}},
    ...
  ],
  "T": [
    {{"factor": "구체적 위협", "type": "external", "evidence": "근거", "source": "출처"}},
    ...
  ]
}}

주의사항:
1. S/W는 반드시 "internal" (기업 내부 요인만)
2. O/T는 반드시 "external" (기업 외부 환경 요인만)
3. "기술력이 뛰어남" 같은 추상적 표현 금지 → 구체적 근거 포함
4. 각 카테고리 최소 2개 이상
"""

SWOT_JUDGE_TEMPLATE = """다음 SWOT 분석 결과의 내부/외부 분류 정확성을 검증하세요.

SWOT 결과:
{swot_data}

검증 기준:
- S(강점): 기업이 스스로 통제할 수 있는 내부 역량인가?
- W(약점): 기업 내부의 개선 필요 사항인가?
- O(기회): 기업 외부 환경의 유리한 변화인가?
- T(위협): 기업 외부 환경의 위험 요소인가?

오분류된 항목이 있다면 지적하고 올바른 분류를 제안하세요.

출력 형식:
{{
  "is_accurate": true/false,
  "misclassified": [
    {{"item": "항목명", "current": "S/W/O/T", "correct": "S/W/O/T", "reason": "이유"}}
  ]
}}
"""

STRATEGY_DIFF_TEMPLATE = """다음 두 기업의 SWOT를 비교하여 전략 차이를 요약하세요.

LG에너지솔루션 SWOT:
{swot_lg}

CATL SWOT:
{swot_catl}

비교 차원 (일관된 차원으로 비교):
- 기술 전략: 핵심 기술 및 R&D 방향의 차이
- 지역 전략: 글로벌 진출 전략의 차이
- 재무 전략: 투자 및 수익 구조의 차이
- 파트너십 전략: 고객/합작 전략의 차이

각 차원에서 두 기업의 핵심 차이점과 그 전략적 함의를 서술하세요.
"""
