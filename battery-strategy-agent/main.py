"""
battery-strategy-agent 실행 엔트리포인트
LG에너지솔루션·CATL 배터리 시장 전략 분석 보고서 생성
"""
from src.state import create_initial_state
from src.graph import graph


def run(query: str) -> str:
    """
    전략 분석 보고서 생성 실행

    Args:
        query: 사용자 질의 (예: "LG에너지솔루션과 CATL의 배터리 전략을 비교 분석해줘")

    Returns:
        최종 보고서 텍스트
    """
    initial_state = create_initial_state(query)
    final_state = graph.invoke(initial_state)
    final_report = final_state.get("final_report") or ""
    report_draft = final_state.get("report_draft") or ""

    if final_report:
        print("✅ Supervisor가 최종 보고서를 확정했습니다")
        return final_report

    if report_draft:
        print("⚠️  Supervisor 승인 전 초안만 생성되었습니다")
        details = final_state.get("quality_score", {}).get("details", [])
        for d in details:
            print(f"   - {d}")
        return report_draft

    print("❌ 보고서 생성 실패")
    return ""


if __name__ == "__main__":
    query = "LG에너지솔루션과 CATL의 배터리 포트폴리오 다각화 전략을 과거, 현재, 미래 관점에서 비교 분석해줘"
    report = run(query)
    if report:
        from config.settings import OUTPUT_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "strategy_report.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"📄 보고서 저장: {output_path}")
