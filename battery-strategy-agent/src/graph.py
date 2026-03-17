"""
LangGraph 워크플로우 정의.
모든 작업 노드가 Supervisor로 되돌아오는 허브 앤 스포크 구조를 사용합니다.
"""
from langgraph.graph import StateGraph, END

from src.state import ReportState
from src.agents.supervisor import supervisor_node, supervisor_route
from src.agents.market_analyst import market_analyst_node
from src.agents.company_analyst import company_analyst_node
from src.agents.swot_extractor import swot_extractor_node
from src.agents.report_writer import report_writer_node


def build_graph() -> StateGraph:
    """Supervisor 패턴 기반 그래프를 구성합니다."""

    workflow = StateGraph(ReportState)

    # ── 노드 등록 ──
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("market_analyst", market_analyst_node)
    workflow.add_node("company_analyst", company_analyst_node)
    workflow.add_node("swot_extractor", swot_extractor_node)
    workflow.add_node("report_writer", report_writer_node)

    # ── 진입점 ──
    workflow.set_entry_point("supervisor")

    # ── 핵심: 모든 Agent → Supervisor로 복귀 (Hub-and-Spoke) ──
    for agent in [
        "market_analyst",
        "company_analyst",
        "swot_extractor",
        "report_writer",
    ]:
        workflow.add_edge(agent, "supervisor")

    # ── Supervisor → 조건부 라우팅 (병렬 Send 포함) ──
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "market_analyst": "market_analyst",
            "company_analyst": "company_analyst",
            "swot_extractor": "swot_extractor",
            "report_writer": "report_writer",
            "end": END,
        },
    )

    return workflow.compile()


# 그래프 인스턴스
graph = build_graph()
