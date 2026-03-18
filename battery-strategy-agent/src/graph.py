"""
LangGraph 워크플로우 정의.
모든 작업 노드가 Supervisor로 되돌아오는 허브 앤 스포크 구조를 사용합니다.
"""
from langgraph.graph import StateGraph, END

from src.state import ReportState


def _supervisor_node(state: ReportState) -> dict:
    from src.agents.supervisor import supervisor_node

    return supervisor_node(state)


def _supervisor_route(state: ReportState):
    from src.agents.supervisor import supervisor_route

    return supervisor_route(state)


def _market_analyst_node(state: ReportState) -> dict:
    from src.agents.market_analyst import market_analyst_node

    return market_analyst_node(state)


def _company_analyst_node(state: ReportState) -> dict:
    from src.agents.company_analyst import company_analyst_node

    return company_analyst_node(state)


def _swot_extractor_node(state: ReportState) -> dict:
    from src.agents.swot_extractor import swot_extractor_node

    return swot_extractor_node(state)


def _report_writer_node(state: ReportState) -> dict:
    from src.agents.report_writer import report_writer_node

    return report_writer_node(state)


def build_graph() -> StateGraph:
    """Supervisor 패턴 기반 그래프를 구성합니다."""

    workflow = StateGraph(ReportState)

    # ── 노드 등록 ──
    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node("market_analyst", _market_analyst_node)
    workflow.add_node("company_analyst", _company_analyst_node)
    workflow.add_node("swot_extractor", _swot_extractor_node)
    workflow.add_node("report_writer", _report_writer_node)

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
        _supervisor_route,
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
