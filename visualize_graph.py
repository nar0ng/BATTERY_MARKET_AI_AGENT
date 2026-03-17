"""
Visualize the LangGraph workflow from the repository root.

Run this with the Python environment that has `langgraph` and
`langchain_teddynote` installed.
"""


def main() -> None:
    try:
        from langchain_teddynote.graphs import visualize_graph
        from src.graph import graph
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing LangGraph visualization dependencies. Activate the Python "
            "environment where `langgraph` and `langchain_teddynote` are "
            "installed, then run `python visualize_graph.py`."
        ) from exc

    visualize_graph(graph)


if __name__ == "__main__":
    main()
