from langgraph.graph import StateGraph, END
from .state import ReportState
from .nodes import create_pdf, create_html, create_pptx
from .supervisor import report_supervisor
def create_report_subgraph():
    workflow = StateGraph(ReportState)
    
    # 노드 추가
    workflow.add_node("supervisor", report_supervisor)
    workflow.add_node("create_pdf", create_pdf)
    workflow.add_node("create_html", create_html)
    workflow.add_node("create_pptx", create_pptx)
    
    # 진입점: 수퍼바이저가 먼저 판단
    workflow.set_entry_point("supervisor")
    
    # 작업자는 작업 후 다시 수퍼바이저에게 보고 (또는 종료)
    workflow.add_edge("create_pdf", "supervisor")
    workflow.add_edge("create_html", "supervisor")
    workflow.add_edge("create_pptx", "supervisor")
    
    # 수퍼바이저의 결정에 따른 분기
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_worker"],
        {
            "create_pdf": "create_pdf",
            "create_html": "create_html",
            "create_pptx": "create_pptx",
            "FINISH": END
        }
    )
    
    return workflow.compile()