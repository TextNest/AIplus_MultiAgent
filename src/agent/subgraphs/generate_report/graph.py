
from langgraph.graph import StateGraph, END
from .state import ReportState
from .nodes import generate_content, create_pdf, create_html, create_pptx
from .supervisor import report_supervisor

def create_report_subgraph():
    workflow = StateGraph(ReportState)
    
    # 1. Add Nodes
    workflow.add_node("supervisor", report_supervisor)
    workflow.add_node("generate_content", generate_content)
    workflow.add_node("create_pdf", create_pdf)
    workflow.add_node("create_html", create_html)
    workflow.add_node("create_pptx", create_pptx)
    
    # 2. Set Entry Point
    workflow.set_entry_point("supervisor")
    
    # 3. Add Edges (Workers -> Supervisor)
    workflow.add_edge("generate_content", "supervisor")
    workflow.add_edge("create_pdf", "supervisor")
    workflow.add_edge("create_html", "supervisor")
    workflow.add_edge("create_pptx", "supervisor")
    
    # 4. Conditional Edges (Supervisor -> Workers)
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_worker"],
        {
            "generate_content": "generate_content",
            "create_pdf": "create_pdf",
            "create_html": "create_html",
            "create_pptx": "create_pptx",
            "FINISH": END
        }
    )
    
    return workflow.compile()