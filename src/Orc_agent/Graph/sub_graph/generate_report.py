from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from ...State.state import ReportState
from ...Node.sub_node.generate_report import (
    report_supervisor, generate_content, create_pdf, create_html, create_pptx
)

def generate_report_graph(CheckPoint=None):
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
    
    return workflow.compile(checkpointer=CheckPoint)