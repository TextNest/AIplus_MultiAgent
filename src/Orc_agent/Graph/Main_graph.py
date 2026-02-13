from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from ..State.state import AgentState
from ..Node import Main_node 
from langgraph.graph import START
from .sub_graph import analyze_data,document_agent
from .sub_graph.generate_report.graph import create_report_subgraph

def create_main_graph():
    analyze_app = analyze_data.analyze_data_graph()
    document_app = document_agent.document_agent_graph()
    main_workflow = StateGraph(AgentState)
    main_workflow.add_node("File_type",Main_node.file_type)
    main_workflow.add_node("File_analysis",Main_node.file_analyze(document_app))
    main_workflow.add_node("Preprocessing",Main_node.preprocessing)#아직 추가 x
    main_workflow.add_node("Analysis",Main_node.analysis(analyze_app))
    main_workflow.add_node("Wait",Main_node.human_review_wait)

    # 기존:
    # main_workflow.add_node("Final_report",Main_node.final_report)#아직 추가 x

    # 변경:
    report_app = create_report_subgraph()
    main_workflow.add_node("generate_report", report_app) 

    main_workflow.add_edge(START,"File_type")
    main_workflow.add_edge("File_analysis",END)
    main_workflow.add_edge("Preprocessing","Analysis")
    
    # 기존:
    # main_workflow.add_edge("Analysis","Final_report")
    # main_workflow.add_edge("Final_report","Wait")

    # 변경:
    main_workflow.add_edge("Analysis", "generate_report")
    main_workflow.add_edge("generate_report", "Wait")

    main_workflow.add_conditional_edges(
        "File_type",
        Main_node.select_agent,
        {
            "tabular":"Preprocessing",
            "document":"File_analysis"
        }
    )
    main_workflow.add_conditional_edges(
        "Wait",
        Main_node.human_review_route,
        {
            "END":END,
            "analysis":"Analysis"
        }
    )
    main_app = main_workflow.compile(checkpointer=MemorySaver(),interrupt_before=["Wait"])
    return main_app