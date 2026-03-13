from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.Orc_agent.State.state import AgentState
from src.Orc_agent.Node import Main_node 
from langgraph.graph import START
from src.Orc_agent.Graph.sub_graph import analyze_data,document_agent,generate_report

def create_main_graph():
    share_memory=MemorySaver()
    analyze_app = analyze_data.analyze_data_graph(share_memory)
    document_app = document_agent.document_agent_graph(share_memory)
    report_app = generate_report.generate_report_graph(share_memory)
    
    main_workflow = StateGraph(AgentState)
    main_workflow.add_node("File_type",Main_node.file_type)
    main_workflow.add_node("File_analysis",Main_node.file_analyze(document_app))
    main_workflow.add_node("Preprocessing",Main_node.preprocessing)#아직 추가 x
    main_workflow.add_node("Analysis",Main_node.analysis(analyze_app))
    main_workflow.add_node("Wait",Main_node.human_review_wait)
    main_workflow.add_node("Final_report",Main_node.final_report(report_app))
    main_workflow.add_edge(START,"File_type")
    main_workflow.add_edge("File_analysis",END)
    main_workflow.add_edge("Preprocessing","Analysis")
    main_workflow.add_edge("Analysis","Final_report")
    main_workflow.add_edge("Final_report","Wait")
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
    main_app = main_workflow.compile(checkpointer=share_memory,interrupt_before=["Wait"])
    return main_app,{
        "analyze": analyze_app,
        "document": document_app,
        "report": report_app
    }