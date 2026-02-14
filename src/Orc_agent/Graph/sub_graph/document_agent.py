from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from ...State.state import DocumentState
from ...Node.sub_node import document_agent
from langgraph.graph import START



def document_agent_graph(CheckPoint=None):
    document_workflow = StateGraph(DocumentState)
    document_workflow.add_node("Read", document_agent.read_file_node)
    document_workflow.add_node("Analyze", document_agent.analyze_doc_node)
    document_workflow.add_edge(START, "Read")
    document_workflow.add_edge("Read", "Analyze")
    document_workflow.add_edge("Analyze", END)
    document_app = document_workflow.compile(checkpointer=CheckPoint)
    return document_app