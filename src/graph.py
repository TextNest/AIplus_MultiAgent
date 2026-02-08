from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.state import AgentState
from src.agent.nodes import (
    load_data,
    preprocess_data,
    analyze_data,
    evaluate_results,
    generate_report,
    human_review,
)

# Configuration
MAX_ANALYSIS_RETRIES = 3  # Maximum number of analysis retry attempts

def should_continue_analysis(state: AgentState):
    feedback = state.get('evaluation_feedback')
    retry_count = state.get('retry_count', 0)
    
    # Force approve if max retries reached
    if retry_count >= MAX_ANALYSIS_RETRIES:
        return "generate_report"
    
    if feedback == "APPROVE":/ㄷ
        return "generate_report"
    else:
        return "analyze_data"

def should_continue_human(state: AgentState):
    feedback = state.get('human_feedback')
    if feedback and "APPROVE" in feedback.upper():
        return END
    else:
        return "analyze_data"

def create_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("load_data", load_data)
    workflow.add_node("preprocess_data", preprocess_data)
    workflow.add_node("analyze_data", analyze_data)
    workflow.add_node("evaluate_results", evaluate_results)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("human_review", human_review)
    
    # Set Entry Point
    workflow.set_entry_point("load_data")
    
    # Add Edges
    workflow.add_edge("load_data", "preprocess_data")
    workflow.add_edge("preprocess_data", "analyze_data")
    workflow.add_edge("analyze_data", "evaluate_results")
    
    # Conditional Edge from Evaluation
    workflow.add_conditional_edges(
        "evaluate_results",
        should_continue_analysis,
        {
            "analyze_data": "analyze_data",
            "generate_report": "generate_report"
        }
    )
    
    workflow.add_edge("generate_report", "human_review")
    
    # Conditional Edge from Human Review
    workflow.add_conditional_edges(
        "human_review",
        should_continue_human,
        {
            "analyze_data": "analyze_data",
            "__end__": END
        }
    )
    
    # Compile with Checkpointer for persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory, interrupt_before=["human_review"])
    return app
