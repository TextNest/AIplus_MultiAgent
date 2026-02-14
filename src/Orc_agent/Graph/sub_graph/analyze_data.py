from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from ...State.state import analyzeState
from ...Node.sub_node import analyze_data
from langgraph.graph import START

def analyze_data_graph(CheckPoint=None):

    analyze_workflow = StateGraph(analyzeState)
    analyze_workflow.add_node("Plan", analyze_data.plan_analysis_code)
    analyze_workflow.add_node("Make", analyze_data.make_analysis_code)
    analyze_workflow.add_node("Run", analyze_data.run_code)
    analyze_workflow.add_node("Insight", analyze_data.derive_insight_node)
    analyze_workflow.add_node("Eval", analyze_data.evaluation_code)
    analyze_workflow.add_node("Wait", analyze_data.route_wait_node)
    analyze_workflow.add_edge(START, "Plan")
    analyze_workflow.add_edge("Plan", "Make")
    analyze_workflow.add_edge("Make", "Run")
    analyze_workflow.add_edge("Insight","Eval")
    analyze_workflow.add_conditional_edges(
        "Run",
        analyze_data.router_error,
        {
            "Make":"Make",
            "Insight":"Insight"
        }
    )
    analyze_workflow.add_conditional_edges(
        "Eval",
        analyze_data.router_Eval,
        {
            "Wait":"Wait",
            "Make":"Make"
        }
    )
    analyze_workflow.add_conditional_edges(
        "Wait",
        analyze_data.router_next_step,
        {
            "Make":"Make",
            "Create":"Plan",
            "END":END
        }
    )
    analyze_app = analyze_workflow.compile(checkpointer=CheckPoint,interrupt_before=["Wait"])
    return analyze_app