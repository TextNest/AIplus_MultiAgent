from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.state import AgentState,analyzeState
from src.agent.nodes import (
    load_data,
    preprocess_data,
    analyze_document,
    evaluate_results,
    generate_report,
    human_review,
)

# Configuration
MAX_ANALYSIS_RETRIES = 3  # Maximum number of analysis retry attempts

def route_by_file_type(state: AgentState):
    file_type = state.get("file_type", "tabular")
    if file_type == "document":
        return "analyze_document"
    return "preprocess_data"

def should_continue_analysis(state: AgentState):
    feedback = state.get('evaluation_feedback')
    retry_count = state.get('retry_count', 0)
    
    # Force approve if max retries reached
    if retry_count >= MAX_ANALYSIS_RETRIES:
        return "generate_report"
    
    if feedback == "APPROVE":
        return "generate_report"
    else:
        file_type = state.get("file_type", "tabular")
        if file_type == "document":
            return "analyze_document"
        return "analyze_data"

def should_continue_human(state: AgentState):
    feedback = state.get('human_feedback')
    if feedback and "APPROVE" in feedback.upper():
        return END
    else:
        file_type = state.get("file_type", "tabular")
        if file_type == "document":
            return "analyze_document"
        return "analyze_data"

def _build_analyze_subgraph():
    """정국호의 analyze 서브그래프 (Plan → Make → Run → Eval → Wait 루프) 빌드"""
    from src.agent.nodes.analyze_data import (
        plan_analysis_code, make_analysis_code, run_code,
        evaluation_code, route_wait_node, router_error,
        router_Eval, router_next_step,
    )
    analyze_workflow = StateGraph(analyzeState)
    analyze_workflow.add_node("Plan", plan_analysis_code)
    analyze_workflow.add_node("Make", make_analysis_code)
    analyze_workflow.add_node("Run", run_code)
    analyze_workflow.add_node("Eval", evaluation_code)
    analyze_workflow.add_node("Wait", route_wait_node)
    analyze_workflow.add_edge(START, "Plan")
    analyze_workflow.add_edge("Plan", "Make")
    analyze_workflow.add_edge("Make", "Run")
    analyze_workflow.add_conditional_edges(
        "Run",
        router_error,
        {
            "Make": "Make",
            "Eval": "Eval"
        }
    )
    analyze_workflow.add_conditional_edges(
        "Eval",
        router_Eval,
        {
            "Wait": "Wait",
            "Make": "Make"
        }
    )
    analyze_workflow.add_conditional_edges(
        "Wait",
        router_next_step,
        {
            "Make": "Make",
            "Create": "Plan",
            END: END
        }
    )
    return analyze_workflow.compile(checkpointer=MemorySaver(), interrupt_before=["Wait"])


def _analyze_data_wrapper(state: AgentState) -> AgentState:
    """
    AgentState ↔ analyzeState 브릿지.
    메인 워크플로우의 file_path/clean_data를 서브그래프의 prepared_data로 매핑.
    """
    file_path = state.get("file_path", "")
    retry_count = state.get("retry_count", 0)

    # 서브그래프 입력 구성
    sub_input = {
        "prepared_data": file_path,
        "user_choice": "",
        "user_query": "이 데이터를 종합적으로 분석해주세요.",
        "code": "",
        "result_summary": "",
        "result_img_path": "",
        "feed_back": "",
        "now_log": "",
        "make_insight": 0,
        "roop_back": 0,
        "plan": "",
        "df_summary": "",
        "error_roop": 0,
        "is_approved": False,
    }

    try:
        analyze_app = _build_analyze_subgraph()
        import uuid
        thread = {"configurable": {"thread_id": f"analyze-{uuid.uuid4().hex[:8]}"}}
        sub_result = analyze_app.invoke(sub_input, thread)

        analysis_text = f"""
## 분석 계획
{sub_result.get('plan', 'N/A')}

## 실행 결과
{sub_result.get('result_summary', 'N/A')}
"""
        return {
            "analysis_results": [analysis_text],
            "steps_log": [f"[Analyze] Completed analysis (attempt {retry_count + 1})"],
            "retry_count": retry_count + 1,
        }
    except Exception as e:
        return {
            "analysis_results": [f"ERROR: {str(e)}"],
            "steps_log": [f"[Analyze] ERROR: {str(e)}"],
            "retry_count": retry_count + 1,
        }


def create_graph():
    memory = MemorySaver()

    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("load_data", load_data)
    workflow.add_node("preprocess_data", preprocess_data)
    workflow.add_node("analyze_data", _analyze_data_wrapper)
    workflow.add_node("analyze_document", analyze_document)
    workflow.add_node("evaluate_results", evaluate_results)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("human_review", human_review)
    
    # Set Entry Point
    workflow.set_entry_point("load_data")
    
    # Conditional Edge from Load Data
    workflow.add_conditional_edges(
        "load_data",
        route_by_file_type,
        {
            "preprocess_data": "preprocess_data",
            "analyze_document": "analyze_document",
        }
    )
    
    # Tabular Flow
    workflow.add_edge("preprocess_data", "analyze_data")
    workflow.add_edge("analyze_data", "evaluate_results")
    
    # Document Flow
    workflow.add_edge("analyze_document", "evaluate_results")
    
    # Conditional Edge from Evaluation
    workflow.add_conditional_edges(
        "evaluate_results",
        should_continue_analysis,
        {
            "analyze_data": "analyze_data",
            "analyze_document": "analyze_document",
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
            "analyze_document": "analyze_document",
            "__end__": END
        }
    )
    
    # Compile with Checkpointer for persistence
    
    app = workflow.compile(checkpointer=memory, interrupt_before=["human_review"])
    return app

