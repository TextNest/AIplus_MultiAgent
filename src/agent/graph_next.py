# src/agent/graph_next.py
from typing import TypedDict, Annotated, List, Optional, Literal, Dict, Any
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# src/agent/factory.py (삭제됨 - 직접 import 사용)
# from src.agent.subgraphs.data_prep.graph import create_prep_graph
# from src.agent.subgraphs.data_analysis.graph import create_analysis_graph
from src.agent.state import AgentState  # Main State

# [가상의 Subgraph State - 실제로는 각 폴더에서 import]
# from src.agent.subgraphs.data_prep.state import PrepState
# from src.agent.subgraphs.data_analysis.state import AnalysisState

# ---------------------------------------------------------------------------
# 1. Wrapper Function (Closure Pattern)
# ---------------------------------------------------------------------------

def run_prep_agent_wrapper(sub_app):
    """
    [Data Prep] 서브그래프를 호출하는 래퍼 노드
    """
    def node_func(state: AgentState, config: RunnableConfig):
        # 1. State Mapping (Main -> Sub)
        # Main State에 있는 정보를 Prep State에 맞게 변환
        sub_input = {
            "raw_data_path": state.get("file_path"),
            "query": state.get("user_query", ""),
            # 필요한 다른 초기값들...
        }
        
        # 2. Invoke Subgraph (Config 전달 필수!)
        # config를 전달해야 langfuse 등의 메타데이터가 유지됨
        print(f"[Main Graph] Invoking Data Prep Subgraph...")
        result = sub_app.invoke(sub_input, config=config)
        
        # 3. Result Mapping (Sub -> Main)
        # 서브그래프 결과를 Main State에 업데이트
        return {
            "clean_data": result.get("processed_data"),
            "steps_log": [f"[Prep] 완료: {result.get('summary', '')}"]
        }
        
    return node_func

def run_analysis_agent_wrapper(sub_app):
    """
    [Data Analysis] 서브그래프를 호출하는 래퍼 노드
    """
    def node_func(state: AgentState, config: RunnableConfig):
        # 1. State Mapping (Main -> Sub)
        # 전처리된 데이터가 메인 State에 있다고 가정
        sub_input = {
            "prepared_data": state.get("clean_data"),
            "code": "", # 초기화
            "make_insight": 0,
            # ...
        }
        
        # 2. Invoke Subgraph
        print(f"[Main Graph] Invoking Data Analysis Subgraph...")
        child_result = sub_app.invoke(sub_input, config=config)
        
        # 3. Result Mapping (Sub -> Main)
        return {
            "analysis_results": child_result.get("insight_analysis", []),
            "steps_log": ["Analysis Completed"]
        }
    
    return node_func


# ---------------------------------------------------------------------------
# 2. Nodes (Main Graph 전용)
# ---------------------------------------------------------------------------

def create_data_node(state: AgentState):
    # 파일 경로 확인 등 초기화 작업
    print(f"[Main Graph] Checking data path: {state.get('file_path')}")
    return {"steps_log": ["Data check passed"]}

def route_main_node(state: AgentState):
    # 사용자 결정 또는 자동 분기
    # 예: 분석 결과가 부족하면 다시 분석으로 돌리거나 등등
    return "write_report"

def write_report_node(state: AgentState):
    # 최종 보고서 작성
    print("[Main Graph] Writing Final Report...")
    return {"final_report": "최종 보고서 내용..."}


# ---------------------------------------------------------------------------
# 3. Graph Construction
# ---------------------------------------------------------------------------

def create_graph_v2():
    # 1. 서브그래프 앱 로드 (Factory 사용)
    # 실제 구현 시에는 Factory가 컴파일된 앱을 반환해야 함
    # prep_app = SubgraphFactory.get_subgraph("data_prep")
    # analysis_app = SubgraphFactory.get_subgraph("data_analysis")
    
    # [DEMO용 Mock App] - 실제로는 Factory에서 받아옴
    class MockApp:
        def invoke(self, input, config=None):
            return {"processed_data": {"mock": "data"}, "summary": "Done", "insight_analysis": ["Great Insight"]}
    
    prep_app = MockApp()
    analysis_app = MockApp()

    # 2. 메인 워크플로우 정의
    workflow = StateGraph(AgentState)

    # 3. 노드 추가 (Wrapper 사용!)
    workflow.add_node("CreateData", create_data_node)
    
    # 래퍼 함수를 통해 서브그래프를 노드로 등록
    workflow.add_node("RunA", run_prep_agent_wrapper(prep_app))
    workflow.add_node("RunB", run_analysis_agent_wrapper(analysis_app))
    
    workflow.add_node("WriteReport", write_report_node)

    # 4. 엣지 연결
    workflow.add_edge(START, "CreateData")
    workflow.add_edge("CreateData", "RunA")  # 데이터 확인 -> 전처리
    workflow.add_edge("RunA", "RunB")        # 전처리 -> 분석
    workflow.add_edge("RunB", "WriteReport") # 분석 -> 보고서
    workflow.add_edge("WriteReport", END)

    # 5. 컴파일
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app
