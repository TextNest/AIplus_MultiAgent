"""
서브그래프 정의
==============

이 파일에서 서브그래프의 워크플로우를 정의합니다.

구성 요소:
1. 조건부 라우팅 함수 (should_xxx)
2. 서브그래프 생성 함수 (create_xxx_subgraph)
3. 컴파일된 서브그래프 인스턴스
"""

from langgraph.graph import StateGraph, END
from .state import TemplateState
from .nodes import process_input, process_output


# =============================================================================
# 조건부 라우팅 함수
# =============================================================================

def should_continue(state: TemplateState) -> str:
    """
    다음 노드를 결정하는 라우팅 함수
    
    Returns:
        "continue": 다음 처리 단계로
        "finish": 종료
        "retry": 재시도 (루프)
    """
    retry_count = state.get("retry_count", 0)
    
    # 최대 재시도 횟수 초과
    if retry_count >= 3:
        return "finish"
    
    # 결과가 있으면 종료
    if state.get("output_result"):
        return "finish"
    
    # 그 외에는 재시도
    return "retry"


# =============================================================================
# 서브그래프 생성 함수
# =============================================================================

def create_template_subgraph():
    """
    서브그래프를 생성하고 컴파일합니다.
    
    Returns:
        CompiledGraph: 실행 가능한 서브그래프
    
    사용 예시:
        subgraph = create_template_subgraph()
        result = subgraph.invoke({
            "input_data": {"key": "value"},
            "input_config": None,
            "intermediate_result": None,
            "retry_count": 0,
            "steps_log": [],
            "output_result": None,
            "output_metadata": None
        })
    """
    workflow = StateGraph(TemplateState)
    
    # -------------------------------------------------------------------------
    # 노드 등록
    # -------------------------------------------------------------------------
    workflow.add_node("process_input", process_input)
    workflow.add_node("process_output", process_output)
    
    # -------------------------------------------------------------------------
    # 엣지 정의
    # -------------------------------------------------------------------------
    
    # 시작점 설정
    workflow.set_entry_point("process_input")
    
    # 순차 엣지
    workflow.add_edge("process_input", "process_output")
    
    # 조건부 엣지 (예시: 루프가 필요한 경우)
    workflow.add_conditional_edges(
        "process_output",
        should_continue,
        {
            "continue": "process_input",  # 다음 단계로
            "retry": "process_input",      # 재시도
            "finish": END                  # 종료
        }
    )
    
    # -------------------------------------------------------------------------
    # 컴파일 및 반환
    # -------------------------------------------------------------------------
    return workflow.compile()


# =============================================================================
# 컴파일된 서브그래프 인스턴스 (선택적)
# =============================================================================
# 매번 create_template_subgraph()를 호출하지 않고 싱글톤으로 사용할 수 있습니다.
# template_subgraph = create_template_subgraph()
