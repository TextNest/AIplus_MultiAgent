"""
입력 처리 노드
=============

서브그래프의 첫 번째 노드로, 입력 데이터를 검증하고 처리합니다.
"""

from ..state import TemplateState


def process_input(state: TemplateState) -> TemplateState:
    """
    입력 데이터를 처리합니다.
    
    Args:
        state: 현재 서브그래프 상태
        
    Returns:
        업데이트된 상태 (변경된 필드만 반환)
    
    주의:
        - 전체 state를 반환하지 말고, 변경된 필드만 dict로 반환하세요.
        - LangGraph가 자동으로 기존 state에 병합합니다.
    """
    input_data = state.get("input_data")
    retry_count = state.get("retry_count", 0)
    
    # -------------------------------------------------------------------------
    # 입력 검증
    # -------------------------------------------------------------------------
    if input_data is None:
        return {
            "steps_log": ["[ProcessInput] ERROR: No input data"],
            "output_result": "ERROR: No input data"
        }
    
    # -------------------------------------------------------------------------
    # 처리 로직 (TODO: 실제 로직으로 교체)
    # -------------------------------------------------------------------------
    try:
        # 예시: 입력 데이터를 문자열로 변환
        intermediate = str(input_data)
        
        return {
            "intermediate_result": intermediate,
            "retry_count": retry_count + 1,
            "steps_log": [f"[ProcessInput] Processed input (attempt {retry_count + 1})"]
        }
        
    except Exception as e:
        return {
            "steps_log": [f"[ProcessInput] ERROR: {str(e)}"],
            "retry_count": retry_count + 1
        }
