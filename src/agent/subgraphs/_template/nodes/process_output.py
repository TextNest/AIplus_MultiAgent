"""
출력 처리 노드
=============

서브그래프의 마지막 노드로, 결과를 정리하여 출력합니다.
"""

from ..state import TemplateState


def process_output(state: TemplateState) -> TemplateState:
    """
    중간 결과를 최종 출력으로 변환합니다.
    
    Args:
        state: 현재 서브그래프 상태
        
    Returns:
        업데이트된 상태 (output_result 포함)
    """
    intermediate = state.get("intermediate_result")
    
    # -------------------------------------------------------------------------
    # 출력 생성 (TODO: 실제 로직으로 교체)
    # -------------------------------------------------------------------------
    if intermediate is None:
        return {
            "output_result": "ERROR: No intermediate result",
            "steps_log": ["[ProcessOutput] ERROR: No intermediate result"]
        }
    
    try:
        # 예시: 중간 결과를 그대로 출력으로 사용
        output = f"Processed: {intermediate}"
        
        return {
            "output_result": output,
            "output_metadata": {
                "length": len(output),
                "retry_count": state.get("retry_count", 0)
            },
            "steps_log": ["[ProcessOutput] Generated output successfully"]
        }
        
    except Exception as e:
        return {
            "output_result": f"ERROR: {str(e)}",
            "steps_log": [f"[ProcessOutput] ERROR: {str(e)}"]
        }
