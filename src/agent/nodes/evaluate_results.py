"""
Evaluate Results Node
담당: [팀원 D]

역할: 분석 결과를 평가하여 승인/재시도 결정
입력: analysis_results, clean_data
출력: evaluation_feedback, steps_log
"""
import pandas as pd
from ..state import AgentState
from ...core.llm_factory import LLMFactory


def evaluate_results(state: AgentState) -> AgentState:
    """
    분석 결과의 품질을 평가하고 승인 여부를 결정합니다.
    
    TODO: 팀원이 구현해야 할 내용
    - LLM 기반 품질 평가 프롬프트 설계
    - 평가 기준 정의 (완전성, 정확성, 유용성)
    - APPROVE / REJECT 결정 로직
    
    Args:
        state: 현재 AgentState (analysis_results 필수)
    
    Returns:
        AgentState: evaluation_feedback와 steps_log가 업데이트된 상태
    """
    analysis_results = state.get("analysis_results", [])
    retry_count = state.get("retry_count", 0)
    
    if not analysis_results:
        return {
            "evaluation_feedback": "REJECT: No analysis results to evaluate",
            "steps_log": ["[Evaluate] REJECT: No results available"]
        }
    
    try:
        # TODO: LLM을 사용하여 분석 결과 평가
        # llm, callbacks = LLMFactory.create("google", "gemini-2.0-flash")
        # prompt = f"Evaluate this analysis: {analysis_results}"
        # response = llm.invoke(prompt, config={"callbacks": callbacks})
        
        # 임시 구현: 결과가 있으면 승인
        latest_result = analysis_results[-1] if analysis_results else ""
        
        if "ERROR" in latest_result:
            feedback = "REJECT: Analysis contains errors"
        else:
            feedback = "APPROVE"
        
        return {
            "evaluation_feedback": feedback,
            "steps_log": [f"[Evaluate] Decision: {feedback}"]
        }
    except Exception as e:
        return {
            "evaluation_feedback": f"REJECT: Evaluation error - {str(e)}",
            "steps_log": [f"[Evaluate] ERROR: {str(e)}"]
        }
