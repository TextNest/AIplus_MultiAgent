"""
Analyze Data Node
담당: [팀원 C]

역할: LLM을 활용하여 데이터 분석 수행
입력: clean_data
출력: analysis_results, steps_log, retry_count
"""
import os
import pandas as pd
from ..state import AgentState
from ..tools import get_python_repl
from ...core.llm_factory import LLMFactory


def analyze_data(state: AgentState) -> AgentState:
    """
    LLM이 생성한 Python 코드로 데이터를 분석합니다.
    
    TODO: 팀원이 구현해야 할 내용
    - LLM 프롬프트 설계
    - 분석 코드 생성 및 실행
    - 결과 파싱 및 저장
    
    Args:
        state: 현재 AgentState (clean_data 필수)
    
    Returns:
        AgentState: analysis_results와 steps_log가 업데이트된 상태
    """
    clean_data = state.get("clean_data")
    retry_count = state.get("retry_count", 0)
    previous_feedback = state.get("evaluation_feedback")
    
    if clean_data is None:
        return {
            "analysis_results": ["ERROR: No clean data available"],
            "steps_log": ["[Analyze] ERROR: No clean data to analyze"],
            "retry_count": retry_count
        }
    
    try:
        df = pd.DataFrame(clean_data)
        repl = get_python_repl()
        
        # TODO: LLM을 사용하여 분석 코드 생성
        # llm, callbacks = LLMFactory.create("google", "gemini-2.0-flash")
        # prompt = f"Analyze this data: {df.describe()}"
        # response = llm.invoke(prompt, config={"callbacks": callbacks})
        
        # TODO: 생성된 코드 실행
        # result = repl.run(generated_code)
        
        # 임시 구현: 기본 통계 분석
        analysis_result = f"Basic Statistics:\n{df.describe().to_string()}"
        
        return {
            "analysis_results": [analysis_result],
            "steps_log": [f"[Analyze] Completed analysis (attempt {retry_count + 1})"],
            "retry_count": retry_count + 1
        }
    except Exception as e:
        return {
            "analysis_results": [f"ERROR: {str(e)}"],
            "steps_log": [f"[Analyze] ERROR: {str(e)}"],
            "retry_count": retry_count + 1
        }
