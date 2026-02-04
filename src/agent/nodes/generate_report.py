"""
Generate Report Node
담당: [팀원 E]

역할: 분석 결과를 기반으로 보고서 생성
입력: analysis_results, clean_data
출력: final_report, steps_log
"""
import pandas as pd
from ..state import AgentState
from ...core.llm_factory import LLMFactory


def generate_report(state: AgentState) -> AgentState:
    """
    분석 결과를 바탕으로 Markdown 보고서를 생성합니다.
    
    TODO: 팀원이 구현해야 할 내용
    - LLM 기반 보고서 생성 프롬프트 설계
    - Markdown 포맷팅
    - 차트/그래프 이미지 포함 (선택)
    - PPT/HTML 출력 지원 (확장)
    
    Args:
        state: 현재 AgentState (analysis_results 필수)
    
    Returns:
        AgentState: final_report와 steps_log가 업데이트된 상태
    """
    analysis_results = state.get("analysis_results", [])
    clean_data = state.get("clean_data")
    
    if not analysis_results:
        return {
            "final_report": "# Error\n\nNo analysis results available.",
            "steps_log": ["[Report] ERROR: No analysis results"]
        }
    
    try:
        # TODO: LLM을 사용하여 보고서 생성
        # llm, callbacks = LLMFactory.create("google", "gemini-2.0-flash")
        # prompt = f"Generate a report based on: {analysis_results}"
        # response = llm.invoke(prompt, config={"callbacks": callbacks})
        
        # 임시 구현: 기본 Markdown 보고서
        report = "# Data Analysis Report\n\n"
        report += "## Summary\n\n"
        report += "This report contains the analysis results.\n\n"
        report += "## Analysis Results\n\n"
        
        for i, result in enumerate(analysis_results, 1):
            report += f"### Result {i}\n\n"
            report += f"```\n{result}\n```\n\n"
        
        report += "## Conclusion\n\n"
        report += "TODO: Add conclusions based on analysis.\n"
        
        return {
            "final_report": report,
            "steps_log": ["[Report] Generated Markdown report"]
        }
    except Exception as e:
        return {
            "final_report": f"# Error\n\n{str(e)}",
            "steps_log": [f"[Report] ERROR: {str(e)}"]
        }
