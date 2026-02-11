"""
Docstring for agent.subgraphs.generate_report.supervisor
- LLM을 사용해 사용자의 요청이나 설정을 해석
- 어떤 작업자를 부를지 결정
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from state import ReportState

# 수퍼바이저의 결정 구조체
class SupervisorDecision(BaseModel):
    next: Literal["create_pdf", "create_html", "create_pptx", "FINISH"] = Field(
        description="다음으로 실행할 작업자 이름. 작업이 모두 끝나면 FINISH."
    )
def report_supervisor(state: ReportState):
    # 설정이나 사용자 피드백에서 포맷 확인
    report_format = state.get("report_format", "markdown")
    
    # 간단한 로직 또는 LLM 사용
    # 예: 이미 수행한 작업인지 체크하여 종료하거나, 포맷에 맞는 작업자 호출
    
    if "PDF" in report_format.upper():
        return {"next_worker": "create_pdf"}
    elif "HTML" in report_format.upper():
        return {"next_worker": "create_html"}
    # ...
    
    return {"next_worker": "FINISH"}