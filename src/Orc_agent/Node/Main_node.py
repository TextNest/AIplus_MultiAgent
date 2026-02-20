
from ..State.state import AgentState
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
class MakeCodeOutput(BaseModel):
    code:str= Field(description="실행 가능한 파이썬 분석 코드. 설명이나 사족은 절대 포함하지 마세요.")
from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session


def file_analyze(sub_app):
    def file_analyze_node(state: AgentState, config: RunnableConfig):
        sub_input ={
            "file_path":state["file_path"]
        }
        result = sub_app.invoke(sub_input,config=config)
        return{
            "result_summary":result["analysis_summary"]
        }
    return file_analyze_node

# def preprocessing(sub_app):
#     def preprocessing_node(state: AgentState, config: RunnableConfig):
#         sub_input ={

#         }
#         result = sub_app.invoke(sub_input,config=config)
#         return{
#         }
#     return preprocessing_node


def preprocessing(state: AgentState, config: RunnableConfig):
    # 아직 구현 안 됨 (Pass)
    return {"steps_log": ["Preprocessing skipped (Not implemented yet)"]}

def analysis(sub_app):
    def analysis_node(state: AgentState, config: RunnableConfig):
        sub_input ={
            "preprocessing_data":state["file_path"],
            "user_query":state["user_query"]
        }
        result = sub_app.invoke(sub_input,config=config)
        return{
            "analysis_results":result["final_insight"]
        }
    return analysis_node

def human_review(sub_app):
    def human_review_node(state: AgentState, config: RunnableConfig):
        sub_input ={

        }
        result = sub_app.invoke(sub_input,config=config)
        return{
        }
    return human_review_node
# def final_report(state: AgentState, config: RunnableConfig):
#     # 아직 구현 안 됨 (Pass)
#     return {"steps_log": ["Final report skipped (Not implemented yet)"]}

def final_report(sub_app):
    def final_report_node(state: AgentState, config: RunnableConfig):
        sub_input ={

        }
        result = sub_app.invoke(sub_input,config=config)
        return{
        }
    return final_report_node

def determine_format(state: AgentState, config: RunnableConfig):
    user_query = state.get("user_query", "").lower()
    
    # 기본값 설정
    detected_format = "markdown"
    
    # 키워드 감지
    if "pdf" in user_query:
        detected_format = "pdf"
    elif "html" in user_query:
        detected_format = "html"
    elif "ppt" in user_query or "pptx" in user_query:
        detected_format = "pptx"
        
    # 상태 업데이트
    return {"report_format": detected_format}


def file_type(state:AgentState,config:RunnableConfig)->AgentState:
    file_path = state["file_path"]
    file_type = file_path.split(".")[-1]
    if file_type == "csv":
        return {
            "file_type":"tabular"
        }
    else:
        return {
            "file_type":"document"
        }

def select_agent(state:AgentState,config:RunnableConfig)->AgentState:
    file_type = state["file_type"]
    if file_type == "tabular":
        return "tabular"
    else:
        return "document"

def human_review_wait(state:AgentState,config:RunnableConfig)->AgentState:
    pass

def human_review_route(state:AgentState,config:RunnableConfig)->AgentState:
    if state["human_feedback"] == "APPROVE":
        return "END"
    else:
        return "analysis"