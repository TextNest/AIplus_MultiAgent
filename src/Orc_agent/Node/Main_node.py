from typing import Any, Optional

from ..State.state import AgentState
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session, observe
from src.Orc_agent.core.logger import logger

class MakeCodeOutput(BaseModel):
    code:str= Field(description="실행 가능한 파이썬 분석 코드. 설명이나 사족은 절대 포함하지 마세요.")

def file_analyze(sub_app):
    @observe(name="file_analyze")
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


@observe(name="preprocessing")
def preprocessing(state: AgentState, config: RunnableConfig):
    # 아직 구현 안 됨 (Pass)
    logger.info("Preprocessing skipped (Not implemented yet)")
    return {"steps_log": ["Preprocessing skipped (Not implemented yet)"]}

def analysis(sub_app):
    @observe(name="analysis")
    def analysis_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        logger.info(f">>> [분석 노드] 서브그래프 상태 확인 중...")
        configurable = config.get("configurable", {})
        parent_thread_id = configurable.get("thread_id", "main")
        sub_thread_id = f"{parent_thread_id}_sub"
        sub_config = config.copy()
        sub_config["configurable"] = {
            "thread_id": sub_thread_id,
            "user_id": configurable.get("user_id"),
            "session_id": configurable.get("session_id"),
        }

        snapshot = sub_app.get_state(sub_config)
        logger.info(f">>> [분석 노드] 다음 서브그래프: {snapshot.next}")
        
        result: Optional[dict[str, Any]] = None
        
        if snapshot.next:
            logger.info(f">>> [분석 노드]  {snapshot.next} 부터 다시 시작합니다.")
            
            for chunk in sub_app.stream(None, config=sub_config, stream_mode="values"):
                result = chunk
                logger.info(f">>> [분석 노드] Step: {list(chunk.keys())}")
                if "now_log" in chunk and chunk["now_log"]:
                    logger.info(f"    [에러/로그]: {chunk['now_log']}")
        
        else:
            logger.info(f">>> [분석 노드] 새로운 서브그래프 시작")
            sub_input = {
                "preprocessing_data": state["file_path"],
                "user_query": state["user_query"],
                "feed_back": [],
            }
            
            for chunk in sub_app.stream(sub_input, config=sub_config, stream_mode="values"):
                result = chunk
                logger.info(f">>> [분석 노드] Step: {list(chunk.keys())}")
                if "now_log" in chunk and chunk["now_log"]:
                    logger.info(f"    [에러/로그]: {chunk['now_log']}")
                if "code" in chunk:
                    logger.info(f"    [코드생성]: {chunk['code'][:50]}...")
        
        final_snapshot = sub_app.get_state(sub_config)
        
        if final_snapshot.next:
            logger.info(f">>> [분석 노드] 서브그래프가 {final_snapshot.next} 에서 일시정지되었습니다. 현재 결과를 사용해 계속 진행합니다.")
            if final_snapshot.next != ("Wait",):
                return {
                    "analysis_results": {
                        "error": f"Unexpected subgraph pause: {final_snapshot.next}"
                    },
                    "figure_list": result.get("result_img_paths", []) if result else []
                }
        
        if result and "final_insight" in result:
             logger.info(f">>> [분석 노드] 서브그래프가 성공적으로 종료되었습니다.")
             return {
                "analysis_results": result.get("final_insight", {}),
                "figure_list": result.get("result_img_paths", [])
            }
        else:
             logger.info(f">>> [분석 노드] 서브그래프가 종료되었으나 결과를 찾을 수 없습니다.")
             return {
                 "analysis_results": {},
                 "figure_list": []
             }

    return analysis_node

def final_report(sub_app):
    @observe(name="final_report")
    def final_report_node(state: AgentState, config: RunnableConfig):
        insights = state.get("analysis_results") or {}
        
        # Log for debugging
        logger.info(f"Passing insights to report subgraph: {list(insights.keys())}")

        sub_input = {
            "analysis_results": insights,
            "figure_list": state.get("figure_list", []),
            "file_path": state.get("file_path", ""),
            "report_format": state.get("report_type", ["markdown"]),
            "clean_data": state.get("clean_data")
        }

        logger.info(f">>> [최종리포트 노드] 서브그래프 상태 확인 중...")
        result = sub_app.invoke(sub_input, config=config)
        logger.info(f">>> [최종리포트 노드] 서브그래프가 성공적으로 종료되었습니다.")
        return {
            "final_report": result.get("final_report"),
            "steps_log": result.get("steps_log", [])
        }
    return final_report_node




def file_type(state: AgentState, config: RunnableConfig) -> dict[str, str]:
    file_path = state["file_path"]
    file_type = file_path.split(".")[-1]
    if file_type == "csv":
        logger.info(f">>> [파일타입 노드] 파일 타입: {file_type}")
        return {
            "file_type":"tabular"
        }
    else:
        logger.info(f">>> [파일타입 노드] 파일 타입: {file_type}")
        return {
            "file_type":"document"
        }

def select_agent(state: AgentState, config: RunnableConfig) -> str:
    file_type = state["file_type"]
    if file_type == "tabular":
        logger.info(f">>> [에이전트선택 노드] 에이전트 타입: {file_type}")
        return "tabular"
    else:
        logger.info(f">>> [에이전트선택 노드] 에이전트 타입: {file_type}")
        return "document"

def human_review_wait(state: AgentState, config: RunnableConfig) -> None:
    logger.info(f">>> [휴먼리뷰 대기 노드] 서브그래프 상태 확인 중...")
    return None

def human_review_route(state: AgentState, config: RunnableConfig) -> str:
    if state["human_feedback"] == "APPROVE":
        logger.info(f">>> [휴먼리뷰 라우트 노드] 승인됨")
        return "final_report"
    else:
        logger.info(f">>> [휴먼리뷰 라우트 노드] 거절됨")
        return "analysis"
