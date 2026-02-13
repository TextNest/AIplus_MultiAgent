"""
Analyze Data Node (서브그래프 함수들)
담당: 정국호

역할: LLM을 활용하여 데이터 분석 수행 (Plan → Make → Run → Eval → Wait 루프)
"""
import re
import pandas as pd
from ..state import analyzeState
from ..tools import get_python_repl
from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session
from ...core.df_summary import get_df_summary
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig


class MakeCodeOutput(BaseModel):
    code: str = Field(description="실행 가능한 파이썬 분석 코드. 설명이나 사족은 절대 포함하지 마세요.")


## INPUT : "prepared_data": 전처리 데이터 파일 경로 , "make_insight": 인사이트 생성 여부 및 개수 , "user_query": 사용자 질문

def plan_analysis_code(state: analyzeState, config: RunnableConfig) -> analyzeState:
    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    file_path = state.get("prepared_data", "")
    df = pd.read_csv(file_path)
    df_summary = get_df_summary(df)
    prompt = f"""
당신은 마케팅 데이터 전략가입니다. 제공된 데이터프레임의 요약 정보를 바탕으로 사용자의 질문에 답하기 위한 최적의 분석 시나리오를 설계하고 코드를 작성하세요.

    [데이터 정보]: {df_summary}
    [사용자 질문]: {state['user_query']}

    위 데이터를 바탕으로 분석 계획을 세우세요. 
    - 어떤 KPI(ROAS, CTR 등)를 계산할 것인가?
    - 어떤 시각화(막대그래프, 선그래프 등)가 필요한가?
    - 단계별 분석 순서를 나열하세요.
    *주의: 파이썬 코드는 작성하지 말고 오직 '계획'만 작성하세요.*
    """
    llm, callbacks = LLMFactory.create('google', 'gemma-3-27b-it')

    with langfuse_session(session_id=s_id, user_id=u_id, tags=["analyze_data", "plan"]) as lf_metadata:
        response = llm.invoke(prompt, config={'callbacks': callbacks, 'metadata': lf_metadata})

    plan = response.content
    return {"plan": plan, "df_summary": df_summary}


def make_analysis_code(state: analyzeState, config: RunnableConfig) -> analyzeState:
    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    llm, callbacks = LLMFactory.create('anthropic', 'claude-3-5-sonnet')

    if state.get("feed_back", None):
        text = f"수정사항: {state['feed_back']}"
    elif state.get("now_log", None):
        text = f"오류 및 수정사항 :{state['now_log']}"
    else:
        text = ""
    structured_llm = llm.with_structured_output(MakeCodeOutput)
    prompt = f"""
    분석 계획: {state['plan']}
    데이터 요약: {state['df_summary']}
    {text}
    위 분석 계획을 확인하고 실행하기 위한 파이썬 코드를 작성하세요. 
    이미지 저장 경로는 이후에 제공될 예정이니, 이미지 저장 코드는 포함하지 마세요.
    - pandas, matplotlib, seaborn 라이브러리를 사용하세요.
    """
    with langfuse_session(session_id=s_id, user_id=u_id, tags=["analyze_data", "make_code"]) as lf_metadata:
        response = structured_llm.invoke(prompt, config={'callbacks': callbacks, 'metadata': lf_metadata})
    code = response.code
    return {"code": code}


def run_code(state: analyzeState) -> analyzeState:
    code = state.get("code", "")
    num = state.get("make_insight", 0)
    roop = str(state.get("roop_back", 0))
    img_path = f"insght_{num}_{roop}.png"
    img_code = f"plt.savefig('{img_path}', dpi=300, bbox_inches='tight') \nplt.show() "
    full_code = code + "\n" + img_code
    repl = get_python_repl()
    try:
        result = repl.run(full_code)
        return {"result_summary": result, "result_img_path": img_path, "now_log": None}
    except Exception as e:
        return {
            "now_log": str(e),
            "error_roop": state.get("error_roop", 0) + 1
        }


def evaluation_code(state: analyzeState, config: RunnableConfig):
    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    prompt = f"""
    당신은 마케팅 분석 검증 전문가(LLM-as-a-judge)입니다.
    [분석 계획]: {state['plan']}
    [실행 결과]: {state['result_summary']}

    위 결과가 계획대로 도출되었으며, 수치가 논리적으로 타당한지 검증하세요.
    결과가 타당하면 'APPROVE', 부족하거나 오류가 보이면 'REJECT'와 이유를 적으세요.
    """
    llm, callbacks = LLMFactory.create('google', 'gemma-3-27b-it')

    with langfuse_session(session_id=s_id, user_id=u_id, tags=["analyze_data", "eval"]) as lf_metadata:
        response = llm.invoke(prompt, config={'callbacks': callbacks, 'metadata': lf_metadata})

    if "APPROVE" in response.content:
        return {"is_approved": True}
    else:
        return {"is_approved": False, "now_log": response.content}


def route_wait_node(state: analyzeState):
    pass


def router_error(state: analyzeState):
    if state.get("error_roop", 0) >= 3:
        raise Exception("분석 코드 실행이 3회 연속 실패했습니다. 코드를 검토해주세요.")
    elif state.get("now_log", None):
        return "Make"
    else:
        return "Eval"


def router_Eval(state: analyzeState):
    if state.get("is_approved", False):
        return "Wait"
    else:
        return "Make"


def router_next_step(state: analyzeState):
    choice = state.get("user_choice", "완료")
    if choice == "수정":
        return "Make"
    elif choice == "추가":
        return "Create", {"make_insight": state.get("make_insight", 0) + 1}
    elif choice == "완료":
        return "End"
    else:
        pass



