
import re
import pandas as pd
from ..state import analyzeState
from ..tools import get_python_repl
<<<<<<< HEAD
from ...core.llm_factory import LLMFactory, langfuse_session
from ...core.df_summary import get_df_summary
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
class MakeCodeOutput(BaseModel):
    code:str= Field(description="실행 가능한 파이썬 분석 코드. 설명이나 사족은 절대 포함하지 마세요.")
=======
from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session
>>>>>>> 5c3f6e80615da8373214b7fe2704ad781b868628

## INPUT : "prepared_data": 전처리 데이터 파일 경로(안전 제일 주의) , "make_insight": 인사이트 생성 여부 및 개수 , "user_query":사용자 질문

<<<<<<< HEAD
def plan_analysis_code(state:analyzeState , config:RunnableConfig)-> analyzeState:
    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    file_path = state.get("prepared_data","")
    df = pd.read_csv(file_path)
    df_summary = get_df_summary(df)
    prompt =  f"""
당신은 마케팅 데이터 전략가입니다. 제공된 데이터프레임의 요약 정보를 바탕으로 사용자의 질문에 답하기 위한 최적의 분석 시나리오를 설계하고 코드를 작성하세요.

 
    [데이터 정보]: {df_summary}
    [사용자 질문]: {state['user_query']}
=======
def analyze_data(state: AgentState) -> AgentState:
    """
    LLM이 생성한 Python 코드로 데이터를 분석합니다.
    """
    clean_data = state.get("clean_data")
    retry_count = state.get("retry_count", 0)
    previous_feedback = state.get("evaluation_feedback")
    wf_session_id = state.get("session_id", "unknown")
    
    if clean_data is None:
        return {
            "analysis_results": ["ERROR: No clean data available"],
            "steps_log": ["[Analyze] ERROR: No clean data to analyze"],
            "retry_count": retry_count
        }
>>>>>>> 5c3f6e80615da8373214b7fe2704ad781b868628
    
    위 데이터를 바탕으로 분석 계획을 세우세요. 
    - 어떤 KPI(ROAS, CTR 등)를 계산할 것인가?
    - 어떤 시각화(막대그래프, 선그래프 등)가 필요한가?
    - 단계별 분석 순서를 나열하세요.
    *주의: 파이썬 코드는 작성하지 말고 오직 '계획'만 작성하세요.*
    """
    llm, callbacks = LLMFactory.create('google', 'gemma-3-27b-it')
        
    with langfuse_session(session_id=s_id, user_id=u_id):
        response = llm.invoke(prompt, config={'callbacks': callbacks})

    plan = response.content
    return {"plan":plan , "df_summary":df_summary}

def make_analysis_code(state:analyzeState,config:RunnableConfig)-> analyzeState:
    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    llm, callbacks = LLMFactory.create('anthropic', 'claude-3-5-sonnet')
        
    if state.get("feed_back",None):
        text = f"수정사항: {state['feed_back']}"
    elif state.get("now_log",None):
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
    with langfuse_session(session_id=s_id, user_id=u_id):
        response = structured_llm.invoke(prompt, config={'callbacks': callbacks})
    code = response.code
    return {"code": code}

def run_code(state:analyzeState)->analyzeState:
    code = state.get("code","")
    num = state.get("make_insight",0)
    roop = str(state.get("roop_back", 0))
    img_path = f"insght_{num}_{roop}.png"
    img_code = f"plt.savefig('{img_path}', dpi=300, bbox_inches='tight') \nplt.show() "
    full_code = code + "\n" + img_code
    repl = get_python_repl()
    try:
<<<<<<< HEAD
        result = repl.run(full_code)
        return {"result_summary": result, "result_img_path": img_path,"now_log":None}
=======
        df = pd.DataFrame(clean_data)
        repl = get_python_repl()
        
        # =====================================================================
        # 구현 예시 시작 (TODO: 팀원이 수정/확장)
        # =====================================================================
        
        # Step 1: LLM 생성
        llm, callbacks = LLMFactory.create(
            provider="google",
            model="gemma-3-27b-it",
            temperature=0,  # 코드 생성은 결정적으로
        )
        
        # Step 2: REPL에 데이터 로드 (분석 코드가 df에 접근 가능하도록)
        repl.run(f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({clean_data})
print(f"Data loaded: {{len(df)}} rows, {{len(df.columns)}} columns")
""")
        
        # Step 3: 프롬프트 구성
        # 재시도인 경우 이전 피드백 포함
        feedback_context = ""
        if previous_feedback and "REJECT" in previous_feedback:
            feedback_context = f"""
## 이전 분석 피드백 (개선 필요)
{previous_feedback}

위 피드백을 반영하여 분석을 개선하세요.
"""
        
        prompt = f"""
당신은 데이터 분석 전문가입니다. Python 코드를 생성하여 데이터를 분석하세요.

## 데이터 정보
- 행 수: {len(df)}
- 컬럼: {list(df.columns)}
- 데이터 타입:
{df.dtypes.to_string()}

## 데이터 샘플 (처음 5행)
{df.head().to_string()}

## 기초 통계
{df.describe().to_string()}
{feedback_context}
## 요청 사항
1. 이 데이터에 대해 의미있는 분석을 수행하세요.
2. 기초 통계, 상관관계, 주요 패턴 등을 분석하세요.
3. 분석 결과를 print()로 출력하세요.

## 출력 형식
```python
# 여기에 분석 코드 작성
# 반드시 print()로 결과 출력
```

코드만 작성하고 설명은 하지 마세요.
"""

        # Step 4: LLM 호출 (Langfuse 세션으로 추적)
        with langfuse_session(
            session_id=wf_session_id,
            tags=["analyze_data", f"attempt_{retry_count + 1}"]
        ) as lf_metadata:
            response = llm.invoke(prompt, config={
                "callbacks": callbacks,
                "metadata": lf_metadata,
            })
        
        # Step 5: 응답에서 코드 추출
        content = response.content
        if isinstance(content, list):
            content = "".join([str(part) for part in content])
        
        generated_code = _extract_python_code(content)
        
        # Step 6: 코드 실행
        execution_result = repl.run(generated_code)
        
        # Step 7: 결과 반환
        analysis_result = f"""
## 생성된 분석 코드
```python
{generated_code}
```

## 실행 결과
{execution_result}
"""
        
        return {
            "analysis_results": [analysis_result],
            "steps_log": [f"[Analyze] Completed analysis (attempt {retry_count + 1})"],
            "retry_count": retry_count + 1
        }
        
        # =====================================================================
        # 구현 예시 끝
        # =====================================================================
        
>>>>>>> 5c3f6e80615da8373214b7fe2704ad781b868628
    except Exception as e:
        return {
            "now_log": str(e), 
            "error_roop": state.get("error_roop",0)  + 1
        }
    
def evaluation_code(state: analyzeState,config:RunnableConfig):
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
        
    with langfuse_session(session_id=s_id, user_id=u_id):
        response = llm.invoke(prompt, config={'callbacks': callbacks})
    
    if "APPROVE" in response.content:
        return {"is_approved": True}
    else:
        return {"is_approved": False, "now_log": response.content}


def route_wait_node(state: analyzeState):
    pass

def router_error(state: analyzeState):

    if state.get("error_roop",0) >=3:
        raise Exception("분석 코드 실행이 3회 연속 실패했습니다. 코드를 검토해주세요.")
    elif state.get("now_log",None):
        return "Make"

    else:
        return "Eval" 

def router_Eval(state: analyzeState):
    if state.get("is_approved", False):
        return "Wait"
    else:
        return "Make"

def router_next_step(state: analyzeState):
    choice = state.get("user_choice","완료")
    if choice == "수정":
        return "Make"
    elif choice == "추가":
        return "Create",{"make_insight": state.get("make_insight",0) +1}
    elif choice == "완료":
        return "End"
    else:
        pass



