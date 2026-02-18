
import re
import pandas as pd
from ...State.state import analyzeState

from ...core.df_summary import get_df_summary
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session
from langchain_core.messages import HumanMessage
import base64
import os

from ...core.observe import langfuse_session, observe
import  matplotlib
from src.Orc_agent.core.logger import logger
from src.Orc_agent.core.executor import executor_instance

class MakeCodeOutput(BaseModel):
    code:str= Field(description="실행 가능한 파이썬 분석 코드. 설명이나 사족은 절대 포함하지 마세요.")
    
## INPUT : "preprocessing_data": 전처리 데이터 파일 경로(안전 제일 주의) , "user_query":사용자 질문,"feed_back":피드백

@observe(name="Plan")
def plan_analysis_code(state:analyzeState , config:RunnableConfig)-> analyzeState:
    if state.get("user_choice")=="추가":
        roop_back = state.get("roop_back",0) +1
    else:
        roop_back = 0
    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    file_path = state.get("preprocessing_data","")
    df = pd.read_csv(file_path)
    df_summary = get_df_summary(df)
    prompt =  f"""
    당신은 마케팅 데이터 전략가입니다. 제공된 데이터프레임의 요약 정보를 바탕으로 사용자의 질문에 답하기 위한 최적의 분석 시나리오를 설계하고 코드를 작성하세요.
    
 
    [데이터 정보]: {df_summary}
    [사용자 질문]: {state['user_query']}
    [피드백]: {state['feed_back']}
    위 데이터를 바탕으로 분석 계획을 세우세요. 
    - 어떤 KPI(ROAS, CTR 등)를 계산할 것인가?
    - 어떤 시각화(막대그래프, 선그래프 등)가 필요한가?
    - 단계별 분석 순서를 나열하세요.
    *주의: 파이썬 코드는 작성하지 말고 오직 '계획'만 작성하세요.*
    이미지 파일은 최대 3개만 만들 수 있도록 계획을 구축하세요. 다만 각각의 이미지 파일은 하나의 그래프 또는 표만 들어가야합니다.
    """
    llm, callbacks = LLMFactory.create('openai', 'gpt-5-nano', temperature=0.3)
        
    with langfuse_session(session_id=s_id, user_id=u_id):
        response = llm.invoke(prompt, config={'callbacks': callbacks})

    plan = response.content
    return {"plan":plan , "df_summary":df_summary,"roop_back":roop_back,"error_roop": 0}

@observe(name="Make")
def make_analysis_code(state:analyzeState,config:RunnableConfig)-> analyzeState:
    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    llm, callbacks = LLMFactory.create('openai', 'gpt-5.2')
        
    if state.get("feed_back",None):
        text = f"수정사항: {state['feed_back']} 해당 수정사항을 반영하여 코드를 수정하세요"
    elif state.get("now_log",None):
        text = f"오류 및 수정사항 :{state['now_log']} 해당 오류가 발생 하지 않도록 수정을 진행하세요"
    else:
        text = ""
    file_path_raw = state.get("preprocessing_data", "")
    if file_path_raw:
        file_path = os.path.abspath(file_path_raw).replace("\\", "/")
    else:
        file_path = ""
    current_dir = os.getcwd().replace("\\", "/") 
    img_dir = f"{current_dir}/img"
    structured_llm = llm.with_structured_output(MakeCodeOutput)
    prompt = f"""
    분석 계획: {state['plan']}
    데이터 요약: {state['df_summary']}
    [데이터 파일 경로]: {file_path}
    {text} 
    위 분석 계획을 확인하고 실행하기 위한 파이썬 코드를 작성하세요. 
    
    [필수]
    - 변수명 앞에 _df 이렇게 작성하지마세요 추가적인 df가 필요하다면 copy1_df,copy2_df ... 이렇게 작성하세요 절대로 변수명 앞에 _ 사용하지 마세요.
    - 코드 시작 부분에서 반드시 데이터를 로드하세요: df = pd.read_csv(r'{file_path}') 
    - 설명이나 마크다운(```python ... ```) 없이 오직 파이썬 코드만 출력하세요. 
    - print 구문 사용 하지 마세요
    [이미지 저장 규칙]
    - 시각화가 필요한 경우, 각 그래프를 '{img_dir}/figure_{state.get("roop_back", 0)}_0.png', '{img_dir}/figure_{state.get("roop_back", 0)}_1.png', ... 와 같이 순서대로 저장하세요. ({state.get("make_insight", 0)}은 현재 인사이트 번호입니다.)
    - 반드시 절대경로를 사용하여 저장하세요: plt.savefig(r'{img_dir}/figure_{state.get("roop_back", 0)}_n.png')
    - 한글 폰트 깨짐을 방지하기 위해 'koreanize_matplotlib' 라이브러리가 설치되어 있다고 가정하고 import하세요. 또는 폰트 설정을 직접 하세요.
    
    - 각각의 이미지 파일은 하나의 그래프 또는 표만 들어가야합니다
    - csv파일은 생성하지마세요.
    - numpy,pandas, matplotlib, seaborn ,koreanize_matplotlib 라이브러리를 사용하세요.
    """
    
    try:
        with langfuse_session(session_id=s_id, user_id=u_id):
            response = structured_llm.invoke(prompt, config={'callbacks': callbacks})
        code = response.code
        font_name = executor_instance.available_font or 'Malgun Gothic' # Fallback

        header = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform
import matplotlib.font_manager as fm
from matplotlib import font_manager

# 1. 시스템 폰트 찾기 (실제 설치된 폰트 확인)
def get_korean_font():
    font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    
    # 우선순위별 한글 폰트
    preferred_fonts = [
        'Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 
        'AppleGothic', 'Apple SD Gothic Neo',
        'Noto Sans KR', 'Noto Sans CJK KR',
        'DejaVu Sans'  # fallback
    ]
    
    for pref_font in preferred_fonts:
        for font_path in font_list:
            font_name = fm.FontProperties(fname=font_path).get_name()
            if pref_font.lower() in font_name.lower():
                return pref_font
    
    # 아무 한글 폰트나 찾기
    for font_path in font_list:
        font_name = fm.FontProperties(fname=font_path).get_name()
        if any(keyword in font_name.lower() for keyword in ['gothic', 'nanum', 'malgun', 'apple']):
            return font_name
    
    return 'DejaVu Sans'  # 최종 fallback

korean_font = get_korean_font()

# 2. Matplotlib 전역 설정 (강제)
plt.rcParams.update({{
    'font.family': korean_font,
    'font.sans-serif': [korean_font, 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.autolayout': True
}})

# 3. Seaborn 설정 (폰트 설정 후에)
try:
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    # Seaborn이 폰트를 덮어쓰지 않도록 재설정
    sns.set(font=korean_font, rc={{'font.family': korean_font}})
except:
    pass

# 4. 폰트 캐시 무효화 (필요시)
try:
    fm._rebuild()
except:
    pass

print(f"사용 중인 한글 폰트: {{korean_font}}")
"""
        code = header + "\n" + code
    except Exception as e:
        return {"now_log": [f"Code Generation Failed: {str(e)}"], "error_roop": state.get("error_roop", 0) + 1}
        
    return {"code": code}

@observe(name="Run")
def run_code(state:analyzeState)->analyzeState:
    code = state.get("code","")
    roop = str(state.get("roop_back", 0))
    img_paths = []
    # 기존에 생성된 이미지 파일이 있다면 삭제 (초기화)
    import glob
    import os
    
    current_dir = os.getcwd().replace("\\", "/")
    img_dir = f"{current_dir}/img"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    target_pattern = f"{img_dir}/figure_{roop}_*.png"
    target_csv = f"{img_dir}/*.csv"
    for f in glob.glob(target_pattern):
        os.remove(f)
    for f in glob.glob(target_csv):
        os.remove(f)
    # [Safety] 코드 주입 및 로깅
   
    try:
        # [NEW] Persistent Executor 사용
        result = executor_instance.run(code)
        
        logger.info(f"실행 결과: {result[:500]}")
        
        # Traceback이 포함되어 있으면 에러로 간주
        if "Traceback" in result or "Error" in result:
             return {
                "now_log": [result], 
                "error_roop": state.get("error_roop",0) + 1
            }
        
        # 생성된 이미지 파일 확인
        img_paths = sorted(glob.glob(target_pattern))
        
        return {"result_summary": result, "result_img_paths": img_paths,"now_log":None,"error_roop": 0}
    except Exception as e:
        return {
            "now_log": [str(e)], 
            "error_roop": state.get("error_roop",0)  + 1
        }
@observe(name="Eval")    
def evaluation_code(state: analyzeState,config:RunnableConfig):
    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    prompt = f"""
    당신은 마케팅 분석 검증 전문가(LLM-as-a-judge)입니다.
    [분석 계획]: {state['plan']}
    [실행 결과]: {state['final_insight']}

    위 결과가 계획대로 도출되었으며, 수치가 논리적으로 타당한지 검증하세요.
    결과가 타당하면 'APPROVE', 부족하거나 오류가 보이면 'REJECT'와 이유를 적으세요.
    지금은 테스트 상황이니 'APPROVE'를 반환해주세요.
    """
    llm, callbacks = LLMFactory.create('openai', 'gpt-5-nano', temperature=0.3)
        
    with langfuse_session(session_id=s_id, user_id=u_id):
        response = llm.invoke(prompt, config={'callbacks': callbacks})
    
    if "APPROVE" in response.content:
        return {"is_approved": True}
    else:
        return {"is_approved": False, "now_log": [response.content]}


def route_wait_node(state: analyzeState):
    pass

def router_error(state: analyzeState):

    if state.get("error_roop",0) >=3:
        raise Exception("분석 코드 실행이 3회 연속 실패했습니다. 코드를 검토해주세요.")
    elif state.get("now_log",None):
        return "Make"

    else:
        return "Insight" 

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
        return "Create"
    elif choice == "완료":
        return "END"

    else:
        pass


class ImageInsight(BaseModel):
    img_name: str = Field(description="분석한 이미지 파일명 (예: figure_0_0.png)")
    insight: str = Field(description="해당 이미지에 대한 상세 분석 및 해석")

class InsightOutput(BaseModel):
    overall_insight: str = Field(description="전체 데이터를 아우르는 종합 인사이트 (비즈니스 액션 아이템 포함)")
    image_specific_insights: List[ImageInsight] = Field(description="각 이미지별 개별 분석 결과 리스트")


@observe(name="Insight")
def derive_insight_node(state: analyzeState, config: RunnableConfig):

    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    roop = str(state.get("roop_back", 0))

    # 기존에 생성된 이미지 파일이 있다면 삭제 (초기화)
    import glob
    import os
    img_paths = []
    current_dir = os.getcwd().replace("\\", "/")
    img_dir = f"{current_dir}/img"
    
    plan = state.get("plan", "")
    df_summary = state.get("df_summary", "")
    
    # [Fix] 현재 루프에서 생성된 이미지만 로드 (TypeError 해결 & 증분 분석)
    img_paths = sorted(glob.glob(f"{img_dir}/figure_{roop}_*.png"))
    new_img_paths = img_paths

    # 1. 메시지 구성
    messages_content = []
    messages_content.append({"type": "text", "text": f"""
    당신은 수석 데이터 분석가입니다.
    
    [분석 배경]
    - 계획: {plan}
    - 데이터 요약 정보: {df_summary}
    
    [새로운 시각화 이미지 목록]:
    {[os.path.basename(p) for p in new_img_paths]}
    
    위 정보와 새롭게 제공되는 이미지를 바탕으로 다음을 수행하세요.
    1. **새로운 개별 이미지 분석**: 새로 추가된 이미지({[os.path.basename(p) for p in new_img_paths]})에 대해서만 구체적인 수치와 패턴을 분석하세요.
    2. **이번 회차({roop}) 종합 인사이트**: **이번에 새로 추가된 시각화 결과**가 전체 분석에 어떤 의미를 주는지 설명하는 **독립적인 종합 결과**를 작성하세요.
    """})
    
    # 2. 이미지 첨부
    for img_path in new_img_paths:
        if os.path.exists(img_path):
            with open(img_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            messages_content.append({"type": "text", "text": f"Image Filename: {os.path.basename(img_path)}"})
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"}
            })
            
    if not new_img_paths:
         messages_content.append({"type": "text", "text": "(생성된 이미지가 없습니다. 텍스트 결과 및 데이터 요약을 바탕으로 분석해 주세요.)"})

    msg = HumanMessage(content=messages_content)
    
    llm, callbacks = LLMFactory.create('openai', 'gpt-5-nano', temperature=0.3)
    structured_llm = llm.with_structured_output(InsightOutput)
    
    try:
        with langfuse_session(session_id=s_id, user_id=u_id):
            response = structured_llm.invoke([msg], config={'callbacks': callbacks})
            
        filename_map = {os.path.basename(p): p for p in img_paths}
        
        # [Fix] Overall Insight를 병합하지 않고 회차별로 분리하여 저장
        overall_key = f"overall_{roop}"
        
        final_insight = {
            overall_key: {
                "insight": response.overall_insight,
                "img_path": None
            }
        }
        
        for item in response.image_specific_insights:
            full_path = filename_map.get(item.img_name, None)
            
            final_insight[item.img_name] = {
                "insight": item.insight,
                "img_path": full_path
            }
                
    except Exception as e:
        print(f"Structured Output Failed: {e}. Fallback to text.")
        llm, callbacks = LLMFactory.create('openai', 'gpt-5-nano', temperature=0.3)
        with langfuse_session(session_id=s_id, user_id=u_id):
            plain_response = llm.invoke([msg], config={'callbacks': callbacks})
            
        overall_key = f"overall_{roop}"
        final_insight = {
            overall_key: {
                "insight": plain_response.content,
                "img_path": None
            }
        }

    return {"final_insight": final_insight}
