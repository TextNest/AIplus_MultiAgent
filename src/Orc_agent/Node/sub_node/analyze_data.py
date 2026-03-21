
import re
import pandas as pd
from ...State.state import analyzeState

from ...core.df_summary import get_df_summary
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session, merge_runnable_config
from langchain_core.messages import HumanMessage
import base64
import os
import json

from ...core.observe import observe
import  matplotlib
from src.Orc_agent.core.logger import logger
from src.Orc_agent.core.executor import executor_instance

# --- helper ---
def _get_metadata_prompt(state: dict) -> str:
    """전처리 메타데이터(catalogue, roles)를 프롬프트용 텍스트로 변환하는 공통 함수"""
    catalogue_raw = state.get("feature_catalogue", [])
    roles_raw = state.get("column_roles", {})
    prep_report = state.get("formatted_output", "") 

    # 1. 컬럼 역할(Roles) 처리
    if roles_raw:
        roles_text = f"  - 컬럼별 마케팅 역할: {json.dumps(roles_raw, ensure_ascii=False)}"
        roles_instruction = "  * 분석 시 [컬럼별 마케팅 역할]을 참고하여 각 컬럼의 의미(비용, 수익, 세그먼트 등)를 정확히 파악하세요."
    else:
        roles_text = "  - 컬럼별 마케킹 역할: (제공된 역할 정보 없음)"
        roles_instruction = "  * 특별한 역할 정의가 없으므로 컬럼명을 통해 의미를 추론하세요."
    # 2. 카탈로그(Catalogue) 처리
    if catalogue_raw:
        cat_text = f"  - 주요 기능 카탈로그: {json.dumps(catalogue_raw, ensure_ascii=False)}"
        cat_instruction = "  * [주요 기능 카탈로그]에 정의된 수식과 지표명을 우선적으로 사용하세요."
    else:
        cat_text = "  - 주요 기능 카탈로그: (추가된 파생 지표 없음)"
        cat_instruction = "  * 파생 지표가 없으므로 기본 컬럼들을 조합하여 분석하세요."
    # 3. 전처리 요약 리포트(formatted_output) 처리
    if prep_report:
        prep_text = f"  - 전처리 단계 요약 리포트: {prep_report}"
        prep_instruction = "  * 위 리포트에서 언급된 데이터 특이사항(결측치, 이상치, 퍼널 분석 결과 등)을 분석의 배경 지식으로 활용하세요."
    else:
        prep_text = f"  - 전처리 단계 요약 리포트: (추가된 요약 리포트 없음)"
        prep_instruction = "  * 요약 리포트가 없으므로 기본 컬럼들을 조합하여 분석하세요."
    # 4. 하나로 합치기
    formatted_prompt_string = f"""
    [전처리 분석 결과]:
    {roles_text}
    {cat_text}
    {prep_text}
    [분석 지침]:
    {roles_instruction}
    {cat_instruction}
    {prep_instruction}
    """
    return formatted_prompt_string

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
    # df = pd.read_csv(file_path)
    df = pd.read_parquet(state["preprocessing_data"])
    df_summary = get_df_summary(df)
    # 추가: 전처리 결과물 연동
    metadata_context = _get_metadata_prompt(state)
    prompt =  f"""
    당신은 마케팅 데이터 전략가입니다. 제공된 데이터프레임의 요약 정보를 바탕으로 사용자의 질문에 답하기 위한 최적의 분석 시나리오를 설계하고 코드를 작성하세요.
    
    [데이터 정보]: {df_summary}
    [전처리 분석 결과]: {metadata_context}  
    [사용자 질문]: {state['user_query']}
    [피드백]: {state['feed_back']}
    위 데이터를 바탕으로 분석 계획을 세우세요. 
    - 어떤 KPI(ROAS, CTR 등)를 계산할 것인가?
    - 어떤 시각화(막대그래프, 선그래프 등)가 필요한가?
    - 단계별 분석 순서를 나열하세요.
    *주의: 파이썬 코드는 작성하지 말고 오직 '계획'만 작성하세요.*
    이미지 파일은 최대 3개만 만들 수 있도록 계획을 구축하세요. 다만 각각의 이미지 파일은 하나의 그래프 또는 표만 들어가야합니다.
    """
    node_conf = state.get("node_models", {}).get("plan_node", {})
    provider = node_conf.get("provider") or "google"
    model_name = node_conf.get("model") or "gemma-3-27b-it"
    llm, callbacks = LLMFactory.create(provider, model_name, temperature=0.3)
        
    with langfuse_session(session_id=s_id, user_id=u_id) as lf_metadata:
        invoke_cfg = merge_runnable_config(
            config,
            callbacks=callbacks,
            metadata=lf_metadata,
        )
        response = llm.invoke(prompt, config=invoke_cfg)

    plan = response.content
    return {"plan":plan , "df_summary":df_summary,"roop_back":roop_back,"error_roop": 0}

@observe(name="Make")
def make_analysis_code(state:analyzeState,config:RunnableConfig)-> analyzeState:
    thread_id = config["configurable"].get("thread_id")
    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    node_conf = state.get("node_models", {}).get("make_node", {})
    provider = node_conf.get("provider") or "google"
    model_name = node_conf.get("model") or "gemma-3-27b-it"
    llm, callbacks = LLMFactory.create(provider, model_name)
        
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
    img_dir = f"{current_dir}/img/{thread_id}"
    # 추가: 전처리 결과물 연동
    metadata_context = _get_metadata_prompt(state)
    prompt = f"""
    분석 계획: {state['plan']}
    데이터 요약: {state['df_summary']}
    [전처리 분석 결과]: {metadata_context}
    [데이터 파일 경로]: {file_path}
    {text} 
    위 분석 계획을 확인하고 실행하기 위한 파이썬 코드를 작성하세요. 
    
    [필수]
    - 변수명 앞에 _df 이렇게 작성하지 마세요. 추가적인 df가 필요하다면 copy1_df, copy2_df, ...처럼 작성하세요. 절대 변수명 앞에 _ 사용하지 마세요.
    - 코드 시작 부분에서 반드시 데이터를 로드하세요: df = pd.read_parquet(r'{file_path}') 
    - 설명이나 마크다운(```python ... ```) 없이 오직 파이썬 코드만 출력하세요. 

    - 분석 결과값 출력을 위한 print는 지양하되, 에러 방지 및 데이터 유무 확인을 위한 로그성 print는 허용합니다
    
    - 모든 수치 컬럼은 연산(sum, mean 등)을 수행하기 **전**에 반드시 `.fillna(0)`를 먼저 적용하세요.
    - 예: plot_df = df.fillna(0).groupby('category').sum(numeric_only=True) 
    - 또는 개별 컬럼: total = df['column'].fillna(0).sum()
    - 수치 결과값(Scalar)에는 `.fillna()`를 사용할 수 없으므로, 반드시 연산 대상이 되는 Series나 DataFrame 단계에서 미리 결측치를 제거해야 합니다.
    - 연산 결과가 단일 숫자(scalar)인 경우, 결과값이 np.nan인지 확인하려면 np.isnan(value)을 사용하거나, 계산 전 데이터에서 NaN을 완벽히 제거한 후 연산하세요.

    - 모든 시각화(특히 plt.pie) 실행 전에는 반드시 아래와 같은 '데이터 검증 패턴'을 적용하세요.
    - 패턴: 
      if not plot_df.empty and plot_df['수치컬럼'].sum() > 0:
          # 여기에 그래프 그리기 및 저장 코드 작성
      else:
          print(f"{{img_name}}: 시각화할 유효 데이터가 부족하여 생략합니다.")
    - 시각화(특히 plt.pie)를 하기 전에는 반드시 데이터에 NaN이 있는지 확인하고 .dropna() 또는 .fillna(0) 처리를 하세요.
    - 데이터프레임이 비어있는지(if df.empty:) 확인하여, 비어있다면 그래프를 그리는 대신 print('데이터가 없어 시각화를 건너뜁니다.')와 같은 로그를 남기고 다음 단계로 넘어가게 하세요.

    [이미지 저장 규칙]
    - 시각화가 필요한 경우, 각 그래프를 '{img_dir}/figure_{state.get("roop_back", 0)}_0.png', '{img_dir}/figure_{state.get("roop_back", 0)}_1.png', ... 와 같이 순서대로 저장하세요. ({state.get("make_insight", 0)}은 현재 인사이트 번호입니다.)
    - 반드시 절대경로를 사용하여 저장하세요: plt.savefig(r'{img_dir}/figure_{state.get("roop_back", 0)}_n.png')
    - 한글 폰트 깨짐을 방지하기 위해 'koreanize_matplotlib' 라이브러리가 설치되어 있다고 가정하고 import하세요. 또는 폰트 설정을 직접 하세요.
    - 모든 시각화 전에는 반드시 데이터프레임이 비어있는지 혹은 수치에 NaN이 포함되어 있는지 체크하여, 오류 없이 실행되도록 방어 코드를 작성하세요.
    
    - 각각의 이미지 파일은 하나의 그래프 또는 표만 들어가야합니다
    - csv파일은 생성하지마세요.
    - numpy,pandas, matplotlib, seaborn ,koreanize_matplotlib 라이브러리를 사용하세요.
    
    [에러 방지 규칙]
    - 데이터 형변환(예: pd.to_datetime)을 할 때는 반드시 해당 컬럼의 데이터가 실제로 그 형식이 맞을 때만 사용하세요. 'SKU'나 순수 문자열 컬럼을 날짜로 바꾸려고 시도하지 마세요. (errors='coerce' 옵션을 적극 활용하세요)
    - 사칙연산(더하기, 나누기 등)을 수행할 때는 반드시 pd.to_numeric() 처리를 먼저 하거나, df.select_dtypes(include=[np.number]) 를 사용하여 숫자형 데이터만 있는 컬럼인지 확인한 후에 계산을 수행하세요. 문자열 컬럼으로는 절대 수식 연산을 하지 마세요.
    - 데이터에 결측치(NaN)나 무한대(inf) 값이 있을 수 있으니, 연산 전후로 dropna() 나 fillna()로 데이터를 정제하는 코드를 포함하세요.
    """
    
    try:
        with langfuse_session(session_id=s_id, user_id=u_id) as lf_metadata:
            invoke_cfg = merge_runnable_config(
                config,
                callbacks=callbacks,
                metadata=lf_metadata,
            )
            response = llm.invoke(prompt, config=invoke_cfg)

        # Content 추출 로직 강화: 리스트 블록인 경우 텍스트만 추출
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                code = "\n".join([part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text"])
            else:
                code = str(content)
        else:
            code = str(response)

        # Markdown 코드 블록 제거
        if "```" in code:
            match = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                # 백틱만 잇고 끝나는 경우 대비
                code = code.replace("```python", "").replace("```", "").strip()

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
        
    return {"code": code, "now_log": None}

@observe(name="Run")
def run_code(state:analyzeState, config: RunnableConfig)->analyzeState:
    thread_id = config["configurable"].get("thread_id") 
    code = state.get("code","")
    roop = str(state.get("roop_back", 0))
    img_paths = []
    # 기존에 생성된 이미지 파일이 있다면 삭제 (초기화)
    import glob
    import os
    
    current_dir = os.getcwd().replace("\\", "/")
    img_dir = f"{current_dir}/img/{thread_id}"
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
    """
    node_conf = state.get("node_models", {}).get("eval_node", {})
    provider = node_conf.get("provider") or "google"
    model_name = node_conf.get("model") or "gemma-3-27b-it"
    llm, callbacks = LLMFactory.create(provider, model_name, temperature=0.3)
        
    with langfuse_session(session_id=s_id, user_id=u_id) as lf_metadata:
        invoke_cfg = merge_runnable_config(
            config,
            callbacks=callbacks,
            metadata=lf_metadata,
        )
        response = llm.invoke(prompt, config=invoke_cfg)
    
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = "\n".join([part["text"] for part in content if isinstance(part, dict) and part.get("type") == "text"])
        
    if "APPROVE" in content.upper():
        return {"is_approved": True}
    else:
        return {"is_approved": False, "now_log": [str(content)]}


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
    thread_id = config["configurable"].get("thread_id")
    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    roop = str(state.get("roop_back", 0))

    # 기존에 생성된 이미지 파일이 있다면 삭제 (초기화)
    import glob
    import os
    img_paths = []
    current_dir = os.getcwd().replace("\\", "/")
    img_dir = f"{current_dir}/img/{thread_id}"
    
    plan = state.get("plan", "")
    df_summary = state.get("df_summary", "")
    # 추가: 전처리 결과물 연동
    metadata_context = _get_metadata_prompt(state)
    
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
    - 전처리 분석 결과: {metadata_context} 
    
    [새로운 시각화 이미지 목록]:
    {[os.path.basename(p) for p in new_img_paths]}
    
    위 정보와 새롭게 제공되는 이미지를 바탕으로 다음을 수행하세요.
    1. **새로운 개별 이미지 분석**: 새로 추가된 이미지({[os.path.basename(p) for p in new_img_paths]})에 대해서만 구체적인 수치와 패턴을 분석하세요.
    2. **이번 회차({roop}) 종합 인사이트**: **이번에 새로 추가된 시각화 결과**가 전체 분석에 어떤 의미를 주는지 설명하는 **독립적인 종합 결과**를 작성하세요.

    [주의]
    - 숫자나 기간의 범위를 표현할 때 절대 물결표(~) 기호를 쓰지 말고, 하이픈(-)이나 '부터 ~ 까지' 같은 한글을 사용하세요. (마크다운 오작동 방지)
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
    
    node_conf = state.get("node_models", {}).get("eval_node", {})
    provider = node_conf.get("provider") or "google"
    model_name = node_conf.get("model") or "gemma-3-27b-it"
    llm, callbacks = LLMFactory.create(provider, model_name, temperature=0.3)
    structured_llm = llm.with_structured_output(InsightOutput)
    
    try:
        with langfuse_session(session_id=s_id, user_id=u_id) as lf_metadata:
            invoke_cfg = merge_runnable_config(
                config,
                callbacks=callbacks,
                metadata=lf_metadata,
            )
            response = structured_llm.invoke([msg], config=invoke_cfg)
            
        filename_map = {os.path.basename(p): p for p in img_paths}
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
        logger.warning(f">>> [Insight] Structured Output Failed: {e}. Fallback to plain text.")
        node_conf = state.get("node_models", {}).get("eval_node", {})
        provider = node_conf.get("provider") or "google"
        model_name = node_conf.get("model") or "gemma-3-27b-it"
        llm, callbacks = LLMFactory.create(provider, model_name, temperature=0.3)
        
        # Fallback prompt for better parsing
        # HumanMessage.content가 리스트(이미지 포함)이므로 텍스트 파트만 합쳐서 프롬프트 생성
        text_parts = [p["text"] for p in msg.content if isinstance(p, dict) and p.get("type") == "text"]
        fallback_prompt = "\n".join(text_parts) + "\n\n각 이미지별 분석은 '### [이미지파일명]'으로 시작하는 섹션으로 구분해서 작성해 주세요."
        
        with langfuse_session(session_id=s_id, user_id=u_id) as lf_metadata:
            invoke_cfg = merge_runnable_config(
                config,
                callbacks=callbacks,
                metadata=lf_metadata,
            )
            plain_response = llm.invoke(fallback_prompt, config=invoke_cfg)
            content = plain_response.content if hasattr(plain_response, 'content') else str(plain_response)
            
        overall_key = f"overall_{roop}"
        final_insight = {overall_key: {"insight": content, "img_path": None}}
        
        # Simple parsing for fallback
        filename_map = {os.path.basename(p): p for p in img_paths}
        for filename, full_path in filename_map.items():
            pattern = rf"### \[{re.escape(filename)}\](.*?)(?=### \[|$)"
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                final_insight[filename] = {
                    "insight": match.group(1).strip(),
                    "img_path": full_path
                }

    return {"final_insight": final_insight}
