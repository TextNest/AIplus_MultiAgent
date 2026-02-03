"""
에이전트 노드 정의

각 노드는 AgentState를 입력받아 부분 상태 업데이트를 반환합니다.
팀원들은 TODO 표시된 부분을 구현하면 됩니다.

담당자:
- load_data, human_review: 인프라 (구현 완료)
- preprocess_data: [팀원 A] 데이터 전처리 담당
- analyze_data: [팀원 B] LLM 분석 담당  
- evaluate_results: [팀원 C] 결과 평가 담당
- generate_report: [팀원 D] 보고서 생성 담당
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from .state import AgentState
from .tools import python_repl_tool, get_python_repl
import pandas as pd
import os
import re

# =============================================================================
# LLM 및 Langfuse 설정 (인프라 - 수정 불필요)
# =============================================================================

from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

def get_langfuse_handler():
    """Get Langfuse callback handler if configured, else None."""
    # TODO: Langfuse 로깅 임시 비활성화
    return None
    # if all(key in os.environ for key in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]):
    #     return LangfuseCallbackHandler()
    # return None

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment")
    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0
)

def _invoke_llm(prompt: str, metadata: dict = None) -> str:
    """
    LLM 호출 헬퍼 함수 (Langfuse 트레이싱 자동 적용)
    
    Args:
        prompt: LLM에 전달할 프롬프트
        metadata: Langfuse 메타데이터 (예: {"langfuse_tags": ["node_name"]})
    
    Returns:
        LLM 응답 텍스트
    
    사용 예시:
        content = _invoke_llm("분석해줘", metadata={"langfuse_tags": ["analyze"]})
    """
    handler = get_langfuse_handler()
    config = {}
    if handler:
        config["callbacks"] = [handler]
        if metadata:
            config["metadata"] = metadata
    
    response = llm.invoke(prompt, config=config if config else None)
    content = response.content
    if isinstance(content, list):
        content = "".join([str(c) for c in content])
    return content


# =============================================================================
# 인프라 노드 (구현 완료 - 수정 불필요)
# =============================================================================

def load_data(state: AgentState) -> AgentState:
    """
    [인프라] 데이터 로드 노드
    
    CSV 파일을 로드하고 REPL 환경에 df로 등록합니다.
    """
    file_path = state['file_path']
    try:
        df = pd.read_csv(file_path)
        repl = get_python_repl()
        repl.run(f"import pandas as pd\ndf = pd.read_csv('{file_path}')")
        
        return {
            "raw_data": df.to_dict(),
            "steps_log": [f"Loaded data from {file_path}. Shape: {df.shape}"]
        }
    except Exception as e:
        return {"steps_log": [f"Error loading data: {e}"]}


def human_review(state: AgentState) -> AgentState:
    """
    [인프라] 사람 리뷰 대기 노드
    
    이 노드에서 워크플로우가 중단되고, 사람의 피드백을 기다립니다.
    graph.py에서 interrupt_before로 설정되어 있습니다.
    """
    return {"steps_log": ["Waiting for Human Review..."]}


# =============================================================================
# 팀원 구현 노드 - 아래 TODO를 구현하세요
# =============================================================================

def preprocess_data(state: AgentState) -> AgentState:
    """
    [팀원 A 담당] 데이터 전처리 노드
    
    입력:
        - state['raw_data']: 로드된 원본 데이터 (dict)
        - REPL 환경에 'df' 변수로 데이터프레임 접근 가능
    
    해야 할 일:
        - 결측치 처리
        - 데이터 타입 변환
        - 이상치 탐지/처리
        - 데이터 정규화/스케일링 (필요시)
    
    출력:
        - clean_data: 전처리된 데이터 (Optional)
        - analysis_results: 전처리 결과 요약 추가
        - steps_log: 처리 단계 로그 추가
    
    사용 가능한 도구:
        - get_python_repl(): REPL 환경 접근
        - python_repl_tool.invoke(code): 코드 실행
    """
    # TODO: 팀원 A가 구현
    
    # 예시 코드 (실제 구현으로 교체하세요)
    repl = get_python_repl()
    
    # 예시: 기본 데이터 정보 확인
    info_output = repl.run("print(df.info())")
    
    # 예시: 결측치 확인
    # null_check = repl.run("print(df.isnull().sum())")
    
    # 예시: 결측치 처리
    # repl.run("df = df.fillna(0)")
    
    # 예시: 이상치 탐지
    # repl.run("from scipy import stats; z_scores = stats.zscore(df.select_dtypes(include=['number']))")
    
    return {
        "steps_log": [f"[TODO] Data preprocessing - Info:\n{info_output}"],
        "analysis_results": [f"Data Info:\n{info_output}"],
        # "clean_data": processed_df.to_dict()  # 전처리 완료시 추가
    }


def analyze_data(state: AgentState) -> AgentState:
    """
    [팀원 B 담당] LLM 기반 데이터 분석 노드
    
    입력:
        - state['raw_data'] 또는 state['clean_data']: 분석할 데이터
        - state['human_feedback']: 사용자 피드백 (재분석 요청시)
        - state['retry_count']: 현재 재시도 횟수
        - REPL 환경에 'df' 변수로 데이터프레임 접근 가능
    
    해야 할 일:
        - LLM에게 분석 코드 생성 요청
        - 생성된 코드를 REPL에서 실행
        - 분석 결과 (텍스트, 시각화 등) 수집
    
    출력:
        - analysis_results: 분석 결과 추가 (누적됨)
        - steps_log: 처리 단계 로그 추가
        - retry_count: 증가된 재시도 횟수
    
    사용 가능한 도구:
        - _invoke_llm(prompt, metadata): LLM 호출
        - python_repl_tool.invoke(code): 코드 실행
        - get_python_repl(): REPL 환경 직접 접근
    
    참고:
        - MAX_ANALYSIS_RETRIES=3 (graph.py에서 설정)
        - 재시도시 human_feedback에 피드백 내용이 들어옴
    """
    # TODO: 팀원 B가 구현
    
    # 재시도 카운트 증가
    current_retry = state.get('retry_count', 0) + 1
    
    # 예시 코드 (실제 구현으로 교체하세요)
    user_request = state.get('human_feedback') or 'Analyze the data and find interesting patterns.'
    previous_steps = state.get('steps_log', [])
    
    prompt = f"""
    You have a pandas dataframe 'df'.
    User request: {user_request}
    Previous steps: {previous_steps}
    
    Write python code to analyze 'df'. Print the results.
    Wrap code in ```python ... ```.
    """
    
    content = _invoke_llm(prompt, metadata={"langfuse_tags": ["analyze_data"]})
    
    # 코드 추출 및 실행
    code_match = re.search(r"```python(.*?)```", content, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        result = python_repl_tool.invoke(code)
    else:
        result = "[TODO] No code found in LLM response."
    
    return {
        "analysis_results": [f"Analysis Code:\n{code_match.group(1) if code_match else 'None'}", f"Output:\n{result}"],
        "steps_log": [f"[TODO] Executed analysis code. (attempt {current_retry})"],
        "retry_count": current_retry
    }


def evaluate_results(state: AgentState) -> AgentState:
    """
    [팀원 C 담당] 분석 결과 평가 노드 (Critic Agent)
    
    입력:
        - state['analysis_results']: 지금까지의 분석 결과들 (List)
    
    해야 할 일:
        - 분석 결과의 품질 평가
        - 보고서 작성에 충분한지 판단
        - APPROVE 또는 RETRY 결정
    
    출력:
        - evaluation_feedback: "APPROVE" 또는 "RETRY"
        - steps_log: 평가 결과 로그 추가
    
    중요:
        - "APPROVE" 반환시 → generate_report로 이동
        - "RETRY" 반환시 → analyze_data로 돌아감 (최대 3회)
    
    사용 가능한 도구:
        - _invoke_llm(prompt, metadata): LLM 호출
    
    평가 기준 예시:
        - 분석이 충분히 깊이 있는가?
        - 인사이트가 명확한가?
        - 시각화가 포함되어 있는가?
        - 결론이 도출되었는가?
    """
    # TODO: 팀원 C가 구현
    
    # 예시 코드 (실제 구현으로 교체하세요)
    latest_result = state.get('analysis_results', ['No results'])[-1]
    
    prompt = f"""
    Review the analysis results:
    {latest_result}
    
    Is this analysis sufficient to form a report? 
    Consider:
    - Are the insights clear and actionable?
    - Is there enough depth in the analysis?
    - Are visualizations or statistics included?
    
    Reply with "APPROVE" or "RETRY" followed by reasoning.
    """
    
    content = _invoke_llm(prompt, metadata={"langfuse_tags": ["evaluate_results"]})
    decision = content.strip()
    
    if "APPROVE" in decision.upper():
        return {
            "evaluation_feedback": "APPROVE", 
            "steps_log": ["[TODO] Analysis Approved by Critic Agent."]
        }
    else:
        return {
            "evaluation_feedback": "RETRY", 
            "steps_log": [f"[TODO] Analysis Critique: {decision}"]
        }


def generate_report(state: AgentState) -> AgentState:
    """
    [팀원 D 담당] 보고서 생성 노드
    
    입력:
        - state['analysis_results']: 모든 분석 결과 (List)
        - state['raw_data']: 원본 데이터 (참조용)
    
    해야 할 일:
        - 분석 결과를 종합하여 보고서 작성
        - Markdown 형식으로 구조화
        - (선택) PPT, HTML 등 추가 형식 지원
    
    출력:
        - final_report: 최종 보고서 (Markdown)
        - steps_log: 처리 단계 로그 추가
    
    보고서 구조 예시:
        1. 요약 (Executive Summary)
        2. 데이터 개요
        3. 주요 발견사항
        4. 상세 분석
        5. 결론 및 제언
    
    사용 가능한 도구:
        - _invoke_llm(prompt, metadata): LLM 호출
    
    확장 가능:
        - python-pptx로 PPT 생성
        - Jinja2로 HTML 템플릿 생성
    """
    # TODO: 팀원 D가 구현
    
    # 예시 코드 (실제 구현으로 교체하세요)
    all_results = state.get('analysis_results', [])
    context = "\n".join(all_results)
    
    prompt = f"""
    Based on the following analysis results, write a comprehensive report in Markdown.
    
    Structure:
    1. Executive Summary
    2. Data Overview  
    3. Key Findings
    4. Detailed Analysis
    5. Conclusions & Recommendations
    
    Analysis Context:
    {context}
    """
    
    content = _invoke_llm(prompt, metadata={"langfuse_tags": ["generate_report"]})
    
    return {
        "final_report": content,
        "steps_log": ["[TODO] Report Generated."]
    }
