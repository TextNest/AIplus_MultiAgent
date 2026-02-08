"""
Analyze Data Node
담당: [팀원 C]

역할: LLM을 활용하여 데이터 분석 수행
입력: clean_data
출력: analysis_results, steps_log, retry_count

=============================================================================
구현 가이드
=============================================================================
이 노드는 LLM이 Python 분석 코드를 생성하고, REPL에서 실행하는 핵심 노드입니다.

흐름:
1. clean_data를 DataFrame으로 변환
2. LLM에게 데이터 정보를 주고 분석 코드 생성 요청
3. 생성된 코드를 REPL에서 실행
4. 실행 결과를 analysis_results에 저장

재시도 로직:
- evaluate_results에서 REJECT되면 이 노드가 다시 호출됨
- previous_feedback에 이전 피드백이 담겨있음 → 프롬프트에 반영
=============================================================================
"""
import re
import pandas as pd
from ..state import AgentState
from ..tools import get_python_repl
from ...core.llm_factory import LLMFactory, langfuse_session


def analyze_data(state: AgentState) -> AgentState:
    """
    LLM이 생성한 Python 코드로 데이터를 분석합니다.
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
        
        # =====================================================================
        # 구현 예시 시작 (TODO: 팀원이 수정/확장)
        # =====================================================================
        
        # Step 1: LLM 생성
        llm, callbacks = LLMFactory.create(
            provider="google",
            model="gemini-2.0-flash",
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
            session_id=f"analyze-attempt-{retry_count + 1}",
            tags=["analyze_data", f"attempt_{retry_count + 1}"]
        ):
            response = llm.invoke(prompt, config={"callbacks": callbacks})
        
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
        
    except Exception as e:
        return {
            "analysis_results": [f"ERROR: {str(e)}"],
            "steps_log": [f"[Analyze] ERROR: {str(e)}"],
            "retry_count": retry_count + 1
        }


def _extract_python_code(text: str) -> str:
    """마크다운 코드 블록에서 Python 코드 추출"""
    pattern = r"```(?:python)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()
