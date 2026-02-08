"""
Evaluate Results Node
담당: [팀원 D]

역할: 분석 결과를 평가하여 승인/재시도 결정
입력: analysis_results, clean_data
출력: evaluation_feedback, steps_log

=============================================================================
구현 가이드
=============================================================================
이 노드는 Critic Agent 역할을 합니다.
analyze_data의 결과를 LLM이 평가하고, 품질이 부족하면 REJECT + 피드백을 반환합니다.

평가 기준 (예시):
- 완전성: 주요 통계가 모두 포함되었는가?
- 정확성: 코드 실행 에러 없이 완료되었는가?
- 유용성: 비즈니스에 유의미한 인사이트가 있는가?
- 명확성: 결과가 이해하기 쉽게 정리되었는가?

반환값:
- "APPROVE" → generate_report로 진행
- "REJECT: {구체적 피드백}" → analyze_data 재실행 (피드백 반영)
=============================================================================
"""
import pandas as pd
from ..state import AgentState
from ...core.llm_factory import LLMFactory, langfuse_session


def evaluate_results(state: AgentState) -> AgentState:
    """
    분석 결과의 품질을 평가하고 승인 여부를 결정합니다.
    """
    analysis_results = state.get("analysis_results", [])
    clean_data = state.get("clean_data")
    retry_count = state.get("retry_count", 0)
    
    if not analysis_results:
        return {
            "evaluation_feedback": "REJECT: No analysis results to evaluate",
            "steps_log": ["[Evaluate] REJECT: No results available"]
        }
    
    try:
        # =====================================================================
        # 구현 예시 시작 (TODO: 팀원이 수정/확장)
        # =====================================================================
        
        # Step 1: LLM 생성
        llm, callbacks = LLMFactory.create(
            provider="google",
            model="gemini-2.0-flash",
            temperature=0,  # 평가는 일관성 있게
        )
        
        # Step 2: 데이터 컨텍스트 준비
        data_context = ""
        if clean_data:
            df = pd.DataFrame(clean_data)
            data_context = f"""
## 원본 데이터 정보
- 행 수: {len(df)}
- 컬럼: {list(df.columns)}
"""
        
        # Step 3: 평가 프롬프트 구성
        latest_result = analysis_results[-1] if analysis_results else ""
        
        prompt = f"""
당신은 데이터 분석 품질을 평가하는 시니어 데이터 사이언티스트입니다.

{data_context}

## 분석 결과 (평가 대상)
{latest_result}

## 평가 기준
1. **완전성**: 기초 통계, 분포, 상관관계 등 주요 분석이 포함되었는가?
2. **정확성**: 코드가 에러 없이 실행되고 결과가 올바른가?
3. **유용성**: 데이터에서 의미있는 패턴이나 인사이트를 발견했는가?
4. **명확성**: 결과가 이해하기 쉽게 정리되었는가?

## 지시사항
- 각 기준을 1-5점으로 평가하세요.
- 평균 점수가 3점 이상이면 APPROVE, 미만이면 REJECT하세요.
- REJECT시 구체적으로 무엇을 개선해야 하는지 피드백을 제공하세요.

## 출력 형식 (반드시 이 형식으로)
```
평가:
- 완전성: X/5
- 정확성: X/5  
- 유용성: X/5
- 명확성: X/5
- 평균: X.X/5

결정: APPROVE 또는 REJECT

피드백: (REJECT인 경우) 개선이 필요한 구체적인 사항
```
"""

        # Step 4: LLM 호출
        with langfuse_session(
            session_id=f"evaluate-attempt-{retry_count}",
            tags=["evaluate_results"]
        ):
            response = llm.invoke(prompt, config={"callbacks": callbacks})
        
        # Step 5: 응답 처리
        content = response.content
        if isinstance(content, list):
            content = "".join([str(part) for part in content])
        
        # Step 6: 결정 추출
        if "APPROVE" in content.upper():
            feedback = "APPROVE"
        else:
            # REJECT인 경우 전체 평가 내용을 피드백으로 전달
            feedback = f"REJECT: {content}"
        
        return {
            "evaluation_feedback": feedback,
            "steps_log": [f"[Evaluate] Decision: {'APPROVE' if 'APPROVE' in feedback else 'REJECT'}"]
        }
        
        # =====================================================================
        # 구현 예시 끝
        # =====================================================================
        
    except Exception as e:
        return {
            "evaluation_feedback": f"REJECT: Evaluation error - {str(e)}",
            "steps_log": [f"[Evaluate] ERROR: {str(e)}"]
        }
