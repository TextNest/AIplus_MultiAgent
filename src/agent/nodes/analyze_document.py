"""
Analyze Document Node
담당: [팀원 B/shere2 통합]

역할: LLM을 활용하여 문서 텍스트 분석 수행
입력: raw_data (document type), evaluation_feedback
출력: analysis_results, steps_log, retry_count
"""
from ..state import AgentState
from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session


def analyze_document(state: AgentState) -> AgentState:
    """
    문서 텍스트를 LLM으로 분석하여 요약 및 키워드를 추출합니다.
    """
    raw_data = state.get("raw_data")
    retry_count = state.get("retry_count", 0)
    previous_feedback = state.get("evaluation_feedback")
    wf_session_id = state.get("session_id", "unknown")
    
    if not raw_data or "content" not in raw_data:
        return {
            "analysis_results": ["ERROR: No document content available"],
            "steps_log": ["[AnalyzeDoc] ERROR: No content to analyze"],
            "retry_count": retry_count
        }
    
    text = raw_data["content"]
    # Truncate if too long (Gemini 2.0 Flash supports 1M tokens, but let's be safe with 100k chars for now)
    truncated_text = text[:100000] if len(text) > 100000 else text
    
    try:
        # Step 1: LLM 생성
        llm, callbacks = LLMFactory.create(
            provider="google",
            model="gemma-3-27b-it",
            temperature=0,
        )
        
        # Step 2: 피드백 반영
        feedback_context = ""
        if previous_feedback and "REJECT" in previous_feedback:
            feedback_context = f"""
## 이전 분석 피드백 (개선 필요)
{previous_feedback}

위 피드백을 반영하여 분석을 개선하세요.
"""

        # Step 3: 프롬프트 구성
        prompt = f"""
당신은 문서 분석 전문가입니다.
아래 문서를 읽고 다음 정보를 한국어로 구조화해서 반환해 주세요.

## 문서 내용
{truncated_text}

{feedback_context}

## 요청 사항
1. **한 문단 요약** (3~5문장): 문서의 핵심 내용을 요약하세요.
2. **주요 키워드** (5~10개): 문서를 대표하는 키워드를 쉼표로 구분하여 나열하세요.
3. **주요 수치/날짜/고유명사**: 문서에 나오는 중요한 정량적 지표나 날짜, 인물/기관명을 불릿 리스트로 정리하세요.
4. **인사이트**: 문서에서 발견할 수 있는 비즈니스적 함의나 특이사항을 기술하세요.

## 출력 형식
반드시 마크다운 형식으로 출력하세요.
"""

        # Step 4: LLM 호출
        with langfuse_session(
            session_id=wf_session_id,
            tags=["analyze_document", f"attempt_{retry_count + 1}"]
        ) as lf_metadata:
            response = llm.invoke(prompt, config={
                "callbacks": callbacks,
                "metadata": lf_metadata,
            })
        
        # Step 5: 응답 처리
        content = response.content
        if isinstance(content, list):
            content = "".join([str(part) for part in content])
            
        analysis_result = f"""
## 문서 분석 결과 (Attempt {retry_count + 1})
{content}
"""
        
        return {
            "analysis_results": [analysis_result],
            "steps_log": [f"[AnalyzeDoc] Completed analysis (attempt {retry_count + 1})"],
            "retry_count": retry_count + 1
        }
        
    except Exception as e:
        return {
            "analysis_results": [f"ERROR: {str(e)}"],
            "steps_log": [f"[AnalyzeDoc] ERROR: {str(e)}"],
            "retry_count": retry_count + 1
        }
