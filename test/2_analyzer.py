# 2단계: 분석 (Analyzer)
# LLM을 사용하여 텍스트 구조화 및 분석

from typing import Dict
from src.core.observe import observe
from src.core.llm_factory import LLMFactory, langfuse_session


@observe(name="analyze_document")
def analyze_text(text: str) -> Dict:
    """
    LLM(Gemini 2.0 Flash)으로 문서 텍스트 분석 및 구조화
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        분석 결과 (요약, 키워드, 메타데이터 포함)
    """
    # 1. Gemini 2.0 Flash 모델 생성
    llm, callbacks = LLMFactory.create(
        provider="google",
        model="gemini-2.0-flash",
        temperature=0.0,
    )

    # 2. 텍스트 길이 제한
    truncated = text[:3000] if len(text) > 3000 else text

    # 3. 분석 프롬프트
    prompt = f"""
    너는 문서 분석 전문가야.
    아래 문서를 읽고 다음 정보를 한국어로 구조화해서 반환해.

    1. 한 문단 요약 (3~5문장)
    2. 주요 키워드 5~10개 (쉼표로 구분)
    3. 중요한 수치/날짜/고유명사 목록 (불릿 리스트)

    문서 내용:
    {truncated}
    """

    # 4. LLM 호출 (Langfuse 트레이싱 포함)
    with langfuse_session(session_id="document_analyzer"):
        response = llm.invoke(
            prompt,
            config={"callbacks": callbacks},
        )

    # 5. 응답 추출
    analysis_result = response.content if hasattr(response, "content") else str(response)

    return {
        "analysis": analysis_result,
        "model": "gemini-2.0-flash",
        "text_length": len(text),
        "truncated": len(text) > 3000,
    }


# 사용 예시
if __name__ == "__main__":
    # sample_text = "..."
    # result = analyze_text(sample_text)
    # print(result["analysis"])
    pass
