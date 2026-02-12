# 3단계: 요약 및 통합 (Summarizer/Orchestrator)
# 파일 읽기 → 분석 → 최종 결과 통합

from typing import Dict, Optional
from src.core.observe import observe

from file_reader import read_file
from analyzer import analyze_text


@observe(name="document_analyzer")
def document_analyzer(state: Dict) -> Dict:
    """
    파일 읽기 → 분석 → 결과 통합
    
    Args:
        state: 입력 상태 (file_path 포함)
            {
                "file_path": "report.pdf",
                "steps_log": []
            }
            
    Returns:
        처리 결과
            {
                "clean_data": {
                    "text": "원본 텍스트",
                    "analysis": "LLM 분석 결과",
                    "metadata": {...}
                },
                "steps_log": ["처리 단계 기록"]
            }
    """
    state.setdefault("steps_log", [])
    state.setdefault("clean_data", None)

    # === 1단계: 파일 읽기 ===
    file_path = state.get("file_path")
    if not file_path:
        state["steps_log"].append("[document_analyzer] ERROR: file_path missing")
        return state

    text = read_file(file_path)
    if text is None:
        state["steps_log"].append(f"[document_analyzer] ERROR: Failed to read {file_path}")
        return state

    state["steps_log"].append(f"[document_analyzer] ✓ 파일 읽기 완료: {len(text)} 글자")

    # === 2단계: 분석 ===
    analysis_result = analyze_text(text)
    state["steps_log"].append("[document_analyzer] ✓ 분석 완료")

    # === 3단계: 결과 통합 ===
    state["clean_data"] = {
        "text": text,
        "summary": analysis_result["analysis"],
        "metadata": {
            "length": len(text),
            "model": analysis_result["model"],
            "truncated": analysis_result["truncated"],
        },
    }
    
    state["steps_log"].append("[document_analyzer] ✓ 요약 완료")

    return state


def process_document(file_path: str, verbose: bool = True) -> Optional[Dict]:
    """
    간단한 문서 처리 인터페이스
    
    Args:
        file_path: 처리할 파일 경로
        verbose: 로그 출력 여부
        
    Returns:
        처리 결과
    """
    state = {
        "file_path": file_path,
        "steps_log": [],
    }

    result = document_analyzer(state)

    if verbose:
        print("\n=== 처리 단계 ===")
        for log in result["steps_log"]:
            print(log)
        
        if result["clean_data"]:
            print("\n=== 결과 ===")
            print(f"📝 요약:\n{result['clean_data']['summary']}")
            print(f"\n📊 메타데이터: {result['clean_data']['metadata']}")

    return result


# 사용 예시
if __name__ == "__main__":
    # result = process_document("report.pdf", verbose=True)
    pass
