"""
서브그래프 State 정의
====================

State를 세 가지 영역으로 구분하세요:
1. INPUT: 메인 그래프에서 받는 데이터
2. INTERNAL: 서브그래프 내부에서만 사용하는 데이터
3. OUTPUT: 메인 그래프로 반환할 결과

규칙:
- 모든 필드에 타입 힌트 필수
- Optional 필드는 None 기본값 허용
- List 필드는 Annotated + merge 함수로 자동 병합 가능
"""

from typing import TypedDict, Optional, List, Annotated


def merge_lists(left: List[str], right: List[str]) -> List[str]:
    """리스트 필드 자동 병합 함수"""
    if right is None:
        return left or []
    if left is None:
        return right or []
    return left + right


class TemplateState(TypedDict):
    """
    서브그래프 State 템플릿
    
    이 클래스를 복사하여 자신의 서브그래프에 맞게 수정하세요.
    클래스명도 변경하세요 (예: AnalysisState, RAGState 등)
    """
    
    # =========================================================================
    # INPUT: 메인 그래프에서 받는 데이터
    # =========================================================================
    input_data: dict
    """메인 그래프에서 전달받는 입력 데이터"""
    
    input_config: Optional[dict]
    """(선택) 추가 설정 파라미터"""
    
    # =========================================================================
    # INTERNAL: 서브그래프 내부에서만 사용
    # =========================================================================
    intermediate_result: Optional[str]
    """중간 처리 결과"""
    
    retry_count: int
    """재시도 횟수 (루프 제어용)"""
    
    steps_log: Annotated[List[str], merge_lists]
    """처리 단계 로그 (자동 병합)"""
    
    # =========================================================================
    # OUTPUT: 메인 그래프로 반환할 결과
    # =========================================================================
    output_result: Optional[str]
    """최종 처리 결과"""
    
    output_metadata: Optional[dict]
    """(선택) 결과에 대한 메타데이터"""
