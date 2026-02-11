"""
서브그래프 모듈
==============

팀원들이 독립적인 서브에이전트를 구축할 수 있는 서브그래프 패턴을 제공합니다.

사용법:
    1. _template 폴더를 복사하여 새 서브그래프 생성
    2. state.py에서 Input/Internal/Output 필드 정의
    3. nodes/ 폴더에 노드 함수 구현
    4. graph.py에서 워크플로우 구성
    5. 메인 graph.py에서 서브그래프 호출

예시:
    from src.agent.subgraphs.analysis import create_analysis_subgraph
    
    analysis_subgraph = create_analysis_subgraph()
    result = analysis_subgraph.invoke(input_state)
"""

from .analysis import create_analysis_subgraph, AnalysisState

__all__ = [
    "create_analysis_subgraph",
    "AnalysisState",
]
