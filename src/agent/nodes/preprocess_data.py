"""
Preprocess Data Node
담당: [팀원 B]

역할: raw_data를 정제하여 clean_data로 변환
입력: raw_data
출력: clean_data, steps_log
"""
import pandas as pd
from ..state import AgentState


def preprocess_data(state: AgentState) -> AgentState:
    """
    Raw data를 전처리하여 분석 가능한 형태로 변환합니다.
    
    TODO: 팀원이 구현해야 할 내용
    - 결측치 처리
    - 데이터 타입 변환
    - 이상치 처리
    - 필요시 피처 엔지니어링
    
    Args:
        state: 현재 AgentState (raw_data 필수)
    
    Returns:
        AgentState: clean_data와 steps_log가 업데이트된 상태
    """
    raw_data = state.get("raw_data")
    
    if raw_data is None:
        return {
            "clean_data": None,
            "steps_log": ["[Preprocess] ERROR: No raw data available"]
        }
    
    try:
        df = pd.DataFrame(raw_data)
        
        # TODO: 전처리 로직 구현
        # 예시: df = df.dropna()
        # 예시: df = df.fillna(df.mean())
        
        clean_data = df.to_dict()
        
        return {
            "clean_data": clean_data,
            "steps_log": [f"[Preprocess] Processed {len(df)} rows"]
        }
    except Exception as e:
        return {
            "clean_data": None,
            "steps_log": [f"[Preprocess] ERROR: {str(e)}"]
        }
