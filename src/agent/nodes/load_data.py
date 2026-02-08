"""
Load Data Node
담당: [팀원 A]

역할: CSV 파일을 로드하여 raw_data로 변환
입력: file_path
출력: raw_data, steps_log
"""
import pandas as pd
from ..state import AgentState


def load_data(state: AgentState) -> AgentState:
    """
    CSV 파일을 로드하여 dictionary 형태로 변환합니다.
    
    Args:
        state: 현재 AgentState (file_path 필수)
    
    Returns:
        AgentState: raw_data와 steps_log가 업데이트된 상태
    """
    file_path = state["file_path"]
    
    try:
        df = pd.read_csv(file_path)
        return {
            "raw_data": df.to_dict(),
            "steps_log": [f"[LoadData] Loaded {len(df)} rows from {file_path}"]
        }
    except FileNotFoundError:
        return {
            "raw_data": None,
            "steps_log": [f"[LoadData] ERROR: File not found - {file_path}"]
        }
    except Exception as e:
        return {
            "raw_data": None,
            "steps_log": [f"[LoadData] ERROR: {str(e)}"]
        }
