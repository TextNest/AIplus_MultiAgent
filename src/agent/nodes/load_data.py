"""
Load Data Node
Role: Load CSV file into raw_data dictionary.
"""
import pandas as pd
from ..state import AgentState

def load_data(state: AgentState) -> AgentState:
    """
    Loads CSV file into a dictionary.
    
    Args:
        state: Current AgentState (requires file_path)
    
    Returns:
        AgentState: Updates raw_data and steps_log
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
