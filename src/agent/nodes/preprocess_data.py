"""
Preprocess Data Node
Role: Clean raw_data and convert to clean_data.
"""
import pandas as pd
from ..state import AgentState

def preprocess_data(state: AgentState) -> AgentState:
    """
    Preprocesses raw data.
    
    Args:
        state: Current AgentState (requires raw_data)
    
    Returns:
        AgentState: Updates clean_data and steps_log
    """
    raw_data = state.get("raw_data")
    
    if raw_data is None:
        return {
            "clean_data": None,
            "steps_log": ["[Preprocess] ERROR: No raw data available"]
        }
    
    try:
        df = pd.DataFrame(raw_data)
        
        # Simple preprocessing: dropna for now as placeholder
        # In a real scenario, this would be more robust
        df_clean = df.dropna()
        
        clean_data = df_clean.to_dict()
        
        return {
            "clean_data": clean_data,
            "steps_log": [f"[Preprocess] Processed {len(df)} -> {len(df_clean)} rows (dropped na)"]
        }
    except Exception as e:
        return {
            "clean_data": None,
            "steps_log": [f"[Preprocess] ERROR: {str(e)}"]
        }
