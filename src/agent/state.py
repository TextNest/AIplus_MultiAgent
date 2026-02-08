from typing import TypedDict, Annotated, List, Optional
import operator
import pandas as pd

def merge_logs(left: List[str], right: List[str]) -> List[str]:
    if right is None:
        return left
    if left is None:
        return right
    return left + right

class AgentState(TypedDict):
    # Input data path or raw data
    file_path: str
    raw_data: Optional[dict]
    
    # Processed data
    clean_data: Optional[dict]
    
    # Analysis results (logs, figures, summary text)
    analysis_results: Annotated[List[str], merge_logs]
    
    # Evaluation feedback (if analysis is insufficient)
    evaluation_feedback: Optional[str]
    
    # Final Report
    final_report: Optional[str]

    # Human Feedback
    human_feedback: Optional[str]
    
    # Processing steps log
    steps_log: Annotated[List[str], merge_logs]
    
    # Retry counter for analysis loop (prevents infinite loops)
    retry_count: int


class analyzeState(TypedDict):
    user_choice: str
    prepared_data: str
    code: str
    result_summary: str
    result_img_path: str
    feed_back: str
    now_log: str
    make_insight: int
    roop_back: int
    plan:str
    df_summary:str
    error_roop:int
    is_approved:bool