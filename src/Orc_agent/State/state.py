from typing import TypedDict, Annotated, List, Optional, Literal, Dict, Any
import operator
import pandas as pd

def merge_logs(left: List[str], right: List[str]) -> List[str]:
    if right is None:
        return left
    if left is None:
        return right
    return left + right

class analyzeState(TypedDict):
    user_choice: str
    preprocessing_data : str
    code: str
    result_summary: str
    result_img_paths: List[str]
    feed_back: str
    now_log: str
    roop_back: int
    plan:str
    df_summary:str
    error_roop:int
    is_approved:bool
    final_insight: Dict[str, Any]
    user_query : str



class DocumentState(TypedDict):
    file_path:str
    file_text:Optional[str]
    raw_data: Optional[dict]
    analysis_summary: Optional[str]
    steps_log: Annotated[List[str], merge_logs]


class AgentState(TypedDict):
    # Workflow session identifier (shared across all nodes for Langfuse tracing)
    session_id: Optional[str]
    
    # Input data path or raw data
    file_path: str
    file_type: Optional[Literal["tabular", "document"]]
    raw_data: Optional[dict]
    

    user_query: Optional[str]
    
    # Processed data
    clean_data: Optional[dict]
    
    # Analysis results (logs, figures, summary text)
    analysis_results: Optional[dict]
    
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