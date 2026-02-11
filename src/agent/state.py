from typing import TypedDict, Annotated, List, Optional, Literal
import operator
import pandas as pd

def merge_logs(left: List[str], right: List[str]) -> List[str]:
    if right is None:
        return left
    if left is None:
        return right
    return left + right

class AgentState(TypedDict):
    # Workflow session identifier (shared across all nodes for Langfuse tracing)
    session_id: Optional[str]
    
    # Input data path or raw data
    file_path: str
    file_type: Optional[Literal["tabular", "document"]]
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
