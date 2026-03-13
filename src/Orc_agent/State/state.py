from typing import TypedDict, Annotated, List, Optional, Literal, Dict, Any
import operator
import pandas as pd

def merge_logs(left: List[str], right: List[str]) -> List[str]:
    if right is None:
        return left
    if left is None:
        return right
    return left + right

def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    if right is None:
        return left
    if left is None:
        return right
    return {**left, **right}

class analyzeState(TypedDict):
    user_choice: str
    preprocessing_data : str
    code: str
    result_summary: str
    result_img_paths: Annotated[List[str], merge_logs]
    feed_back: Annotated[List[str], merge_logs]
    now_log: Annotated[List[str], merge_logs]
    roop_back: int
    plan:str
    df_summary:str
    error_roop: Annotated[int, operator.add]
    is_approved:bool
    final_insight: Annotated[Dict[str, Any], merge_dicts]
    user_query : str





class preprocessState(TypedDict, total=False):
    file_path: str
    raw_dataframe: Any
    output_format: str
    cleaned_dataframe: Any 
    raw_preprocessing_report: dict
    categorical_standardization_report: dict
    duplicate_cleanup_report: dict
    date_integrity_report: dict
    clean_file_path: Optional[str]
    data_state_report: dict
    column_roles: dict
    role_justifications: dict
    detected_layers: dict
    derived_metrics: list
    derivation_skipped: list
    reliability_flags: dict
    cleanup_actions: list
    funnel_chains: list
    leakage_analysis: dict
    share_metrics: list
    volume_efficiency_flags: list
    has_time_dimension: bool
    trend_metrics: list
    saturation_flags: list
    objective_notes: dict
    safety_checks: list
    anomaly_flags: list
    reporting_dataframe: Any
    feature_catalogue: list
    column_order: list
    formatted_output: str
    current_stage: str
    iteration_count: int
    error: Optional[str]
    agent_feedback: Annotated[List[str], merge_logs]
    steps_log: Annotated[List[str], merge_logs]
    clean_data: Optional[Dict[str, Any]]



class ReportState(TypedDict):
    """
    State specifically for the Report Generation Subgraph.
    Independent from AgentState to minimize coupling.
    """
    # Inputs from Main Agent
    analysis_results: Optional[dict]  # Text insights
    figure_list: List[str]            # Image paths
    file_path: str                    # Data source path
    clean_data: Optional[dict]        # Raw data sample (optional)

    # Internal State
    final_report: Optional[str]  # Markdown content
    report_format: List[str]     # Requested formats (pdf, html, pptx)
    report_style: Optional[str]   # Report type (general, decision, marketing)
    generated_formats: Annotated[List[str], merge_logs] # Track generated formats
    steps_log: Annotated[List[str], merge_logs]
    next_worker: str             # Control flow

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
    
    # [NEW] Configuration for LLM Models across nodes
    node_models: Optional[dict]

    user_query: Optional[str]
    
    # Processed data
    clean_data: Optional[dict]
    
    # Analysis results (logs, figures, summary text)
    analysis_results: Optional[dict]
    figure_list: Optional[List[str]]
    
    # Evaluation feedback (if analysis is insufficient)
    evaluation_feedback: Optional[str]
    
    # Final Report
    report_type: Optional[List[str]]
    final_report: Optional[str]


    # Human Feedback
    human_feedback: Optional[str]
    
    # Processing steps log
    steps_log: Annotated[List[str], merge_logs]
    
    # Retry counter for analysis loop (prevents infinite loops)
    retry_count: int

    feed_back:str
