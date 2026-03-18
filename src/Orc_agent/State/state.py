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

class preprocessState(TypedDict):
    """
    State specifically for the Preprocessing Subgraph.
    """
    file_path: Optional[str]
    node_models: Optional[Dict[str, Any]]
    # raw_dataframe: Optional[Any]       # pd.DataFrame 및 JSON 호환 → 아키텍처 변경으로 미사용 처리
    # cleaned_dataframe: Optional[Any]   # pd.DataFrame 및 JSON 호환 → 아키텍처 변경으로 미사용 처리
    current_stage: Optional[str]
    iteration_count: Optional[int]
    error: Optional[str]
    # agent_feedback: Annotated[List[str], merge_logs]
    steps_log: Annotated[List[str], merge_logs]
    raw_df_path: Optional[str]
    working_df_path: Optional[str]
    # 추가: 전처리 결과물 및 리포트 저장용 키들
    feature_catalogue: Optional[List[Dict[str, Any]]]
    column_roles: Optional[Dict[str, str]]
    formatted_output: Optional[str]
    raw_preprocessing_report: Optional[Dict[str, Any]]
    categorical_standardization_report: Optional[Dict[str, Any]]
    duplicate_cleanup_report: Optional[Dict[str, Any]]
    date_integrity_report: Optional[Dict[str, Any]]
    data_state_report: Optional[Dict[str, Any]]
    derived_metrics: Optional[List[Dict[str, Any]]]
    reliability_flags: Optional[Dict[str, Any]]
    role_justifications: Optional[Dict[str, str]]
    detected_layers: Optional[Dict[str, List[str]]]

class analyzeState(TypedDict):
    user_choice: str
    node_models: Optional[Dict[str, Any]]
    preprocessing_data : str
    code: str
    result_summary: str
    result_img_paths: Annotated[List[str], merge_logs]
    feed_back: Annotated[List[str], merge_logs]
    now_log: Optional[List[str]]
    roop_back: int
    plan:str
    df_summary:str
    error_roop: Annotated[int, operator.add]
    is_approved:bool
    final_insight: Annotated[Dict[str, Any], merge_dicts]
    user_query : str

    # 추가: 전처리 결과물 연동
    working_df_path: Optional[str]
    feature_catalogue: Optional[List[Dict[str, Any]]]
    column_roles: Optional[Dict[str, str]]
    formatted_output: Optional[str]

class ReportState(TypedDict):
    """
    State specifically for the Report Generation Subgraph.
    Independent from AgentState to minimize coupling.
    """
    # Inputs from Main Agent
    analysis_results: Optional[dict]  # Text insights
    node_models: Optional[Dict[str, Any]]
    figure_list: List[str]            # Image paths
    file_path: str                    # Data source path
    clean_data: Optional[dict]        # Raw data sample (optional)
    # 추가: 전처리 결과물 연동
    formatted_output: Optional[str]

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

    # 추가: 전처리 결과물 연동
    working_df_path: Optional[str]
    feature_catalogue: Optional[List[Dict[str, Any]]]
    column_roles: Optional[Dict[str, str]]
    formatted_output: Optional[str]