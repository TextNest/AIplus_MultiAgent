from typing import TypedDict, Annotated, List, Optional, Literal
from src.Orc_agent.State.state import AgentState

class ReportState(AgentState):
    """
    State specifically for the Report Generation Subgraph.
    Inherits from AgentState to access analysis results and data.
    """
    next_worker: str # The next worker node to execute (or FINISH)
    generated_formats: Annotated[List[str], "merge_logs"] # Track which formats were already generated