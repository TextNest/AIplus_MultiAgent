
from typing import Literal
from .state import ReportState

def report_supervisor(state: ReportState) -> dict:
    """
    Supervisor node that decides the next step in report generation.
    """
    final_report = state.get("final_report")
    report_format = state.get("report_format", "markdown").lower()
    generated_formats = state.get("generated_formats", [])
    
    # helper to safely check format validity
    def needs_format(fmt):
        return fmt in report_format and fmt not in generated_formats

    # 1. If Markdown content is missing, generate it first
    if not final_report:
        return {"next_worker": "generate_content"}
    
    # 2. Check for requested formats that haven't been generated yet
    if "pdf" in report_format and "pdf" not in generated_formats:
        return {"next_worker": "create_pdf", "generated_formats": ["pdf"]}
        
    if "html" in report_format and "html" not in generated_formats:
        return {"next_worker": "create_html", "generated_formats": ["html"]}
        
    if "pptx" in report_format and "pptx" not in generated_formats:
        return {"next_worker": "create_pptx", "generated_formats": ["pptx"]}
        
    # 3. If all done (or just markdown requested), finish
    return {"next_worker": "FINISH"}