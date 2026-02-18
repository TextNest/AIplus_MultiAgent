import os
import io
import pandas as pd
import markdown

from pptx import Presentation
from pptx.util import Inches, Pt
from xhtml2pdf import pisa

from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session, observe
from src.agent.prompt_engineering.prompts import REPORT_PROMPT
from ...State.state import ReportState

from src.Orc_agent.core.logger import logger

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


from typing import Literal, List
@observe(name="Supervisor")
def report_supervisor(state: ReportState) -> ReportState:
    """
    Supervisor node that decides the next step in report generation.
    """
    final_report = state.get("final_report")
    report_format = state.get("report_format", ["markdown"])
    logger.info(f"현재 등록된 보고서 형식 : {report_format}")

    
    # Ensure report_format is a list of lowercase strings
    if isinstance(report_format, str):
        report_format = [report_format]
    report_format = [f.lower() for f in report_format]
    
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
@observe(name="generate_content")
def generate_content(state: ReportState) -> ReportState:
    """
    Generates report content in Markdown format using LLM.
    """
    analysis_results = state.get("analysis_results", [])
    clean_data = state.get("clean_data")
    file_path = state.get("file_path", "Data")
    figure_list = state.get("figure_list", [])
    
    if not analysis_results:
        return {
            "final_report": "# Error\n\nNo analysis results available.",
            "steps_log": ["[Report] ERROR: No analysis results"]
        }

    try:
        # LLM Setup
        llm, callbacks = LLMFactory.create(
            provider="openai",
            model="gpt-4o",
            temperature=0.3,
        )
        
        # Data Context
        data_summary = ""
        if clean_data:
            df = pd.DataFrame(clean_data)
            data_summary = f"""
- Data Source: {file_path}
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Column List: {', '.join(df.columns)}
"""
        
        # Visualization Context
        figure_markdown = ""
        if figure_list:
            figure_markdown = "### Key Visualizations\n"
            for fig in figure_list:
                figure_markdown += f"![{fig}]({fig})\n"
        
        all_results = "\n\n---\n\n".join(analysis_results)
        
        prompt = REPORT_PROMPT.format(
            data_summary=data_summary,
            all_results=all_results,
            figure_markdown=figure_markdown
        )

        with langfuse_session(session_id="generate-report", tags=["generate_report"]):
            response = llm.invoke(prompt, config={"callbacks": callbacks})
        
        # Handle DummyLLM response
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        if isinstance(content, list):
            content = "".join([str(part) for part in content])
            
        return {
            "final_report": content,
            "steps_log": ["[Report] Generated Markdown content via LLM"]
        }

    except Exception as e:
        return {
            "final_report": f"# Error\n\nReport generation failed: {str(e)}",
            "steps_log": [f"[Report] ERROR: {str(e)}"]
        }
@observe(name="create_pdf")
def create_pdf(state: ReportState) -> ReportState:
    """
    Converts Markdown report to PDF.
    """
    markdown_content = state.get("final_report", "")
    if not markdown_content:
        return {"steps_log": ["[Report] PDF Generation Skipped (No Content)"]}
        
    try:
        import markdown
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        html_content = markdown.markdown(markdown_content)
        
        # 1. Font Source (Windows)
        font_name = "MalgunGothic"
        system_font_path = r"C:/Windows/Fonts/malgun.ttf"
        
        # 2. Register Font via ReportLab directly
        # This bypasses xhtml2pdf's internal font loading which can cause temp file issues
        font_registered = False
        if os.path.exists(system_font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, system_font_path))
                font_registered = True
            except Exception as e:
                print(f"Font registration warning: {e}")
        
        if not font_registered:
             # Fallback to a standard font if registration fails
             font_name = "Helvetica" # standard PDF font
             print("Using fallback font: Helvetica")

        styled_html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: '{font_name}', sans-serif;
                }}
                img {{
                    max-width: 100%;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        output_path = os.path.join(OUTPUT_DIR, "report.pdf")
        with open(output_path, "wb") as f:
            pisa_status = pisa.CreatePDF(styled_html, dest=f, encoding='utf-8')
        
        if pisa_status.err:
            raise Exception("PDF generation failed")
            
        return {
            "steps_log": [f"[Report] Generated PDF report at {output_path}"],
            "generated_formats":["pdf"]
        }
    except Exception as e:
        return {"steps_log": [f"[Report] PDF Generation Error: {str(e)}"]}
@observe(name="create_html")
def create_html(state: ReportState) -> ReportState:
    """
    Converts Markdown report to HTML.
    """
    markdown_content = state.get("final_report", "")
    if not markdown_content:
        return {"steps_log": ["[Report] HTML Generation Skipped (No Content)"]}
        
    try:
        html_content = markdown.markdown(markdown_content)
        # Add basic styling
        styled_html = f"<html><body><style>body {{ font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; }} img {{ max-width: 100%; }}</style>{html_content}</body></html>"
        
        output_path = os.path.join(OUTPUT_DIR, "report.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(styled_html)
            
        return {
            "steps_log": [f"[Report] Generated HTML report at {output_path}"],
            "generated_formats":["html"]
        }
    except Exception as e:
        return {"steps_log": [f"[Report] HTML Generation Error: {str(e)}"]}
@observe(name="create_pptx")
def create_pptx(state: ReportState) -> ReportState:
    """
    Generates PowerPoint report.
    """
    analysis_results = state.get("analysis_results", [])
    figure_list = state.get("figure_list", [])
    
    try:
        prs = Presentation()
        
        # Title Slide
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]
        
        title.text = "Data Analysis Report"
        subtitle.text = "Generated by AIplus MultiAgent"
        
        # Content Slides
        if analysis_results:
            bullet_slide_layout = prs.slide_layouts[1]
            slide = prs.slides.add_slide(bullet_slide_layout)
            shapes = slide.shapes
            title_shape = shapes.title
            body_shape = shapes.placeholders[1]
            
            title_shape.text = "Analysis Insights"
            tf = body_shape.text_frame
            
            text_content = analysis_results[0].replace("```python", "").replace("```", "")
            tf.text = text_content[:500] + "..." if len(text_content) > 500 else text_content

        # Figure Slides
        for fig_path in figure_list:
            if os.path.exists(fig_path):
                blank_slide_layout = prs.slide_layouts[6]
                slide = prs.slides.add_slide(blank_slide_layout)
                
                left = Inches(1)
                top = Inches(1)
                height = Inches(5.5)
                slide.shapes.add_picture(fig_path, left, top, height=height)
        
        output_path = os.path.join(OUTPUT_DIR, "report.pptx")
        prs.save(output_path)
        
        return {
            "steps_log": [f"[Report] Generated PowerPoint report at {output_path}"],
            "generated_formats":["pptx"]
        }
    except Exception as e:
        return {"steps_log": [f"[Report] PPTX Generation Error: {str(e)}"]}