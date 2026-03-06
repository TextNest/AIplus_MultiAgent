import os
import io
import pandas as pd
import markdown

from pptx import Presentation
from pptx.util import Inches, Pt
from xhtml2pdf import pisa

from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session, merge_runnable_config, observe
from langchain_core.runnables import RunnableConfig
from src.Orc_agent.core.prompts import (
    REPORT_PROMPT_GENERAL, 
    REPORT_PROMPT_DECISION, 
    REPORT_PROMPT_MARKETING,
    REPORT_STYLE_CLASSIFICATION_PROMPT
)
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
    report_style = state.get("report_style")
    logger.info(f"현재 등록된 보고서 형식 : {report_format}, 스타일: {report_style}")

    
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
        # If report style is not decided or set to "AI 자동 판단 (추천)", classify it first
        if not report_style or report_style == "AI 자동 판단 (추천)":
            return {"next_worker": "classify_report_style"}
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

@observe(name="classify_report_style")
def classify_report_style(state: ReportState, config: RunnableConfig) -> ReportState:
    """
    Classifies the analysis results to choose the best report style using overall_insight.
    """
    analysis_results = state.get("analysis_results", {})
    
    if not analysis_results:
        return {"report_style": "일반 리포트", "steps_log": ["[Report] No analysis results, defaulting to 일반 리포트"]}
    
    try:
        llm, callbacks = LLMFactory.create(
            provider="google",
            model="gemma-3-27b-it",
            temperature=0,
        )
        
        # Extract overall insights for classification
        overall_insights = []
        if isinstance(analysis_results, dict):
            for key, value in analysis_results.items():
                if "overall" in key.lower():
                    insight = value.get("insight", "") if isinstance(value, dict) else str(value)
                    if insight:
                        overall_insights.append(insight)
        
        all_overall_text = "\n".join(overall_insights)
        if not all_overall_text:
             # Fallback: use first few results if overall is missing
             all_overall_text = str(list(analysis_results.values())[0])[:1000]

        prompt = REPORT_STYLE_CLASSIFICATION_PROMPT.format(overall_insight=all_overall_text)

        response = llm.invoke(prompt, config=config)
        selected_style = response.content.strip()
        
        # Validation
        valid_styles = ["일반 리포트", "의사 결정 리포트", "마케팅 예산 분배 리포트"]
        if selected_style not in valid_styles:
            for style in valid_styles:
                if style in selected_style:
                    selected_style = style
                    break
            else:
                selected_style = "일반 리포트"

        logger.info(f"AI가 선택한 보고서 스타일: {selected_style}")
        return {
            "report_style": selected_style,
            "steps_log": [f"[Report] AI classified report style as: {selected_style}"]
        }
    except Exception as e:
        logger.error(f"보고서 스타일 분류 중 오류: {e}")
        return {
            "report_style": "일반 리포트",
            "steps_log": [f"[Report] Classification failed, defaulting to 일반 리포트: {str(e)}"]
        }

@observe(name="generate_content")
def generate_content(state: ReportState, config: RunnableConfig) -> ReportState:
    """
    Generates report content in Markdown format using LLM.
    """
    analysis_results = state.get("analysis_results", {})
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
            provider="google",
            model="gemma-3-27b-it",
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
        
        # Process Analysis Results
        processed_results = []
        if isinstance(analysis_results, dict):
            for key, value in analysis_results.items():
                insight = ""
                if isinstance(value, dict):
                    insight = value.get("insight", "")
                else:
                    insight = str(value)
                
                if insight:
                    if "overall" in key.lower():
                        processed_results.append(f"### [Overall Insight]\n{insight}")
                    else:
                        processed_results.append(f"### [Insight for {key}]\n{insight}")
        elif isinstance(analysis_results, list):
            for res in analysis_results:
                processed_results.append(str(res))
        
        all_results = "\n\n---\n\n".join(processed_results)
        
        prompt_map = {
            "일반 리포트": REPORT_PROMPT_GENERAL,
            "의사 결정 리포트": REPORT_PROMPT_DECISION,
            "마케팅 예산 분배 리포트": REPORT_PROMPT_MARKETING
        }
        selected_style = state.get("report_style", "일반 리포트")
        prompt_template = prompt_map.get(selected_style, REPORT_PROMPT_GENERAL)
        prompt = prompt_template.format(
            data_summary=data_summary,
            all_results=all_results,
            figure_markdown=figure_markdown
        )

        with langfuse_session(session_id="generate-report", tags=["generate_report"]) as lf_metadata:
            invoke_cfg = merge_runnable_config(
                config,
                callbacks=callbacks,
                metadata=lf_metadata,
            )
            response = llm.invoke(prompt, config=invoke_cfg)
        
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
