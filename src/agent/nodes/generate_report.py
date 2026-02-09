"""
Generate Report Node
담당: 이수현

역할: 분석 결과를 기반으로 보고서 생성
입력: analysis_results, clean_data
출력: final_report, steps_log

=============================================================================
구현 가이드
=============================================================================
이 노드는 분석 결과를 바탕으로 읽기 좋은 보고서를 생성합니다.

보고서 형식:
- 기본: Markdown (현재 구현)
- 확장 가능: HTML, PowerPoint (python-pptx)

보고서 구조 (예시):
1. 요약 (Executive Summary)
2. 데이터 개요
3. 주요 발견사항
4. 상세 분석 결과
5. 결론 및 권고사항

팁:
- temperature를 약간 높여서 (0.3~0.5) 자연스러운 문장 생성
- 분석 결과의 핵심만 추출하여 간결하게 정리
=============================================================================
"""
import pandas as pd
from ..state import AgentState
from ...core.llm_factory import LLMFactory, langfuse_session
from ..prompt_engineering.prompts import REPORT_PROMPT

def generate_report(state: AgentState) -> AgentState:
    """
    분석 결과를 바탕으로 Markdown 보고서를 생성합니다.
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
        # Step 1: LLM 생성
        llm, callbacks = LLMFactory.create(
            provider="google",
            model="gemini-2.5-flash",
            temperature=0.3,  # 보고서는 약간의 창의성 허용
        )
        
        # Step 2: 데이터 컨텍스트 준비
        data_summary = ""
        if clean_data:
            df = pd.DataFrame(clean_data)
            data_summary = f"""
- 데이터 출처: {file_path}
- 총 행 수: {len(df):,}
- 총 컬럼 수: {len(df.columns)}
- 컬럼 목록: {', '.join(df.columns)}
"""
        
        # Step 3: 시각화 자료
        figure_markdown = ""
        if figure_list:
            figure_markdown = "### 분석 시각화 자료\n"
            for fig in figure_list:
                figure_markdown += f"![{fig}]({fig})\n"
        
        # Step 4: 프롬프트
                all_results = "\n\n---\n\n".join(analysis_results)
        
        prompt = REPORT_PROMPT.format(
            data_summary=data_summary,
            all_results=all_results,
            figure_markdown=figure_markdown
        )

        # Step 5: LLM 호출
        with langfuse_session(
            session_id="generate-report",
            tags=["generate_report", "markdown"]
        ):
            response = llm.invoke(prompt, config={"callbacks": callbacks})
        
        # Step 5: 응답 처리
        content = response.content
        if isinstance(content, list):
            content = "".join([str(part) for part in content])
        
        # Step 6: 보고서 반환
        return {
            "final_report": content,
            "steps_log": ["[Report] Generated Markdown report successfully"]
        }

    except Exception as e:
        return {
            "final_report": f"# Error\n\n보고서 생성 중 오류 발생: {str(e)}",
            "steps_log": [f"[Report] ERROR: {str(e)}"]
        }


# =============================================================================
# 확장 예시: PowerPoint 보고서 생성 (선택적 구현)
# =============================================================================
"""
from pptx import Presentation
from pptx.util import Inches, Pt

def generate_pptx_report(state: AgentState) -> AgentState:
    '''
    PowerPoint 보고서 생성 예시 (python-pptx 필요)
    pip install python-pptx
    '''
    analysis_results = state.get("analysis_results", [])
    
    # 프레젠테이션 생성
    prs = Presentation()
    
    # 제목 슬라이드
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "데이터 분석 보고서"
    subtitle.text = "자동 생성된 분석 결과"
    
    # 내용 슬라이드 추가
    for i, result in enumerate(analysis_results, 1):
        bullet_slide_layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(bullet_slide_layout)
        shapes = slide.shapes
        
        title_shape = shapes.title
        body_shape = shapes.placeholders[1]
        
        title_shape.text = f"분석 결과 {i}"
        tf = body_shape.text_frame
        tf.text = result[:500]  # 길이 제한
    
    # 파일 저장
    output_path = "output/report.pptx"
    prs.save(output_path)
    
    return {
        "final_report": f"PowerPoint 보고서 생성됨: {output_path}",
        "steps_log": ["[Report] Generated PowerPoint report"]
    }
"""
