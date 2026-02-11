"""
Generate Report Node
담당: [팀원 E]

역할: 분석 결과를 기반으로 보고서 생성
입력: analysis_results, clean_data, raw_data, file_type
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
from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session


def generate_report(state: AgentState) -> AgentState:
    """
    분석 결과를 바탕으로 Markdown 보고서를 생성합니다.
    """
    analysis_results = state.get("analysis_results", [])
    clean_data = state.get("clean_data")
    raw_data = state.get("raw_data")
    file_type = state.get("file_type", "tabular")
    file_path = state.get("file_path", "데이터")
    
    if not analysis_results:
        return {
            "final_report": "# Error\n\nNo analysis results available.",
            "steps_log": ["[Report] ERROR: No analysis results"]
        }
    
    try:
        # Step 1: LLM 생성
        llm, callbacks = LLMFactory.create(
            provider="google",
            model="gemini-2.0-flash",
            temperature=0.3,  # 보고서는 약간의 창의성 허용
        )
        
        all_results = "\n\n---\n\n".join(analysis_results)

        # Step 2: 프롬프트 구성 (파일 타입별 분기)
        if file_type == "document":
            data_summary = ""
            if raw_data:
                data_summary = f"- 문서 경로: {file_path}"

            prompt = f"""
당신은 전문 문서 분석가입니다. 분석 결과를 바탕으로 보고서를 작성하세요.

## 문서 정보
{data_summary}

## 분석 결과
{all_results}

## 보고서 작성 지침
1. **한국어**로 작성하세요.
2. **Markdown** 형식을 사용하세요.
3. 문서의 핵심 내용을 잘 요약하고 정리하세요.

## 보고서 구조
```markdown
# 문서 분석 보고서

## 1. 요약 (Executive Summary)
(핵심 내용 3줄 요약)

## 2. 주요 키워드 및 수치
(분석된 키워드 및 주요 데이터)

## 3. 상세 분석 내용
(문단별 상세 요약 및 내용)

## 4. 인사이트
(비즈니스적 함의 또는 결론)
```
"""
        else:
            # 기존 Tabular 리포트 로직
            data_summary = ""
            if clean_data:
                df = pd.DataFrame(clean_data)
                data_summary = f"""
- 데이터 출처: {file_path}
- 총 행 수: {len(df):,}
- 총 컬럼 수: {len(df.columns)}
- 컬럼 목록: {', '.join(df.columns)}
"""
            
            prompt = f"""
당신은 전문 데이터 분석가입니다. 분석 결과를 바탕으로 비즈니스 보고서를 작성하세요.

## 데이터 정보
{data_summary}

## 분석 결과
{all_results}

## 보고서 작성 지침
1. **한국어**로 작성하세요.
2. **Markdown** 형식을 사용하세요.
3. 기술적 내용을 비전문가도 이해할 수 있게 설명하세요.
4. 핵심 인사이트를 강조하세요.
5. 가능하다면 비즈니스 권고사항을 포함하세요.

## 보고서 구조
```markdown
# 데이터 분석 보고서

## 1. 요약 (Executive Summary)
(핵심 발견사항 3줄 요약)

## 2. 데이터 개요
(분석 대상 데이터 설명)

## 3. 주요 발견사항
### 3.1 [발견사항 1]
### 3.2 [발견사항 2]
...

## 4. 상세 분석
(통계, 패턴, 상관관계 등)

## 5. 결론 및 권고사항
(비즈니스 관점의 제언)
```

위 구조에 맞춰 보고서를 작성하세요.
"""

        # Step 4: LLM 호출
        with langfuse_session(
            session_id="generate-report",
            tags=["generate_report", "markdown", str(file_type)]
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
