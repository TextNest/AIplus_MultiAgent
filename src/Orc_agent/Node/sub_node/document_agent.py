import os
import fitz  # PyMuPDF
from docx import Document
import base64
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ...State.state import DocumentState
from ...core.llm_factory import LLMFactory
from src.core.observe import observe, langfuse_session


def _extract_pdf_via_gemini(file_path: str, session_id: str = "unknown") -> str:
    """이미지 기반 PDF → Gemini Vision으로 텍스트 추출 (폴백)"""
    llm, callbacks = LLMFactory.create(
        provider="google",
        model="gemini-2.0-flash",
        temperature=0.0,
    )

    with open(file_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

    msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "이 PDF 문서의 모든 텍스트를 순서대로 추출해서 그대로 반환해주세요. 한국어와 영어 모두 포함하고, 표나 리스트 구조는 가능한 한 유지해주세요.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:application/pdf;base64,{pdf_b64}"},
            },
        ]
    )

    with langfuse_session(session_id=session_id, tags=["document_agent", "ocr_fallback"]):
        resp = llm.invoke([msg], config={"callbacks": callbacks})

    if hasattr(resp, "content"):
        c = resp.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b)
                for b in c
            )
    return str(resp)


def _extract_pdf_text(file_path: str, session_id: str) -> str:
    doc = fitz.open(file_path)
    pages = list(doc)
    text_parts = [p.get_text("text", sort=True).strip() for p in pages]
    total_chars = sum(len(t) for t in text_parts)
    doc.close()

    # 텍스트가 너무 적으면 이미지로 간주
    if pages and total_chars < 50 * len(pages):
        return _extract_pdf_via_gemini(file_path, session_id)

    return "\n\n".join(p for p in text_parts if p)


def _extract_word_text(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


@observe(name="document_analyzer")
def read_file_node(state: DocumentState, config: RunnableConfig) -> DocumentState:

    s_id = config["configurable"].get("session_id", "unknown")
    file_path = state.get("file_path")
    
    if not file_path or not os.path.exists(file_path):
        return {
            "steps_log": ["ERROR: 파일을 찾을 수 없습니다."],
            "file_text": None
        }

    try:
        ext = file_path.split('.')[-1].lower()
        content = ""
        
        if ext == "pdf":
            content = _extract_pdf_text(file_path, s_id)
        elif ext in ["docx", "doc"]:
            content = _extract_word_text(file_path)
        else:
            return {"steps_log": [f"[FILE_READER] ERROR: {ext} 형식은 지원하지 않습니다"]}

        return {
            "file_text": content,
            "steps_log": [f"추출된 텍스트: {file_path} ({len(content)} 글자)"]
        }
        
    except Exception as e:
        return {"steps_log": [f"ERROR reading file: {str(e)}"]}


def analyze_doc_node(state: DocumentState, config: RunnableConfig) -> DocumentState:

    u_id = config["configurable"].get("user_id")
    s_id = config["configurable"].get("session_id")
    
    raw_data = state.get("raw_data")
    if not raw_data or "content" not in raw_data:
        return {"steps_log": ["파일 분석 전, 파일을 업로드 해주세요."]}
    
    text = raw_data["content"]
    truncated_text = text[:3000] if len(text) > 3000 else text

    llm, callbacks = LLMFactory.create(
        provider="google",
        model="gemini-2.0-flash",
        temperature=0.0
    )
    
    prompt = f"""
    당신은 문서 분석 전문가입니다.
    아래 문서를 읽고 다음 정보를 한국어로 구조화해서 반환해 주세요.

    1. **한 문단 요약** (3~5문장)
    2. **주요 키워드** (5~10개, 쉼표 구분)
    3. **주요 수치/날짜/고유명사** (불릿 리스트)
    4. **인사이트** (비즈니스적 함의)

    ## 문서 내용
    {truncated_text}
    """
    
    with langfuse_session(session_id=s_id, user_id=u_id):
        response = llm.invoke(prompt, config={'callbacks': callbacks})
        
    return {
        "analysis_summary": response.content,
        "steps_log": ["Document analysis completed"]
    }
