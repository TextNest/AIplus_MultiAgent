"""
Load Data Node
담당: [팀원 A] + [팀원 B/shere2 통합]

역할: 파일(CSV, PDF, DOCX)을 로드하여 raw_data로 변환
입력: file_path
출력: raw_data, file_type, steps_log
"""
import os
import base64
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from langchain_core.messages import HumanMessage

from ..state import AgentState
from ...core.llm_factory import LLMFactory
from ...core.observe import langfuse_session


def _extract_pdf_via_gemini(file_path: str, session_id: str = "unknown") -> str:
    """이미지 기반 PDF → Gemini Vision으로 텍스트 추출 (폴백)"""
    llm, callbacks = LLMFactory.create(
        provider="google",
        model="gemini-2.5-flash",
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

    with langfuse_session(session_id=session_id, tags=["load_data", "vision_fallback"]) as lf_metadata:
        resp = llm.invoke([msg], config={"callbacks": callbacks, "metadata": lf_metadata})

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


def extract_pdf_text(file_path: str, session_id: str = "unknown") -> str:
    """
    PDF 파일에서 텍스트 추출
    - 일반 PDF: PyMuPDF로 직접 추출
    - 이미지 기반 PDF: Gemini Vision으로 폴백
    """
    doc = fitz.open(file_path)
    pages = list(doc)
    text_parts = [
        p.get_text("text", sort=True).strip()
        for p in pages
    ]
    total_chars = sum(len(t) for t in text_parts)
    doc.close()

    # 페이지당 평균 50자 미만이면 이미지 기반으로 판단 → Gemini Vision 사용
    if pages and total_chars < 50 * len(pages):
        return _extract_pdf_via_gemini(file_path, session_id=session_id)

    return "\n\n".join(p for p in text_parts if p)


def extract_word_text(file_path: str) -> str:
    """Word(DOCX/DOC) 파일에서 텍스트 추출"""
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_data(state: AgentState) -> AgentState:
    """
    파일을 로드하여 적절한 형태로 변환합니다.
    - CSV: DataFrame → dict (tabular)
    - PDF/DOCX: Text → dict (document)
    
    Args:
        state: 현재 AgentState (file_path 필수)
    
    Returns:
        AgentState: raw_data, file_type, steps_log 업데이트
    """
    file_path = state["file_path"]
    wf_session_id = state.get("session_id", "unknown")
    
    if not os.path.exists(file_path):
        return {
            "raw_data": None,
            "file_type": None,
            "steps_log": [f"[LoadData] ERROR: File not found - {file_path}"]
        }

    ext = file_path.split('.')[-1].lower()
    
    try:
        if ext == "csv":
            df = pd.read_csv(file_path)
            return {
                "file_type": "tabular",
                "raw_data": df.to_dict(),
                "steps_log": [f"[LoadData] Loaded CSV: {len(df)} rows from {file_path}"]
            }
            
        elif ext == "pdf":
            text = extract_pdf_text(file_path, session_id=wf_session_id)
            return {
                "file_type": "document",
                "raw_data": {"content": text, "source": file_path},
                "steps_log": [f"[LoadData] Loaded PDF: {len(text)} chars from {file_path}"]
            }
            
        elif ext in ["docx", "doc"]:
            text = extract_word_text(file_path)
            return {
                "file_type": "document",
                "raw_data": {"content": text, "source": file_path},
                "steps_log": [f"[LoadData] Loaded DOCX: {len(text)} chars from {file_path}"]
            }
            
        else:
            return {
                "raw_data": None,
                "file_type": None,
                "steps_log": [f"[LoadData] ERROR: Unsupported file extension .{ext}"]
            }

    except Exception as e:
        return {
            "raw_data": None,
            "file_type": None,
            "steps_log": [f"[LoadData] ERROR: {str(e)}"]
        }
