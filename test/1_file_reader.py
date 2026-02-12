# 1단계: 파일 읽기 (File Reader)
# PDF/Word 문서에서 텍스트 추출




def extract_pdf_text(file_path: str) -> str:
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
        return _extract_pdf_via_gemini(file_path)

    return "\n\n".join(p for p in text_parts if p)


def extract_word_text(file_path: str) -> str:
    """Word(DOCX/DOC) 파일에서 텍스트 추출"""
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


def read_file(file_path: str) -> Optional[str]:
    """
    파일 경로별로 텍스트 추출
    
    Args:
        file_path: 읽을 파일 경로
        
    Returns:
        추출된 텍스트 또는 None
    """
    ext = file_path.split('.')[-1].lower()

    if ext == "pdf":
        return extract_pdf_text(file_path)
    elif ext in ["docx", "doc"]:
        return extract_word_text(file_path)
    else:
        print(f"[FILE_READER] ERROR: {ext} 형식은 지원하지 않습니다")
        return None


# 사용 예시
if __name__ == "__main__":
    # text = read_file("report.pdf")
    # print(f"추출된 텍스트: {len(text)} 글자")
    pass
