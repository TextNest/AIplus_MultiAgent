from state import ReportState

# 웹 검색(추가 예정)

# PDF 생성
def create_pdf(state: ReportState):
    # PDF 생성 로직 ...
    return {"steps_log": ["PDF Report Generated"]}

# HTML 생성
def create_html(state: ReportState):
    # HTML 생성 로직 ...
    return {"steps_log": ["HTML Report Generated"]}

# PPTX 생성
def create_pptx(state: ReportState):
    # PPTX 생성 로직 ...
    return {"steps_log": ["PPTX Report Generated"]}