# Agent Design Document

> 멀티 에이전트 기반 자동 데이터 분석 및 지능형 보고서 생성 시스템 설계 문서

---

## 1. 시스템 개요

### 1.1 목적
CSV, PDF, DOCX 파일을 입력받아 자동으로 분석하고, 품질 평가를 거쳐 보고서를 생성하는 멀티 에이전트 시스템

### 1.2 핵심 특징
- **파일 타입 기반 라우팅**: CSV(tabular) / PDF·DOCX(document) 자동 분기
- **LLM 기반 분석**: tabular은 Python 코드 생성+REPL 실행, document는 LLM 요약/분석
- **Self-Critique**: 분석 결과를 LLM이 평가하고 부족하면 재시도 (최대 3회)
- **Human-in-the-Loop**: 최종 보고서를 사람이 검토하고 피드백 제공

---

## 2. 아키텍처

### 2.1 워크플로우 다이어그램

```
load_data → route_by_file_type
  ├─ tabular  → preprocess_data → analyze_data ─┐
  └─ document → analyze_document ───────────────┤
                                                 ↓
                              evaluate_results → conditional
                                ├─ APPROVE → generate_report → human_review → END
                                └─ REJECT  → (file_type 기반) analyze_data 또는 analyze_document
```

### 2.2 상태 흐름

| 단계 | State 변화 |
|------|-----------|
| 1. load_data | `file_path` → `raw_data`, `file_type` |
| 2a. preprocess_data | `raw_data` → `clean_data` (tabular only) |
| 3a. analyze_data | `clean_data` → `analysis_results` (tabular) |
| 3b. analyze_document | `raw_data` → `analysis_results` (document) |
| 4. evaluate_results | `analysis_results` → `evaluation_feedback` (file_type 분기) |
| 5. generate_report | `analysis_results` → `final_report` (file_type 분기) |
| 6. human_review | `human_feedback` → END 또는 재분석 |

---

## 3. 노드 상세 설계

### 3.1 load_data (데이터 로드 + 파일 타입 분류)

| 항목 | 내용 |
|------|------|
| **역할** | CSV/PDF/DOCX 파일을 읽어 raw_data로 변환하고 file_type 분류 |
| **입력** | `file_path: str` |
| **출력** | `raw_data: dict`, `file_type: Literal["tabular", "document"]`, `steps_log` |
| **LLM** | ⚡ (이미지 기반 PDF일 때만 Gemini Vision 폴백) |
| **구현 상태** | ✅ 완료 |

**파일 타입별 처리**:

| 확장자 | 처리 방식 | file_type | raw_data 형식 |
|--------|-----------|-----------|---------------|
| `.csv` | `pd.read_csv()` → `df.to_dict()` | `tabular` | `{"col1": {"0": val, ...}, ...}` |
| `.pdf` | PyMuPDF 텍스트 추출 (이미지 PDF → Gemini Vision 폴백) | `document` | `{"content": str, "source": str}` |
| `.docx` | python-docx 텍스트 추출 | `document` | `{"content": str, "source": str}` |

**PDF 이미지 감지 기준**: 페이지당 평균 50자 미만이면 이미지 기반으로 판단하여 Gemini Vision 호출.

```python
# 핵심 로직 (CSV)
df = pd.read_csv(file_path)
return {"file_type": "tabular", "raw_data": df.to_dict(), "steps_log": [...]}

# 핵심 로직 (PDF/DOCX)
text = extract_pdf_text(file_path)  # 또는 extract_word_text()
return {"file_type": "document", "raw_data": {"content": text, "source": file_path}, "steps_log": [...]}
```

---

### 3.2 preprocess_data (전처리 — tabular only)

| 항목 | 내용 |
|------|------|
| **역할** | 결측치 처리, 타입 변환, 이상치 제거 |
| **입력** | `raw_data: dict` |
| **출력** | `clean_data: dict`, `steps_log` |
| **LLM** | ❌ 불필요 (룰 기반) |
| **구현 상태** | 🟡 스켈레톤 |

> document 타입은 이 노드를 거치지 않음 (route_by_file_type에서 분기)

**TODO (팀원 B)**:
- [ ] 결측치 처리 전략 (dropna / fillna)
- [ ] 숫자형 컬럼 자동 감지 및 변환
- [ ] 이상치 탐지 및 처리 (IQR, Z-score 등)

---

### 3.3 analyze_data (LLM 분석 — tabular only)

| 항목 | 내용 |
|------|------|
| **역할** | LLM이 분석 코드를 생성하고 REPL에서 실행 |
| **입력** | `clean_data: dict`, `evaluation_feedback: Optional[str]` |
| **출력** | `analysis_results: List[str]`, `retry_count`, `steps_log` |
| **LLM** | ✅ 필수 (코드 생성) |
| **구현 상태** | 🟢 예시 구현됨 |

**흐름**:
```
1. DataFrame 정보 추출 (컬럼, 타입, 샘플)
2. 프롬프트 구성 (이전 피드백 포함)
3. LLM 호출 → Python 코드 생성
4. 코드 추출 (```python ... ```)
5. REPL에서 실행
6. 결과 반환
```

**프롬프트 핵심 요소**:
- 데이터 정보 (행 수, 컬럼, 타입, 샘플)
- 기초 통계 (df.describe())
- 이전 피드백 (재시도인 경우)
- 출력 형식 지정 (코드만 반환)

---

### 3.4 analyze_document (LLM 문서 분석 — document only)

| 항목 | 내용 |
|------|------|
| **역할** | 문서 텍스트를 LLM으로 분석하여 요약/키워드/수치/인사이트 추출 |
| **입력** | `raw_data: dict` (content, source), `evaluation_feedback: Optional[str]` |
| **출력** | `analysis_results: List[str]`, `retry_count`, `steps_log` |
| **LLM** | ✅ 필수 (Gemini 2.0 Flash, temperature=0) |
| **구현 상태** | ✅ 완료 |

**분석 항목**:
1. **한 문단 요약** (3~5문장)
2. **주요 키워드** (5~10개)
3. **주요 수치/날짜/고유명사**
4. **인사이트** (비즈니스적 함의/특이사항)

**제한**: 입력 텍스트를 100,000자로 truncate
**피드백 반영**: `evaluation_feedback`에 REJECT가 있으면 프롬프트에 개선 피드백 포함

---

### 3.5 evaluate_results (품질 평가 — file_type 분기)

| 항목 | 내용 |
|------|------|
| **역할** | 분석 결과를 평가하고 APPROVE/REJECT 결정 |
| **입력** | `analysis_results: List[str]`, `clean_data: dict`, `raw_data: dict`, `file_type` |
| **출력** | `evaluation_feedback: str`, `steps_log` |
| **LLM** | ✅ 필수 (Critic 역할) |
| **구현 상태** | ✅ 완료 (tabular/document 분기) |

**파일 타입별 평가 기준**:

| 기준 | tabular | document |
|------|---------|----------|
| 완전성 | 주요 통계가 모두 포함되었는가? | 요약이 문서 핵심을 포함하는가? |
| 정확성 | 코드 에러 없이 실행되었는가? | 키워드가 적절한가? |
| 유용성 | 의미있는 인사이트가 있는가? | 수치/고유명사가 포함되었는가? |
| 명확성 | 이해하기 쉽게 정리되었는가? | 인사이트가 유의미한가? |

**결정 로직**:
- 평균 3점 이상 → `"APPROVE"`
- 평균 3점 미만 → `"REJECT: {구체적 피드백}"`

---

### 3.6 generate_report (보고서 생성 — file_type 분기)

| 항목 | 내용 |
|------|------|
| **역할** | 분석 결과를 Markdown 보고서로 변환 |
| **입력** | `analysis_results: List[str]`, `clean_data: dict`, `raw_data: dict`, `file_type` |
| **출력** | `final_report: str`, `steps_log` |
| **LLM** | ✅ 필수 (Gemini 2.0 Flash, temperature=0.3) |
| **구현 상태** | ✅ 완료 (tabular/document 분기) |

**파일 타입별 보고서 구조**:

| tabular | document |
|---------|----------|
| 1. 요약 (Executive Summary) | 1. 요약 |
| 2. 데이터 개요 | 2. 문서 정보 |
| 3. 주요 발견사항 | 3. 분석 결과 |
| 4. 상세 분석 | 4. 요약/인사이트 |
| 5. 결론 및 권고사항 | 5. 결론 |

**확장 가능**:
- HTML 보고서
- PowerPoint (python-pptx)
- PDF (weasyprint)

---

### 3.7 human_review (HITL)

| 항목 | 내용 |
|------|------|
| **역할** | 사람의 피드백을 받아 최종 결정 |
| **입력** | `final_report: str`, `human_feedback: Optional[str]` |
| **출력** | `steps_log` |
| **LLM** | ❌ 불필요 |
| **구현 상태** | ✅ 기본 구현 |

**동작**:
- `interrupt_before=["human_review"]`로 그래프 일시정지
- 사용자가 `human_feedback` 주입
- "APPROVE" → END
- 그 외 → file_type에 따라 `analyze_data` 또는 `analyze_document`로 재시작

---

## 4. 상태 (AgentState)

```python
class AgentState(TypedDict):
    # 입력
    file_path: str
    file_type: Optional[Literal["tabular", "document"]]  # 라우팅 키
    raw_data: Optional[dict]
    
    # 처리된 데이터 (tabular only)
    clean_data: Optional[dict]
    
    # 분석 결과 (List: 자동 병합)
    analysis_results: Annotated[List[str], merge_logs]
    
    # 평가 피드백
    evaluation_feedback: Optional[str]
    
    # 최종 보고서
    final_report: Optional[str]
    
    # 사람 피드백
    human_feedback: Optional[str]
    
    # 로그 (List: 자동 병합)
    steps_log: Annotated[List[str], merge_logs]
    
    # 재시도 카운터 (무한 루프 방지)
    retry_count: int
```

### 4.1 파일 타입별 raw_data 스키마

| file_type | raw_data 형식 | 예시 |
|-----------|---------------|------|
| `tabular` | `df.to_dict()` | `{"col1": {"0": "val", ...}, ...}` |
| `document` | `{"content": str, "source": str}` | `{"content": "추출된 텍스트...", "source": "report.pdf"}` |

### 4.2 List 자동 병합

`Annotated[List[str], merge_logs]` 사용 시:
```python
# 첫 번째 노드 반환
{"analysis_results": ["결과 1"]}

# 두 번째 노드 반환
{"analysis_results": ["결과 2"]}

# 최종 state
{"analysis_results": ["결과 1", "결과 2"]}  # 자동 병합됨
```

---

## 5. 라우팅 로직

### 5.1 route_by_file_type (load_data 이후)

```python
def route_by_file_type(state: AgentState):
    file_type = state.get("file_type", "tabular")
    if file_type == "document":
        return "analyze_document"
    return "preprocess_data"
```

### 5.2 should_continue_analysis (evaluate_results 이후)

```python
def should_continue_analysis(state: AgentState):
    feedback = state.get('evaluation_feedback')
    retry_count = state.get('retry_count', 0)
    
    # 최대 재시도 횟수 도달 시 강제 진행
    if retry_count >= MAX_ANALYSIS_RETRIES:  # 기본값: 3
        return "generate_report"
    
    if feedback == "APPROVE":
        return "generate_report"
    else:
        # REJECT → 파일 타입에 맞는 분석 노드로
        file_type = state.get("file_type", "tabular")
        if file_type == "document":
            return "analyze_document"
        return "analyze_data"
```

### 5.3 should_continue_human (human_review 이후)

```python
def should_continue_human(state: AgentState):
    feedback = state.get('human_feedback')
    
    if feedback and "APPROVE" in feedback.upper():
        return END
    else:
        # 반려 → 파일 타입에 맞는 분석 노드로
        file_type = state.get("file_type", "tabular")
        if file_type == "document":
            return "analyze_document"
        return "analyze_data"
```

---

## 6. 설정

| 설정 | 값 | 설명 |
|------|-----|------|
| `MAX_ANALYSIS_RETRIES` | 3 | 분석 최대 재시도 횟수 |
| `interrupt_before` | `["human_review"]` | HITL 중단점 |
| `checkpointer` | `MemorySaver()` | 상태 저장 (재개 가능) |

---

## 7. 구현 우선순위

| 순위 | 노드 | 상태 | 이유 |
|------|------|------|------|
| 1 | `load_data` | ✅ 완료 | CSV/PDF/DOCX 파일 로드 + 타입 분류 |
| 2 | `analyze_document` | ✅ 완료 | 문서 분석 핵심 기능 |
| 3 | `evaluate_results` | ✅ 완료 | tabular/document 분기 품질 평가 |
| 4 | `generate_report` | ✅ 완료 | tabular/document 분기 보고서 생성 |
| 5 | `analyze_data` | 🟢 예시 | tabular 분석 (코드 생성+REPL) |
| 6 | `preprocess_data` | 🟡 스켈레톤 | 데이터 품질 개선 |
| 7 | `human_review` | ✅ 기본 | 저장/알림 확장 |

---

## 8. 서브그래프 (Subgraph)

### 8.1 개념

서브그래프는 **하나의 노드 내부**에서 복잡한 워크플로우가 필요할 때 사용합니다.

```
┌─────────────────────────────────────────────────────────┐
│  analyze_data (노드)                                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │  [서브그래프]                                        ││
│  │  code_generation → code_execution → result_validation││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### 8.2 언제 사용하나?

| 상황 | 서브그래프 사용 |
|------|----------------|
| 노드 로직이 단순함 | ❌ 불필요 |
| 노드 내부에 여러 단계가 있음 | ✅ 권장 |
| 노드 내부에 조건 분기/반복이 있음 | ✅ 권장 |
| 여러 노드를 묶고 싶음 | ❌ 잘못된 사용 (메인 그래프에서 처리) |

### 8.3 주의사항

⚠️ **서브그래프 ≠ 여러 노드 묶기**

각 노드(`analyze_data`, `evaluate_results` 등)는 **독립적인 에이전트**입니다. 이들을 하나의 서브그래프로 묶지 마세요.

```
# ❌ 잘못된 사용
analysis_subgraph = analyze + evaluate + finalize  # 독립 에이전트를 묶음

# ✅ 올바른 사용
analyze_data 노드 내부에서:
  - 코드 생성 단계
  - 코드 실행 단계
  - 결과 검증 단계
  → 이것을 서브그래프로 구성
```

### 8.4 템플릿 위치

```
src/agent/subgraphs/
└── _template/              ← 팀원용 복사 템플릿
    ├── __init__.py
    ├── state.py            # State 정의 (Input/Internal/Output)
    ├── graph.py            # 워크플로우 정의
    └── nodes/
        ├── __init__.py
        ├── process_input.py
        └── process_output.py
```

### 8.5 사용 방법

1. `_template` 폴더를 복사하여 자신의 서브그래프 생성
2. `state.py`에서 State 정의
3. `nodes/`에 노드 함수 구현
4. `graph.py`에서 워크플로우 구성
5. 메인 노드에서 서브그래프 호출

```python
# 메인 노드에서 서브그래프 호출 예시
from src.agent.subgraphs.my_subgraph import create_my_subgraph

def analyze_data(state: AgentState) -> AgentState:
    subgraph = create_my_subgraph()
    
    # 서브그래프 입력 구성
    sub_input = {
        "input_data": state["clean_data"],
        # ... 기타 필드
    }
    
    # 서브그래프 실행
    sub_result = subgraph.invoke(sub_input)
    
    # 결과 반환
    return {
        "analysis_results": [sub_result["output_result"]],
        "steps_log": sub_result["steps_log"]
    }
```

---

## 9. 데이터 플로우 예시

### 9.1 CSV 파일 처리

```
Input: "data.csv"
  → load_data: CSV 로드, file_type="tabular", raw_data=df.to_dict()
  → preprocess_data: 전처리, clean_data 생성
  → analyze_data: LLM이 Python 코드 생성 → REPL 실행 → 분석 결과
  → evaluate_results: tabular 기준으로 평가
  → (APPROVE) generate_report: 데이터 분석 보고서 생성
  → human_review: 사람 확인
  → END
```

### 9.2 PDF 파일 처리

```
Input: "report.pdf"
  → load_data: PDF 텍스트 추출 (또는 Gemini Vision), file_type="document"
  → analyze_document: LLM이 요약/키워드/수치/인사이트 추출
  → evaluate_results: document 기준으로 평가
  → (REJECT) analyze_document: 피드백 반영하여 재분석
  → evaluate_results: 재평가
  → (APPROVE) generate_report: 문서 분석 보고서 생성
  → human_review: 사람 확인
  → END
```

### 9.3 DOCX 파일 처리

```
Input: "meeting_notes.docx"
  → load_data: python-docx 텍스트 추출, file_type="document"
  → analyze_document: LLM 분석
  → evaluate_results → generate_report → human_review → END
```

---

## 10. 확장 포인트

| 확장 | 방법 |
|------|------|
| 새 파일 형식 (e.g., XLSX, JSON) | `load_data.py`에 추출 함수 추가 + 적절한 `file_type` 분류 |
| 새 분석 노드 | `nodes/` 폴더에 노드 추가 → `graph.py`에 라우팅 확장 |
| 서브그래프 | `src/agent/subgraphs/_template/` 복사하여 독립 워크플로우 생성 |
| 보고서 형식 확장 (HTML, PPT) | `generate_report.py`에 형식 분기 추가 |

---

## 11. 참고 파일

| 파일 | 설명 |
|------|------|
| `src/graph.py` | LangGraph 워크플로우 정의 (라우팅 로직 포함) |
| `src/agent/state.py` | AgentState 정의 (file_type 필드 포함) |
| `src/agent/nodes/load_data.py` | CSV/PDF/DOCX 파일 로드 |
| `src/agent/nodes/analyze_document.py` | 문서 분석 노드 |
| `src/agent/nodes/analyze_data.py` | 데이터 분석 노드 |
| `src/agent/nodes/evaluate_results.py` | 품질 평가 노드 (file_type 분기) |
| `src/agent/nodes/generate_report.py` | 보고서 생성 노드 (file_type 분기) |
| `src/core/llm_factory.py` | LLM 생성 + Langfuse 통합 |
| `src/core/observe.py` | Langfuse observability (langfuse_session, @observe) |

