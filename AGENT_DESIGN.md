# Agent Design Document

> 멀티 에이전트 데이터 분석 시스템 설계 문서

---

## 1. 시스템 개요

### 1.1 목적
CSV 데이터를 입력받아 자동으로 분석하고, 품질 평가를 거쳐 보고서를 생성하는 멀티 에이전트 시스템

### 1.2 핵심 특징
- **LLM 기반 분석**: LLM이 Python 코드를 생성하고 REPL에서 실행
- **Self-Critique**: 분석 결과를 LLM이 평가하고 부족하면 재시도
- **Human-in-the-Loop**: 최종 보고서를 사람이 검토하고 피드백 제공

---

## 2. 아키텍처

### 2.1 워크플로우 다이어그램

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  load_data  │────▶│ preprocess_data  │────▶│ analyze_data │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                    │
                                                    ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    END      │◀────│   human_review   │◀────│ generate_report  │
└─────────────┘     └────────┬─────────┘     └────────▲─────────┘
       ▲                     │                        │
       │                     │ REJECT                 │ APPROVE
       │                     ▼                        │
       │               ┌───────────┐                  │
       │               │  (재분석)  │                  │
       │               └─────┬─────┘                  │
       │                     │                        │
       │                     ▼                        │
       │              ┌──────────────────┐            │
       └──────────────│ evaluate_results │────────────┘
          APPROVE     └──────────────────┘
        (max retry)          │
                             │ REJECT (retry < 3)
                             ▼
                       ┌──────────────┐
                       │ analyze_data │
                       └──────────────┘
```

### 2.2 상태 흐름

| 단계 | State 변화 |
|------|-----------|
| 1. load_data | `file_path` → `raw_data` |
| 2. preprocess_data | `raw_data` → `clean_data` |
| 3. analyze_data | `clean_data` → `analysis_results` |
| 4. evaluate_results | `analysis_results` → `evaluation_feedback` |
| 5. generate_report | `analysis_results` → `final_report` |
| 6. human_review | `human_feedback` → END 또는 재시작 |

---

## 3. 노드 상세 설계

### 3.1 load_data (데이터 로드)

| 항목 | 내용 |
|------|------|
| **역할** | CSV 파일을 읽어 dictionary로 변환 |
| **입력** | `file_path: str` |
| **출력** | `raw_data: dict`, `steps_log` |
| **LLM** | ❌ 불필요 |
| **구현 상태** | ✅ 완료 |

```python
# 핵심 로직
df = pd.read_csv(file_path)
return {"raw_data": df.to_dict(), "steps_log": [...]}
```

---

### 3.2 preprocess_data (전처리)

| 항목 | 내용 |
|------|------|
| **역할** | 결측치 처리, 타입 변환, 이상치 제거 |
| **입력** | `raw_data: dict` |
| **출력** | `clean_data: dict`, `steps_log` |
| **LLM** | ❌ 불필요 (룰 기반) |
| **구현 상태** | 🟡 스켈레톤 |

**TODO (팀원 B)**:
- [ ] 결측치 처리 전략 (dropna / fillna)
- [ ] 숫자형 컬럼 자동 감지 및 변환
- [ ] 이상치 탐지 및 처리 (IQR, Z-score 등)

---

### 3.3 analyze_data (LLM 분석)

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

### 3.4 evaluate_results (품질 평가)

| 항목 | 내용 |
|------|------|
| **역할** | 분석 결과를 평가하고 APPROVE/REJECT 결정 |
| **입력** | `analysis_results: List[str]`, `clean_data: dict` |
| **출력** | `evaluation_feedback: str`, `steps_log` |
| **LLM** | ✅ 필수 (Critic 역할) |
| **구현 상태** | 🟢 예시 구현됨 |

**평가 기준**:
| 기준 | 설명 | 점수 |
|------|------|------|
| 완전성 | 주요 통계가 모두 포함되었는가? | 1-5 |
| 정확성 | 코드 에러 없이 실행되었는가? | 1-5 |
| 유용성 | 의미있는 인사이트가 있는가? | 1-5 |
| 명확성 | 이해하기 쉽게 정리되었는가? | 1-5 |

**결정 로직**:
- 평균 3점 이상 → `"APPROVE"`
- 평균 3점 미만 → `"REJECT: {구체적 피드백}"`

---

### 3.5 generate_report (보고서 생성)

| 항목 | 내용 |
|------|------|
| **역할** | 분석 결과를 Markdown 보고서로 변환 |
| **입력** | `analysis_results: List[str]`, `clean_data: dict` |
| **출력** | `final_report: str`, `steps_log` |
| **LLM** | ✅ 필수 (문서 작성) |
| **구현 상태** | 🟢 예시 구현됨 |

**보고서 구조**:
```markdown
# 데이터 분석 보고서

## 1. 요약 (Executive Summary)
## 2. 데이터 개요
## 3. 주요 발견사항
## 4. 상세 분석
## 5. 결론 및 권고사항
```

**확장 가능**:
- HTML 보고서
- PowerPoint (python-pptx)
- PDF (weasyprint)

---

### 3.6 human_review (HITL)

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
- 그 외 → `analyze_data`로 재시작

---

## 4. 상태 (AgentState)

```python
class AgentState(TypedDict):
    # 입력
    file_path: str
    raw_data: Optional[dict]
    
    # 처리된 데이터
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

### 4.1 List 자동 병합

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

### 5.1 evaluate_results → ?

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
        return "analyze_data"  # 재시도
```

### 5.2 human_review → ?

```python
def should_continue_human(state: AgentState):
    feedback = state.get('human_feedback')
    
    if feedback and "APPROVE" in feedback.upper():
        return END
    else:
        return "analyze_data"  # 처음부터 재분석
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

| 순위 | 노드 | 이유 |
|------|------|------|
| 1 | `analyze_data` | 핵심 분석 기능 |
| 2 | `evaluate_results` | 품질 보장 루프 |
| 3 | `generate_report` | 사용자 가치 전달 |
| 4 | `preprocess_data` | 데이터 품질 개선 |
| 5 | `human_review` | 저장/알림 확장 |

---

## 8. 참고 파일

| 파일 | 설명 |
|------|------|
| `src/graph.py` | LangGraph 워크플로우 정의 |
| `src/agent/state.py` | AgentState 정의 |
| `src/agent/nodes/*.py` | 각 노드 구현 |
| `src/core/llm_factory.py` | LLM 생성 + Langfuse 통합 |
| `AGENTS.md` | 코딩 에이전트용 가이드라인 |
