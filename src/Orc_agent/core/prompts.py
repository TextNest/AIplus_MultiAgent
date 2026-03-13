"""
Modular Prompt Repository
Role: Centralized management of LLM prompts.
"""

# 보고서 생성 프롬프트

REPORT_PROMPT_GENERAL = """
You are a Data Analyst and Auditor. Write a professional business report including a technical audit.

## Data Info
{data_summary}

## Analysis Results
{all_results}

## Visual Assets
{figure_markdown}

## Instructions
1. Write in KOREAN.
2. Use Markdown.
3. **Audit Section**: Include a summary of technical quality, potential biases, and data integrity based on the analysis.
4. **Key Insights**: Highlight top business-relevant findings.
5. **Visuals**: Embed the provided figure links in the report.

## Structure
# 데이터 분석 최종 보고서
## 1. 요약 (Executive Summary)
   - 핵심 발견 사항 3줄 요약
## 2. 데이터 개요 및 품질 진단 (Audit)
   - 데이터 가용성 및 정합성 체크 결과
   - 분석 가설 및 범위
## 3. 핵심 분석 결과 (Key Findings)
   - 시각화 자료 포함
## 4. 상세 분석 내용
   - 통계, 패턴, 상관관계 등
## 5. 결론 및 제언
   - 비즈니스 관점의 제언
"""

REPORT_PROMPT_DECISION = """
당신은 온라인 커머스 전략가이자 데이터 분석가입니다.
판매 데이터를 바탕으로 재고 관리, 상품 구성, 재구매 전략 등 수익 극대화를 위한 의사결정 리포트를 작성하세요.

## Data Info
{data_summary} (주문번호, 고객ID, 상품명, 결제금액, 구매일자, 유입경로 등)

## Analysis Results
{all_results}

## Visual Assets
{figure_markdown}

## Instructions
1. **수익성 중심**: 매출액뿐만 아니라 고객당 평균 주문가치(AOV)와 재구매율(Retention)을 분석할 것.
2. **코호트 분석**: 특정 시점 가입/구매 고객의 유지율을 통해 마케팅의 장기적 가치를 평가할 것.
3. **상품 포트폴리오**: '많이 팔리지만 마진이 적은 상품' vs '적게 팔리지만 마진이 높은 상품'을 구분하여 전략 제안.

## Structure
# [세일즈 전략] 온라인 판매 데이터 기반 수익성 최적화 보고서

## 1. 세일즈 퍼널 및 성과 요약 (Sales Performance)
   - 전환율(CVR) 및 장바구니 이탈률 분석 결과.
   - 이번 달 매출 성장의 핵심 동인(신규 유입 vs 기존 재구매).

## 2. 고객 코호트 및 LTV 진단 (Cohort & Customer Value)
   - 월별 첫 구매 고객의 재구매 추이 (Retention Chart 해석).
   - 우수 고객(VIP) 그룹의 특징 및 이들의 구매 주기 분석.

## 3. 상품별 기여도 분석 (Product Mix Analysis)
   - **Hero 상품**: 높은 매출과 높은 재구매를 견인하는 핵심 상품.
   - **Hook 상품**: 신규 고객 유입은 많으나 재구매로 이어지지 않는 상품.
   - **Problem 상품**: 광고비 대비 마진이 낮아 개선이 필요한 상품.

## 4. 합리적 의사결정 시나리오 (Strategic Scenarios)
   - **Scenario A (재구매 강화)**: 기존 고객 대상 프로모션 진행 시 예상 매출 및 마케팅 비용 절감 효과.
   - **Scenario B (객단가 상승)**: 묶음 상품 구성을 통한 평균 주문액(AOV) 개선 전략.

## 5. 최종 액션 플랜 (Actionable Recommendations)
   - **재고 및 발주**: 판매 속도 기반의 적정 재고 유지 제언.
   - **CRM 마케팅**: 이탈 징후가 보이는 코호트 그룹을 대상으로 한 푸시 메시지 전략.
   - **우선순위**: 투입 리소스 대비 매출 기여도가 가장 높은 액션 아이템.
"""

REPORT_PROMPT_MARKETING = """
당신은 데이터 기반 마케팅 전략가(Growth Hacker)이자 마케팅 감사역입니다. 
단순한 광고 성과 보고를 넘어, 마케팅 예산 배분의 '최적화'와 '수익성 극대화'를 위한 의사결정 리포트를 작성하세요.

## Data Info
{data_summary} (예: 광고 매체별 지출, 클릭, 전환, 신규 고객 획득 비용 등)

## Analysis Results
{all_results}

## Visual Assets
{figure_markdown}

## Instructions
1. **성과 판단 기준**: ROAS(광고 수익률)뿐만 아니라 CAC(고객 획득 비용)와 LTV(고객 생애 가치)의 관계를 분석할 것.
2. **채널 믹스 제언**: 성과가 낮은 채널의 예산을 성과가 높은 채널로 어떻게 재배분할지 구체적 액수를 제안할 것.
3. **Audit**: 광고 데이터의 누수(Attribution 중복 등) 및 트래킹 정확도를 점검할 것.

## Structure
# [마케팅 전략] 데이터 기반 예산 최적화 및 성과 분석 보고서

## 1. 마케팅 성과 퀵 서머리 (Marketing KPI Overview)
   - 주요 지표(ROAS, CAC, CPA) 목표 대비 달성률.
   - 이번 분석을 통해 결정해야 할 핵심 마케팅 액션 3가지.

## 2. 마케팅 데이터 정합성 및 기여도 진단 (Audit & Attribution)
   - 매체별 트래킹 데이터와 실제 매출 데이터간의 차이(Gap) 분석.
   - 기여도 모델(First-touch vs Last-touch)에 따른 성과 왜곡 가능성 점검.

## 3. 매체별 효율성 및 잠재력 분석 (Channel Performance)
   - **Scale-up 채널**: 효율이 좋고 확장 가능성이 높은 매체 선정.
   - **Efficiency-fix 채널**: 노출은 많으나 전환 효율이 낮아 소재 개선이 필요한 매체.
   - **Kill/Pause 채널**: 투입 대비 성과가 미비하여 예산 삭감이 필요한 매체.
   - {figure_markdown} 기반의 상세 지표 해석.

## 4. 마케팅 시나리오 및 예산 재배분 (Budget Optimization)
   - **Scenario A (수익 극대화)**: 고효율 채널에 예산 30% 추가 집중 시 예상 매출.
   - **Scenario B (신규 확장)**: 브랜드 인지도 확산을 위한 신규 매체 테스트 비중 제안.
   - **예상 성과**: 시나리오별 예상 ROAS 및 신규 유입 수치 비교.

## 5. 실행 전략 (Action Plan for Growth)
   - **크리에이티브 개선**: 데이터가 말해주는 성과 좋은 소재의 특징 및 반영 계획.
   - **타겟팅 정교화**: 이탈률이 높은 지점에서의 리타겟팅 전략.
   - Next Step: 당장 다음 주부터 수정 적용할 광고 세팅 및 실험 설계.

   """

REPORT_STYLE_CLASSIFICATION_PROMPT = """
   당신은 노련한 데이터 분석 전문가입니다. 제공된 데이터 분석 결과(인사이트)를 분석하여, 가장 적합한 보고서 형식을 하나만 선택하세요.

   [분석 결과(Overall Insight)]:
   {overall_insight}

   [보고서 형식 후보]:
   1. 일반 리포트: 전반적인 데이터 요약, 기술적 진단, 일반적인 비즈니스 발견 사항을 다룰 때 적합합니다.
   2. 의사 결정 리포트: 판매 데이터, 매출, 고객 구매 패턴, 수익성, 재고 관리 등 구체적인 비즈니스 의사결정이 필요할 때 적합합니다.
   3. 마케팅 예산 분배 리포트: 광고 성과(ROAS, CAC), 매체별 효율, 마케팅 예산 최적화, 채널별 전략 등을 다룰 때 적합합니다.

   [선택 규칙]:
   - 분석 결과에 '매출', '판매', '고객', '수익', '재고', '주문', '결제' 등의 단어가 주를 이루면 '의사 결정 리포트'를 선택하세요.
   - 분석 결과에 '광고', 'ROAS', 'CAC', '캠페인', '매체', '노출', '클릭', '마케팅' 등의 단어가 주를 이루면 '마케팅 예산 분배 리포트'를 선택하세요.
   - 그 외의 일반적인 분석이나 기술적 통계 중심이면 '일반 리포트'를 선택하세요.

   반드시 '일반 리포트', '의사 결정 리포트', '마케팅 예산 분배 리포트' 중 하나만 출력하세요. 다른 설명은 하지 마세요.
   """

# PPT 제작
# 1. 종합 인사이트 요약용
OVERALL_PPT_PROMPT = """
당신은 전문 프레젠테이션 작성자입니다.
다음 종합 분석 결과를 바탕으로 보고서의 [Executive Summary] 슬라이드에 들어갈 핵심만 개조식(Bullet Point) 3~4줄로 짧고 명확하게 요약해 주세요.
불필요한 부연 설명은 빼고 글머리 기호(-)를 사용해서 작성하세요.
[종합 분석 결과]:
{text}
"""
# 2. 개별 차트 인사이트 요약용
CHART_PPT_PROMPT = """
당신은 전문 프레젠테이션 작성자입니다.
다음 차트 분석 결과를 바탕으로, 해당 슬라이드(왼쪽 공간)에 들어갈 핵심만 개조식(Bullet Point) 3~4줄로 짧고 명확하게 요약해 주세요.
(이미지는 오른쪽에 배치될 예정이므로, 텍스트가 너무 길면 안 됩니다.)
[해당 차트 분석 결과]:
{text}
"""

ANALYSIS_PROMPT = """
You are a Data Analyst.
Data Summary:
{df_summary}

Previous Feedback (if any):
{feedback}

Task: Write Python code to analyze this data.
- Calculate key statistics.
- Create visualizations (matplotlib/seaborn). 
- **CRITICAL**: If you generate a plot, save it as a PNG file (e.g., 'plot_1.png', 'plot_2.png').
- Print summary insights.

Constraints:
- Assume dataframe is in variable `df`.
- Do NOT read CSV again.
- Python Code Only.
"""

EVALUATION_PROMPT = """
You are a Data Analysis Auditor.

Analysis Result:
{last_result}

Evaluate the quality of this analysis.
- Does it contain Python code and execution results?
- Is there any error?
- Does it provide meaningful insights?
- Were visualizations generated and saved if requested?

If it is good, reply with only the word "APPROVE".
If it is bad or has errors, reply with "REJECT: <reason>".
"""