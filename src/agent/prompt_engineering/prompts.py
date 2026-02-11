"""
Modular Prompt Repository
Role: Centralized management of LLM prompts.
"""


REPORT_PROMPT = """
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