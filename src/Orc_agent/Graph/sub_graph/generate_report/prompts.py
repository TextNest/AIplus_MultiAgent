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