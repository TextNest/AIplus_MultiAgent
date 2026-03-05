import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. Project Root Path setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 2. Load Environment Variables (.env)
load_dotenv(dotenv_path=project_root / ".env")

# Verify API Key
if not os.environ.get("GOOGLE_API_KEY"):
    print("⚠️ WARNING: GOOGLE_API_KEY not found in environment variables.")
    print("Please ensure it is set in your .env file.")

# 3. Import necessary modules
from src.Orc_agent.Node.sub_node.generate_report import generate_content
from langchain_core.runnables import RunnableConfig

def test_report_generation():
    print("\n" + "="*50)
    print("  Report Generation Prompt Test")
    print("="*50)
    print("1: 일반 리포트 (General Report)")
    print("2: 의사 결정 리포트 (Decision Report)")
    print("3: 마케팅 예산 분배 리포트 (Marketing Report)")
    print("-" * 50)
    
    choice = input("테스트할 리포트 스타일 번호를 입력하세요 (1-3): ").strip()
    
    style_map = {
        "1": "일반 리포트",
        "2": "의사 결정 리포트",
        "3": "마케팅 예산 분배 리포트"
    }
    
    selected_style = style_map.get(choice, "일반 리포트")
    
    # 4. Define Mock Sample Data per Style
    if choice == "2":
        # 의사 결정 리포트용 샘플 데이터
        analysis_results_raw = {
            "overall_0": {
                "insight": "고객 데이터 분석 결과, 상위 10% 고객이 전체 매출의 60%를 차지하는 파레토 법칙이 관찰됨. 재구매 주기는 평균 45일이며, 첫 구매 후 30일 이내 재구매 유도시 리텐션이 2.5배 상승함.",
                "img_path": None
            },
            "figure_0_0.png": {
                "insight": "상품별 마진 기여도 분석 결과, A상품은 매출은 높으나 마진율이 5% 미만으로 'Hook' 상품군에 속함. 반면 B상품은 매출은 적지만 마진율 35%로 수익성 개선의 핵심임.",
                "img_path": "figure_0_0.png"
            }
        }
    elif choice == "3":
        # 마케팅 예산 분배용 샘플 데이터
        analysis_results_raw = {
            "overall_0": {
                "insight": "본 차수에 추가된 3개의 시각화(표 기반 요약, 시계열 비교, CPC-전환 산점도)가 캠페인-채널 조합의 비용효율성 차이를 명확히 드러냄. 특히 New_Product_Launch 캠페인의 Instagram Story 조합이 비용대비 전환효율 측면에서 가장 돋보이며, CPA 9.29로 최저치를 기록함.",
                "img_path": None
            },
            "figure_0_0.png": {
                "insight": "비효율적 대규모 지출 사례인 Retargeting_VIP + YouTube Video는 CPA가 57.77로 가장 높게 나타나 예산 삭감이 시급함.",
                "img_path": "figure_0_0.png"
            }
        }
    else:
        # 일반 리포트용 샘플 데이터
        analysis_results_raw = {
            "overall_0": {
                "insight": "전체 데이터 요약 결과입니다. 수집된 데이터는 2025년 상반기 실적을 포함하고 있으며, 데이터 정합성 검사 결과 누락된 값 없이 깨끗한 상태로 확인되었습니다.",
                "img_path": None
            },
            "figure_0_0.png": {
                "insight": "월별 매출 추이 시각화 결과입니다. 계절성 요인으로 인해 5월 가정의 달 매출이 급격히 상승하는 패턴을 보입니다.",
                "img_path": "figure_0_0.png"
            }
        }

    # generate_content 내부의 .join(analysis_results) 처리를 위해 insight 텍스트 리스트로 변환
    analysis_results = [v["insight"] for v in analysis_results_raw.values()]

    sample_state = {
        "analysis_results": analysis_results,
        "file_path": "fake_marketing_data.csv",
        "report_style": selected_style,
        "report_format": ["markdown"]
    }
    
    # 5. Config Setup (Empty config for testing)
    config = RunnableConfig(configurable={"thread_id": "test_thread"})
    
    # 6. Call the generation node directly
    print(f"\n--- [Selected Style: {selected_style}] ---")
    result = generate_content(sample_state, config)
    
    # 7. Print Results
    if "final_report" in result:
        print("\n[Generated Report Content]\n")
        print("="*60)
        print(result["final_report"])
        print("="*60)
    
    if "steps_log" in result:
        print("\n[Steps Log]:", result["steps_log"])

if __name__ == "__main__":
    test_report_generation()
