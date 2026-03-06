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
from src.Orc_agent.Node.sub_node.generate_report import generate_content, classify_report_style
from langchain_core.runnables import RunnableConfig

def test_report_generation():
    print("\n" + "="*50)
    print("  Report Generation & Classification Test")
    print("="*50)
    print("0: AI 자동 판단 (추천) (Auto Classification)")
    print("1: 일반 리포트 (General Report)")
    print("2: 의사 결정 리포트 (Decision Report)")
    print("3: 마케팅 예산 분배 리포트 (Marketing Report)")
    print("-" * 50)
    
    choice = input("테스트할 리포트 스타일 번호를 입력하세요 (0-3): ").strip()
    
    style_map = {
        "0": "AI 자동 판단 (추천)",
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
                "insight": "전체 제품군 중 'Home-Office' 카테고리가 매출 성장세와 수익성 면에서 압도적이나, 'Electronics' 일부 품목의 재고 회전율 저하로 인해 전체 현금 흐름에 병목이 발생함. 고수익 제품군으로의 재고 확보 우선순위 조정과 비효율 품목의 프로모션 처분이 시급함.",
                "img_path": None
            },
            "figure_1_0.png": {
                "insight": "카테고리별 매출 및 영업이익률 현황:\n- 최고 효자 품목: Home-Office (Ergonomic Chair). 매출 125,000 USD, 영업이익률 28%. 높은 객단가와 안정적인 마진을 유지 중.\n- 외형 성장형 품목: Mobile Accessories. 매출은 전분기 대비 40% 성장했으나, 마진율이 12%로 낮아 박리다매 전략이 지속되고 있음.\n- 수익성 악화 품목: Standard Laptops. 할인 경쟁 심화로 영업이익률이 5% 미만으로 급락함.",
                "img_path": "webapp/static/img/report_v1/sales_margin_summary.png"
            },
            "figure_1_1.png": {
                "insight": "고객 구매 패턴 및 재구매율 분석:\n- 신규 고객의 65%가 'Starter Kit'을 통해 유입되며, 이들의 2차 구매 전환율은 22% 수준임.\n- VIP 고객(상위 5%)의 경우 평균 구매 주기가 14일로 매우 짧으며, 주로 프리미엄 라인에 집중됨.\n- 이탈 징후 구간: 마지막 구매 후 45일이 경과한 고객의 재방문율이 급격히 하락하므로 리마인드 쿠폰 발행이 필요함.",
                "img_path": "webapp/static/img/report_v1/customer_retention.png"
            },
            "figure_1_2.png": {
                "insight": "재고 수준 및 회전율(Inventory Health):\n- 과잉 재고: Winter Apparel. 시즌 종료에도 불구하고 재고 보유량이 적정 수준의 180%를 상회함. 즉각적인 클리어런스 세일 권고.\n- 품절 위험: Premium Coffee Beans. 최근 수요 예측 실패로 안전 재고가 3일치 미만으로 남음. 발주 주기 단축 필요.",
                "img_path": "webapp/static/img/report_v1/inventory_rotation.png"
            }
        }
    elif choice == "3" or choice == "0":
        # 마케팅 예산 분배용 샘플 데이터 (0번 자동판단 테스트용으로도 사용)
        analysis_results_raw = {
            "overall_0": {
                "insight": "최근 30일간 Google Search와 Meta Ads의 성과가 상반됨. Google Search는 고관여 키워드 선점으로 높은 ROAS를 유지 중이나, Meta Ads는 소재 피로도로 인해 CAC가 전월 대비 15% 상승함. 성과가 검증된 Google Search의 예산을 20% 증액하고, Meta는 소재 교체 기간 동안 유지 보수 예산으로 편성할 것을 제안함.",
                "img_path": None
            },
            "figure_2_0.png": {
                "insight": "매체별 KPI 요약 표:\n- Google Search (Brand): ROAS 850%, CAC 12.5 USD. 브랜드 키워드 보호를 통해 가장 안정적인 효율 달성.\n- Meta Ads (Re-targeting): ROAS 420%, CAC 25.0 USD. 장바구니 이탈자 대상 캠페인의 효율이 좋으나 모수 고갈 징후 보임.\n- TikTok (Viral): ROAS 180%, CAC 45.0 USD. 직접 전환보다는 신규 유입(Traffic) 및 브랜드 인지도 측면에서 기여도가 높음.",
                "img_path": "webapp/static/img/marketing_v1/channel_performance_table.png"
            },
            "figure_2_1.png": {
                "insight": "A/B 테스트 결과 (랜딩 페이지 최적화):\n- 시안 A (상품 강조형): 구매 전환율 3.2%, 이탈률 45%.\n- 시안 B (혜택 강조형 - '첫 구매 10%'): 구매 전환율 4.8%, 이탈률 38%.\n- 결론: 혜택 중심의 시안 B가 모든 유입 채널에서 1.5배 높은 효율을 보이므로 전면 교체 적용 예정.",
                "img_path": "webapp/static/img/marketing_v1/ab_test_result.png"
            },
            "figure_2_2.png": {
                "insight": "시간대별/요일별 성과 변동 시계열:\n- 주말(토, 일) 저녁 21시~23시 구간에 모바일 결제 비중이 70% 이상 집중됨. 해당 골든 타임에 비딩(Bidding) 가중치를 +20% 적용하여 노출 점유율 확보 전략이 유효할 것으로 판단됨.",
                "img_path": "webapp/static/img/marketing_v1/time_series_efficiency.png"
            }
        }
    else:
        # 일반 리포트용 샘플 데이터
        analysis_results_raw = {
                "overall_0": {
                    "insight": "본 차수에 추가된 3개의 시각화(표 기반 요약, 시계열 비교, CPC-전환 산점도)가 캠페인-채널 조합의 비용효율성 차이를 명확히 드러냄. 특히 New_Product_Launch 캠페인의 Instagram Story 조합이 비용대비 전환효율 측면에서 가장 돋보이며, 고급 캠페인에서도 일부 조합은 CPA가 높아 재배분 필요성이 큼. ROAS 데이터가 없다는 한계 아래, 각 조합의 CPA, CVR, CPC의 관계를 바탕으로 예산 재배분 포인트를 도출하는 것이 핵심임.",
                    "img_path": None
                },
                "figure_0_0.png": {
                    "insight": "새로 제공된 표에서 Campaign-Channel별 KPI를 한눈에 비교 가능. 핵심 포인트는 다음과 같음.\n- 최고 효율 조합: New_Product_Launch + Instagram Story. Impressions 271,541, Spend_USD 6,268.41, Clicks 10,486, Conversions 675, CTR 3.86%, CPC 0.60, CVR 6.44%, CPA 9.29. 이 조합은 낮은 CPC와 높은 CTR/CVR을 통해 전환당 비용이 비교적 낮은 편에 속함.\n- 안정적 성과의 대형 조합: Brand_Awareness_KR + Facebook Ads도 비교적 낮은 CPA 8.47(Convs 148, Impr 95,539, Spend 1,254.27)으로 관찰되나, 전환 규모는 New_Product_Launch Instagram Story만큼 크지 않음.\n- 비효율적 대규모 지출 사례: Retargeting_VIP + YouTube Video CPA 57.77로 가장 높은 편이며, Impressions 67,659에 불구하고 Conversions 104에 머물러 ROI 측면 부담이 큼. Summer_Sale_2025 YouTube Video도 CPA 36.76으로 높은 편.\n- 고CVR 구간의 다양성: New_Product_Launch YouTube Video의 CVR 8.57%로 CVR은 높으나 Conversions(349)와 Impressions 대비 전체 효율은 CPA가 비교적 높은 편. 같은 Campaign의 Instagram Story 조합은 CVR 6.44%로 우수하나 CPA가 9.29로 비교적 낮은 편.\n- 요약 포인트: CPA가 낮고 CVR이 높은 조합(특히 New_Product_Launch Instagram Story, Brand_Awareness_KR Facebook Ads)은 예산 재배분의 최상위 후보이며, 반대로 CPA가 높은 조합은 재배치 대상임.",
                    "img_path": "webapp/static/img/005a106a-664d-4f15-843f-f67e84e078da/figure_0_0.png"
                },
                "figure_0_1.png": {
                    "insight": "시각화가 요약한 캠페인-채널별 KPI를 한 눈에 확인하기 쉽게 구성된 표의 확장판으로 해석 가능. 주요 인사이트:\n- New_Product_Launch Instagram Story가 여전히 가장 낮은 CPA(약 9.29)와 높은 CVR(6.44%)를 기록하며, Impressions가 271k로 대규모 노출과 결합해 효율적 전환을 견인하는 대표 사례임.\n- Brand_Awareness_KR Facebook Ads의 CPA는 8.47로 낮은 편이지만, Conversions(148) 대비 전체 노출이 95k로 상대적 규모가 작아 전체 영향력은 제한적일 수 있음.\n- Google Search를 포함한 다수의 조합에서 CVR은 5%대 초반에서 8%대까지 다양하게 나타나지만, CPA는 15~36 사이로 분포. 특히 Retargeting_VIP 계열의 YouTube/Instagram 조합은 CPA가 30+로 상대적으로 비효율적임.\n- 종합적으로 보면, CPA와 CVR 간의 균형이 잘 맞는 조합(예: New_Product_Launch Instagram Story, Brand_Awareness_KR Facebook Ads)이 예산 운용의 중심이 되어야 하며, CPA가 높은 조합은 재배치 우선순위에서 하향 조정이 필요함.",
                    "img_path": "webapp/static/img/005a106a-664d-4f15-843f-f67e84e078da/figure_0_1.png"
                },
                "figure_0_2.png": {
                    "insight": "일간 시계열 시각화(CPC-노출/지출 비교)에서 관찰 가능한 패턴:\n- 초기(1월 초)에는 Impressions가 약 60k~120k 범위로 변동하면서 Spend_USD도 상당히 상승/하강하는 양상을 보이며, 대략적인 상관관계가 존재함. 다만 Impressions의 절대치와 Spend의 절대치가 1:1로 움직이지 않는 구간도 있어 조합별 효과 차이가 있음.\n- 2월 말~3월 초에 Spend_USD가 급격히 증가하는 구간이 보이며, 이때 Impressions도 함께 증가하는 경향이 관찰됨. 이는 대규모 캠페인 프로모션이 시작되었거나 예산 증액 시점과 맞물린 시점과 일치하는 것으로 보임.\n- 결론적으로, 예산 증가가 즉시 노출 증가로 이어지는 경향은 확인되나, 캠페인별 CPA/CVR의 차이가 여전히 존재하므로, 시점별 캠페인 구성(어떤 채널/캠페인이 집중되었는지)과 결합해 ROI 영향 분석이 필요합니다.\n- 한계: Revenue 데이터가 없으므로 ROAS를 함께 해석하기 어렵고, 시계열에 따른 변동의 원인을 캠페인별 세부 설정으로 추가 확인이 필요합니다.",
                    "img_path": "webapp/static/img/005a106a-664d-4f15-843f-f67e84e078da/figure_0_2.png"
                }
            }


    # 실제 state와 동일한 구조로 전달합니다.
    analysis_results = analysis_results_raw

    sample_state = {
        "analysis_results": analysis_results,
        "file_path": "fake_marketing_data.csv",
        "report_style": selected_style,
        "report_format": ["markdown"]
    }
    
    # 5. Config Setup
    config = RunnableConfig(configurable={"thread_id": "test_thread", "session_id": "test_session"})
    
    # 6. Test Logic
    print(f"\n--- [Initial Style: {selected_style}] ---")
    
    # 만약 자동 판단인 경우 classify_report_style 노드 먼저 실행
    if selected_style == "AI 자동 판단 (추천)":
        print("\n[Action] Running classify_report_style node...")
        class_result = classify_report_style(sample_state, config)
        new_style = class_result.get("report_style")
        print(f"-> AI Classified Style: {new_style}")
        sample_state["report_style"] = new_style
    
    print("\n[Action] Running generate_content node...")
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
