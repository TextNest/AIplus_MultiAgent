import sys
import os
from dotenv import load_dotenv
# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 환경 변수 로드 (DUMMY_MODE=True 설정 확인 필요)
load_dotenv()
from src.Orc_agent.Graph.Main_graph import create_main_graph
from src.core.config import config
def main():
    print(">>> Orc Agent 시뮬레이션 시작 (Dummy Mode)")
    
    # 그래프 생성
    app = create_main_graph()
    
    # 모의 초기 상태 (Dummy 데이터)
    initial_state = {
        "file_path": "fake_marketing_data.csv",
        "file_type": "tabular", # 분기 처리를 위해 필요
        "user_query": "데이터 분석 후 PDF 보고서 작성해줘",
        "analysis_results": {"summary": "더미 분석 결과입니다.", "details": "..."}, # Orc_agent 구조에 맞춤
        "clean_data": {"col1": [1, 2, 3], "col2": [4, 5, 6]}, # 더미 데이터
        "report_format": "pdf", # 보고서 포맷 지정
        "retry_count": 0,
        "steps_log": [],
        "generated_formats": [],
        "history": [] # 에러 방지용 빈 리스트
    }
    
    config_run = {"configurable": {"thread_id": "sim_thread_1"}}
    
    print(f">>> 초기 설정: {initial_state['user_query']}")
    
    # 실행
    # stream 모드로 각 단계 진행 상황 출력
    try:
        for event in app.stream(initial_state, config=config_run):
            for key, value in event.items():
                print(f"\n--- [Node: {key}] ---")
                
                # 로그 출력
                if "steps_log" in value and value["steps_log"]:
                    print(f"Log: {value['steps_log'][-1]}")
                
                # 수퍼바이저 결정 확인
                if "next_worker" in value:
                    print(f"Supervisor Decision: {value['next_worker']}")
                    
        print("\n>>> 시뮬레이션 완료: 성공적으로 모든 단계를 통과했습니다.")
        
    except Exception as e:
        print(f"\n>>> 시뮬레이션 중단: 에러 발생\n{e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    # DUMMY_MODE 강제 활성화 (테스트용)
    os.environ["DUMMY_MODE"] = "True"
    main()