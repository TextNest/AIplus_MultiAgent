
import os
from dotenv import load_dotenv

# .env 로드 (API 키 등)
load_dotenv()

import sys
# src 폴더를 path에 추가하여 모듈 import 가능하게 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Orc_agent.Graph.Main_graph import create_main_graph

def main():
    # 1. 그래프 생성
    print(">>> 그래프 생성 중...")
    app = create_main_graph()
    
    # 2. 초기 State 설정
    # 주의: preprocessing_data와 user_query는 필수입니다.
    initial_state = {
        "file_path": "fake_marketing_data.csv", # 테스트용 더미 경로
        "user_query": "데이터의 전반적인 추세를 분석하고 시각화해줘"
    }
    
    # 3. 그래프 실행
    print(f">>> 실행 시작: {initial_state['user_query']}")
    config = {"configurable": {"thread_id": "test_thread_2", "user_id": "test_user"}}
    
    # stream 모드로 실행하여 중간 로그 확인
    first = True
    while True:
        if first:
            for event in app.stream(initial_state, config=config):
                for key, value in event.items():
                    print(f"\n--- Node: {key} ---")
                    print(value)
            first = False
                
        print("\n>>> 실행 완료")

        # 인터럽트 상태 확인 (User Request)
        state = app.get_state(config)
        print(f"다음 실행할 노드: {state.next[0] if state.next else 'None'}")
        input_choice = input("현재 그래프를 보고 피드백 여부를 알려주세요(완료/수정/추가):")
        if input_choice in ["수정","추가"]:
            input_feedback = input("피드백 내용을 알려주세요.")
        else:
            input_feedback = ""
        app.update_state(config, {"user_choice": input_choice ,"feedback": input_feedback}, as_node=state.next[0])
        for event in app.stream(None, config=config):
            pass
if __name__ == "__main__":
    main()
