
import os
from dotenv import load_dotenv

load_dotenv()

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Orc_agent.Graph.Main_graph import create_main_graph
os.makedirs("img", exist_ok=True)
def main():

    print(">>> 그래프 생성 중...")
    app = create_main_graph()
    
    initial_state = {
        "file_path": "fake_marketing_data.csv", 
        "user_query": "데이터의 전반적인 추세를 분석하고 시각화해줘"
    }
    

    print(f">>> 실행 시작: {initial_state['user_query']}")
    config = {"configurable": {"thread_id": "test_thread_2", "user_id": "test_user"}}
    first = True
    while True:
        if first:
            for event in app.stream(initial_state, config=config):
                for key, value in event.items():
                    print(f"\n--- Node: {key} ---")
                    print(value)
            first = False
                
        print("\n>>> 실행 완료")


        state = app.get_state(config)
        print(f"다음 실행할 노드: {state.next[0] if state.next else 'None'}")
        if state.next[0] == "Analysis":
            input_choice = input("현재 그래프를 보고 피드백 여부를 알려주세요(완료/수정/추가):")
            if input_choice in ["수정","추가"]:
                input_feedback = input("피드백 내용을 알려주세요.")
            else:
                input_feedback = ""
        else:
            input_choice = input("현재 보고서를 보고 결정해주세요.(APPROVE/REJECT):")
            if input_choice == "REJECT":
                input_feedback = input("거절 이유를 작성해주세요")
            else:
                input_feedback = ""
        app.update_state(config, {"user_choice": input_choice ,"feedback": input_feedback}, as_node=state.next[0])
        for event in app.stream(None, config=config):
            for key, value in event.items():
                print(f"\n--- Node: {key} ---")
                print(value)
if __name__ == "__main__":
    main()
