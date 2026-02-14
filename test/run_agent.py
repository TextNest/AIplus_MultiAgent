
import os
from dotenv import load_dotenv

load_dotenv()

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Orc_agent.core.logger import logger
from src.Orc_agent.Graph.Main_graph import create_main_graph
os.makedirs("img", exist_ok=True)
def main():

    print(">>> 그래프 생성 중...")
    app, sub_apps = create_main_graph()
    analyze_app = sub_apps["analyze"]
    initial_state = {
        "file_path": "fake_marketing_data.csv", 
        "user_query": "데이터의 전반적인 추세를 분석하고 시각화해줘"
    }
    
    logger.info(f">>> 실행 시작: {initial_state['user_query']}")
    config = {"configurable": {"thread_id": "test_thread_5", "user_id": "test_user"}}
    

    def get_sub_config(main_config):
        sub_conf = main_config.copy()
        sub_conf["configurable"] = main_config["configurable"].copy()
        sub_conf["configurable"]["thread_id"] = f"{main_config['configurable']['thread_id']}_sub"
        return sub_conf


    logger.info("\n>>> [메인] 초기 실행...")
    for event in app.stream(initial_state, config=config):
        for key, value in event.items():
            logger.info(f"\n--- Node: {key} ---")
            logger.info(value)
            
    while True:
        sub_config = get_sub_config(config)
        sub_snapshot = analyze_app.get_state(sub_config)
        sub_next = sub_snapshot.next
        main_state = app.get_state(config)
        main_next = main_state.next[0] if main_state.next else None
        
        logger.info(f"\n[상태 확인] 메인: {main_next} | 서브: {sub_next}")
        
        if sub_next:
            logger.info(f">>> [서브 에이전트 중단] {sub_next} 에서 멈춤")
            
            input_choice = input("서브 에이전트 피드백(완료/수정/추가): ")
            if input_choice in ["수정", "추가"]:
                input_feedback = input("피드백 내용: ")
            else:
                input_feedback = ""
            
            logger.info(">>> 서브 에이전트 상태 업데이트 중...")
            analyze_app.update_state(sub_config, {
                "user_choice": input_choice,
                "feed_back": input_feedback
            })
            

            logger.info(">>> 메인 그래프 재개 (분석 노드 재진입)...")
            for event in app.stream(None, config=config):
                for key, value in event.items():
                    logger.info(f"\n--- Node: {key} ---")
                    logger.info(value)
            continue

        elif main_next == "Wait":
            logger.info(f">>> [메인 에이전트 중단] {main_next} 에서 멈춤")
            input_choice = input("보고서 승인(APPROVE/REJECT): ")
            
            if input_choice == "REJECT":
                input_feedback = input("거절 이유: ")
            else:
                input_feedback = ""
            
            app.update_state(config, {
                "human_feedback": input_choice, 
                "feedback": input_feedback
            }, as_node="Wait")
            
            logger.info(">>> 메인 그래프 재개...")
            for event in app.stream(None, config=config):
                for key, value in event.items():
                    logger.info(f"\n--- Node: {key} ---")
                    logger.info(value)
            continue


        elif not main_next and not sub_next:
            logger.info("\n>>> 실행 완료 (All steps done)")
            break
            

        else:
            logger.info(">>> 잔여 단계 실행 중...")
            events_exist = False
            for event in app.stream(None, config=config):
                events_exist = True
                for key, value in event.items():
                    logger.info(f"\n--- Node: {key} ---")
                    logger.info(value)
            
            if not events_exist:
                logger.info(">>> 더 이상 실행할 단계가 없습니다.")
                break
if __name__ == "__main__":
    main()
