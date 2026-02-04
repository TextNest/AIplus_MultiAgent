"""
Human Review Node
담당: [팀원 F]

역할: 사람의 피드백을 처리하는 HITL 노드
입력: final_report, human_feedback
출력: steps_log (human_feedback은 외부에서 주입)
"""
from ..state import AgentState


def human_review(state: AgentState) -> AgentState:
    """
    Human-in-the-Loop 리뷰를 처리합니다.
    
    이 노드는 interrupt_before로 설정되어 실행 전 일시정지됩니다.
    사용자가 피드백을 제공하면 graph가 재개됩니다.
    
    TODO: 팀원이 구현해야 할 내용
    - 피드백 파싱 및 처리
    - 승인 시 최종 보고서 저장
    - 거절 시 수정 요청 전달
    
    Args:
        state: 현재 AgentState (human_feedback 필수)
    
    Returns:
        AgentState: steps_log가 업데이트된 상태
    """
    human_feedback = state.get("human_feedback")
    final_report = state.get("final_report")
    
    if human_feedback is None:
        return {
            "steps_log": ["[HumanReview] Awaiting human feedback..."]
        }
    
    feedback_upper = human_feedback.upper()
    
    if "APPROVE" in feedback_upper:
        # TODO: 승인 시 보고서 저장 로직
        # 예: 파일로 저장, 데이터베이스에 기록 등
        return {
            "steps_log": [f"[HumanReview] APPROVED - Report finalized"]
        }
    else:
        # 거절 시 피드백 내용 기록
        return {
            "steps_log": [f"[HumanReview] REJECTED - Feedback: {human_feedback}"]
        }
