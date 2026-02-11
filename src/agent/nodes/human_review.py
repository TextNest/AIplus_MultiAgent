"""
Human Review Node
Role: Handle HITL feedback.
"""
from ..state import AgentState

def human_review(state: AgentState) -> AgentState:
    """
    Handles Human-in-the-Loop review.
    Interrupts before this node.
    """
    human_feedback = state.get("human_feedback")
    
    if human_feedback is None:
        return {
            "steps_log": ["[HumanReview] Awaiting human feedback..."]
        }
    
    feedback_upper = human_feedback.upper()
    
    if "APPROVE" in feedback_upper:
        return {
            "steps_log": [f"[HumanReview] APPROVED - Report finalized"]
        }
    else:
        return {
            "steps_log": [f"[HumanReview] REJECTED - Feedback: {human_feedback}"]
        }
