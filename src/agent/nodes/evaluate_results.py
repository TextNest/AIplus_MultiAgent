from ..state import AgentState
from src.core.llm_factory import LLMFactory
from ..prompt_engineering.prompts import EVALUATION_PROMPT

def evaluate_results(state: AgentState) -> AgentState:
    """
    Evaluates analysis results using LLM as a judge.
    """
    analysis_results = state.get("analysis_results", [])
    if not analysis_results:
        return {"evaluation_feedback": "REJECT: No analysis results found."}
        
    last_result = analysis_results[-1]
    
    # LLM Setup
    llm, callbacks = LLMFactory.create('google', 'gemini-2.5-flash')
    
    prompt = EVALUATION_PROMPT.format(
        last_result=last_result
    )
    
    try:
        response = llm.invoke(prompt)
        feedback = response.content.strip()
        return {
            "evaluation_feedback": feedback,
            "steps_log": [f"[Evaluate] Feedback: {feedback}"]
        }
    except Exception as e:
        return {
            "evaluation_feedback": f"REJECT: Error during evaluation - {str(e)}",
            "steps_log": [f"[Evaluate] Error: {str(e)}"]
        }
