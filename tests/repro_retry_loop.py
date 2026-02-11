
import sys
import os
sys.path.append(os.getcwd())

from src.agent.nodes.analyze_data import analyze_data
from src.agent.state import AgentState
from unittest.mock import MagicMock, patch

def test_retry_loop_bug():
    # Initial state
    state = {
        "clean_data": [{"sepal_length": 5.1, "sepal_width": 3.5}],
        "retry_count": 0,
        "evaluation_feedback": "REJECT: Bad code", # Simulator previous rejection
        "analysis_results": []
    }

    # Mock invoke to raise exception
    with patch("src.nodes.analyze_data.LLMFactory") as mock_factory:
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        
        # Configure mocks
        mock_factory.create.return_value = (mock_llm, None)
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_structured_llm.invoke.side_effect = Exception("Simulated LLM API Error")
        
        # Run analyze_data
        print(f"Initial Retry Count: {state['retry_count']}")
        
        # Iteration 1
        result1 = analyze_data(state)
        print(f"Result 1 keys: {result1.keys()}")
        
        if "retry_count" in result1:
            state.update(result1)
            print(f"Updated Retry Count: {state['retry_count']}")
        else:
            print("Retry count NOT updated in state!")
            
        # Iteration 2 (Simulator loop prevention check)
        # If retry_count didn't update, it matches the bug condition.
        
        if "retry_count" not in result1:
            print("\nBUG REPRODUCED: retry_count was not returned on exception.")
            # Verify we can fix it by manually adding it (simulation)
        else:
            print("\nBug not reproduced (or fixed).")

if __name__ == "__main__":
    test_retry_loop_bug()
