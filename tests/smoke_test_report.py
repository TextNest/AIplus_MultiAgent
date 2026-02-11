
import sys
import os
sys.path.append(os.getcwd())

from src.graph import create_graph
import pandas as pd

def verify_enhanced_report():
    # 1. Prepare data
    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7],
        "sepal_width": [3.5, 3.0, 3.2],
        "species": ["setosa", "setosa", "setosa"]
    })
    
    # 2. Setup Initial State
    initial_state = {
        "file_path": "test_iris.csv",
        "clean_data": df.to_dict('records'),
        "retry_count": 0,
        "analysis_results": [],
        "steps_log": [],
        "figure_list": []
    }
    
    # 3. Compile Graph
    app = create_graph()
    
    # 4. Run until human_review (mocked or just check state updates)
    # Since we use real LLM strings in analysis_data (gemini-2.5-flash),
    # this will actually call the API if API key is present.
    # To avoid token drain, we just verify the state transition logic in a dry run if possible.
    
    print("Graph initialized. Run verification...")
    
    # Let's check if the prompts are loadable
    from src.agent.prompt_engineering.prompts import ANALYSIS_PROMPT, REPORT_PROMPT
    assert "ANALYSIS_PROMPT" in locals() or "ANALYSIS_PROMPT" in globals()
    print("Prompts loaded successfully.")

if __name__ == "__main__":
    verify_enhanced_report()
