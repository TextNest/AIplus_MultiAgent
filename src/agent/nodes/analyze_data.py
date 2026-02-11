import pandas as pd
import re
from langchain_experimental.utilities import PythonREPL
from ..state import AgentState
from src.core.llm_factory import LLMFactory
# from ..utils.df_summary import get_df_summary
from ..prompt_engineering.prompts import ANALYSIS_PROMPT
from pydantic import BaseModel, Field

class AnalysisCode(BaseModel):
    code: str = Field(description="Executable Python code for analysis. Must use pandas, matplotlib, seaborn.")

def analyze_data(state: AgentState) -> AgentState:
    """
    Generates and executes analysis code using LLM.
    """
    clean_data = state.get("clean_data")
    if clean_data is None:
        return {"steps_log": ["[Analyze] ERROR: No clean data available"]}

    df = pd.DataFrame(clean_data)
    # df_summary = get_df_summary(df)
    
    retry_count = state.get("retry_count", 0)
    feedback = state.get("evaluation_feedback", "")
    
    # 1. Plan & Generate Code
    try:
        # Move LLM initialization inside try block to catch init errors
        llm, callbacks = LLMFactory.create('google', 'gemini-2.5-flash') 
        
        structured_llm = llm.with_structured_output(AnalysisCode)
        
        prompt = ANALYSIS_PROMPT.format(
            df_summary=df_summary,
            feedback=feedback
        )
        
        response = structured_llm.invoke(prompt)
        code = response.code
        
        # 2. Execute Code
        repl = PythonREPL()
        
        temp_csv = "analysis_input.csv"
        df.to_csv(temp_csv, index=False)
        
        setup_code = f"import pandas as pd\ndf = pd.read_csv('{temp_csv}')\n"
        full_code = setup_code + code
        
        result = repl.run(full_code)
        
        # Capture figure filenames from the code
        # Pattern to find strings ending in .png
        figures = re.findall(r"['\"]([^'\"]+\.png)['\"]", code)
        
        return {
            "analysis_results": [f"Code:\n```python\n{code}\n```\n\nResult:\n{result}"],
            "retry_count": retry_count + 1 if feedback else 0,
            "figure_list": figures,
            "steps_log": ["[Analyze] Generated and executed code"]
        }
        
    except Exception as e:
        return {
            "analysis_results": [f"Error: {str(e)}"],
            "steps_log": [f"[Analyze] Error: {str(e)}"],
            "retry_count": retry_count + 1 # Critical Fix: Increment retry count on error
        }
