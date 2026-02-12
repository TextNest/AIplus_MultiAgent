from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
import pandas as pd

# Global REPL instance to share state (e.g. loaded dataframes)
repl = PythonREPL()

def get_python_repl():
    return repl

@tool
def python_repl_tool(code: str):
    """
    Executes Python code. 
    Use this to manipulate data, generate plots, or perform analysis.
    The environment is persistent, so variables defined in previous steps are available.
    """
    try:
        result = repl.run(code)
        return f"Executed:\n{code}\n\nResult:\n{result}"
    except Exception as e:
        return f"Error executing code:\n{code}\n\nError:\n{e}"
