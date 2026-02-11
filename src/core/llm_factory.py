import os
from contextlib import contextmanager
from typing import Optional, List, Dict

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler
from langfuse import propagate_attributes

# --- Dummy LLM Implementation ---
class DummyLLM:
    """
    A dummy LLM that returns hardcoded responses based on the prompt content.
    """
    def __init__(self):
        pass

    def invoke(self, prompt, config=None):
        prompt_str = str(prompt)
        
        class DummyResponse:
            def __init__(self, code=None, content=None):
                self.code = code
                self.content = content

        # Check for Analysis Prompt (searching for keywords in the prompt)
        if "Analyze the following dataset" in prompt_str or "AnalysisCode" in prompt_str or "python code" in prompt_str.lower():
            return DummyResponse(
                code="print('Dummy Analysis Code Executed')\nimport matplotlib.pyplot as plt\nplt.figure()\nplt.plot([1, 2, 3], [1, 4, 9])\nplt.title('Dummy Plot')\nplt.savefig('dummy_plot.png')\nprint('Figure saved: dummy_plot.png')",
                content="Dummy analysis content"
            )
            
        # Check for Evaluation Prompt
        elif "Evaluate the following analysis" in prompt_str or "Evaluation" in prompt_str:
            return DummyResponse(content="APPROVE")
            
        # Check for Report Prompt
        elif "Write a comprehensive report" in prompt_str or "Report" in prompt_str:
            return DummyResponse(content="# Dummy Report\n\nThis is a dummy report generated in test mode.\n\n## Section 1\nDummy content.\n\n## Section 2\nMore dummy content.")
        
        # Default fallback
        return DummyResponse(content="Dummy response")

    def with_structured_output(self, schema):
        return self

# --- Langfuse Session ---
@contextmanager
def langfuse_session(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    tags: Optional[List[str]] = None,
):
    """
    Langfuse session context manager.
    """
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        with propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
        ):
            yield
    else:
        yield


class LLMFactory:
    @staticmethod
    def create(
        provider: str = 'google',
        model: str = 'gemini-2.5-flash',
        temperature: float = 0,
    ):
        """
        Creates an LLM instance and Langfuse Callbacks.
        """
        from .config import config
        
        if config.DUMMY_MODE:
            return DummyLLM(), None

        # 1. Create Model Object
        if provider == "google":
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=temperature,
                convert_system_message_to_human=True
            )
        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                 # Fallback if key missing, or handle gracefully
                 pass
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
            )
        elif provider == "anthropic":
            llm = ChatAnthropic(
                model=model,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # 2. Create Langfuse Callback
        callbacks = []
        
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            handler = CallbackHandler()
            callbacks.append(handler)
            
        return llm, callbacks