import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from .observe import create_callback_handler, is_langfuse_enabled


class LLMFactory:
    @staticmethod
    def create(
        provider: str,
        model: str,
        temperature: float = 0,
    ):
        """
        LLM 객체와 Langfuse Callbacks를 세트로 반환합니다.
        
        Args:
            provider: 'google', 'openai', 'anthropic' 중 하나
            model: 모델 이름 (예: 'gemma-3-27b-it', 'gpt-4o', 'claude-3-5-sonnet')
            temperature: 생성 온도 (기본값: 0)
        
        Returns:
            tuple: (llm, callbacks)
        
        사용 예시:
            from src.core.llm_factory import LLMFactory
            from src.core.observe import langfuse_session
            
            llm, callbacks = LLMFactory.create('google', 'gemma-3-27b-it')
            
            # session_id를 기록하려면 langfuse_session 컨텍스트 사용
            with langfuse_session(session_id="my-session-id") as lf_metadata:
                response = llm.invoke(prompt, config={
                    'callbacks': callbacks,
                    'metadata': lf_metadata,
                })
        """
        # 1. 모델 객체 생성
        if provider == "google":
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=os.environ.get("GOOGLE_API_KEY"),
                temperature=temperature,
            )
        elif provider == "openai":
            llm = ChatOpenAI(
                model=model,
                api_key=os.environ.get("OPENAI_API_KEY"),
                temperature=temperature,
            )
        elif provider == "anthropic":
            llm = ChatAnthropic(
                model=model,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # 2. Langfuse Callback 생성 (SessionAwareCallbackHandler 사용)
        callbacks = []
        
        handler = create_callback_handler()
        if handler is not None:
            callbacks.append(handler)
            
        return llm, callbacks