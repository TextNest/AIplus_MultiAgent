import os
from contextlib import contextmanager
from typing import Optional, List, Dict

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler
from langfuse import propagate_attributes


@contextmanager
def langfuse_session(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    tags: Optional[List[str]] = None,
):
    """
    Langfuse session 컨텍스트 매니저.
    이 컨텍스트 안에서 실행되는 모든 LLM 호출에 session_id, user_id 등이 기록됩니다.
    
    사용 예시:
        llm, callbacks = LLMFactory.create('google', 'gemma-3-27b-it')
        
        with langfuse_session(session_id="session_123", user_id="user_456"):
            response = llm.invoke(prompt, config={'callbacks': callbacks})
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
        # Langfuse가 설정되지 않은 경우 그냥 통과
        yield


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
            from src.core.llm_factory import LLMFactory, langfuse_session
            
            llm, callbacks = LLMFactory.create('google', 'gemma-3-27b-it')
            
            # session_id를 기록하려면 langfuse_session 컨텍스트 사용
            with langfuse_session(session_id="my-session-id"):
                response = llm.invoke(prompt, config={'callbacks': callbacks})
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

        # 2. Langfuse Callback 생성
        callbacks = []
        
        # Langfuse 3.x: 환경변수를 자동으로 읽음
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            handler = CallbackHandler()
            callbacks.append(handler)
            
        return llm, callbacks