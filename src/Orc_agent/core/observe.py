"""
Langfuse Observability 중앙 모듈
- observe: @observe 데코레이터 래퍼 (credentials 미설정 시 no-op)
- langfuse_session: 세션/메타데이터 컨텍스트 매니저
- is_langfuse_enabled: credentials 유효성 검증
- SessionAwareCallbackHandler: 모든 LLM provider에서 session_id가 적용되는 CallbackHandler
"""

import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

_observe_fn: Optional[Callable] = None


def is_langfuse_enabled() -> bool:
    """Langfuse 사용 가능 여부 (llm_factory 등에서 공유)"""
    pk = (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    sk = (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    if not pk or not sk or "..." in pk or "..." in sk or len(pk) < 20 or len(sk) < 20:
        return False
    return True


# ---------------------------------------------------------------------------
# @observe 데코레이터 래퍼
# ---------------------------------------------------------------------------

def _get_observe():
    global _observe_fn
    if _observe_fn is not None:
        return _observe_fn

    if not is_langfuse_enabled():
        def _noop(name: Optional[str] = None):
            def decorator(fn):
                return fn
            return decorator
        _observe_fn = _noop
        return _observe_fn

    try:
        from langfuse import observe
        _observe_fn = observe
    except Exception:
        def _noop(name: Optional[str] = None):
            def decorator(fn):
                return fn
            return decorator
        _observe_fn = _noop

    return _observe_fn


def observe(name: Optional[str] = None):
    """Langfuse observe 또는 no-op (credentials 미설정 시)"""
    return _get_observe()(name=name)


# ---------------------------------------------------------------------------
# SessionAwareCallbackHandler (모든 provider에서 session_id 적용)
# ---------------------------------------------------------------------------

def _create_session_aware_handler_class():
    """
    Langfuse CallbackHandler를 상속하여, 직접 llm.invoke() 호출 시에도
    session_id가 trace에 적용되도록 on_chat_model_start/on_llm_start를 보강합니다.

    원인: 기본 CallbackHandler는 on_chain_start에서만 update_trace(session_id=...)를
    호출합니다. ChatOpenAI/ChatAnthropic은 llm.invoke() 시 on_chain_start를 거치지
    않아 session_id가 trace에 적용되지 않습니다.
    """
    from langfuse.langchain import CallbackHandler

    class SessionAwareCallbackHandler(CallbackHandler):
        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
            # 부모 클래스의 on_chat_model_start를 먼저 실행 (trace/generation 생성)
            super().on_chat_model_start(
                serialized, messages,
                run_id=run_id, parent_run_id=parent_run_id,
                tags=tags, metadata=metadata, **kwargs,
            )
            # 직접 LLM 호출(chain 없이)일 때만 session_id를 trace에 적용
            if parent_run_id is None and metadata:
                self._apply_trace_attributes_from_metadata(run_id, metadata)

        def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
            super().on_llm_start(
                serialized, prompts,
                run_id=run_id, parent_run_id=parent_run_id,
                tags=tags, metadata=metadata, **kwargs,
            )
            if parent_run_id is None and metadata:
                self._apply_trace_attributes_from_metadata(run_id, metadata)

        def _apply_trace_attributes_from_metadata(self, run_id, metadata: Dict[str, Any]):
            """metadata에서 langfuse_session_id 등을 추출하여 trace에 적용"""
            attributes = self._parse_langfuse_trace_attributes_from_metadata(metadata)
            if attributes and run_id in self.runs:
                try:
                    self.runs[run_id].update_trace(**attributes)
                except Exception:
                    pass  # Langfuse 연결 실패 등 — 무시하고 계속 진행

    return SessionAwareCallbackHandler


def create_callback_handler():
    """
    Langfuse가 활성화되어 있으면 SessionAwareCallbackHandler를,
    아니면 None을 반환합니다.
    """
    if not is_langfuse_enabled():
        return None
    try:
        HandlerClass = _create_session_aware_handler_class()
        return HandlerClass()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 세션 컨텍스트 매니저 (llm_factory.py에서 분리)
# ---------------------------------------------------------------------------

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

    Langfuse v3에서는 두 가지 방식으로 session_id를 전달합니다:
    1. propagate_attributes: 컨텍스트 기반 전파 (핸들러가 안에서 생성된 경우)
    2. langfuse_metadata: config["metadata"]에 langfuse_session_id 키로 명시적 전달
       → 핸들러가 밖에서 생성되어도 확실히 작동

    사용 예시:
        llm, callbacks = LLMFactory.create('google', 'gemma-3-27b-it')

        with langfuse_session(session_id="session_123") as lf_metadata:
            response = llm.invoke(prompt, config={
                'callbacks': callbacks,
                'metadata': lf_metadata,
            })
    """
    # Langfuse v3 공식 metadata 키 구성
    # ref: https://langfuse.com/docs/integrations/langchain
    langfuse_metadata: Dict[str, object] = {}
    if session_id:
        langfuse_metadata["langfuse_session_id"] = session_id
    if user_id:
        langfuse_metadata["langfuse_user_id"] = user_id
    if tags:
        langfuse_metadata["langfuse_tags"] = tags
    if metadata:
        langfuse_metadata.update(metadata)

    if is_langfuse_enabled():
        from langfuse import propagate_attributes

        # propagate_attributes도 함께 사용 (이중 안전장치)
        with propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
        ):
            yield langfuse_metadata
    else:
        # Langfuse가 설정되지 않은 경우 빈 딕셔너리 반환
        yield langfuse_metadata