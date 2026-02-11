"""
Langfuse Observability 중앙 모듈
- observe: @observe 데코레이터 래퍼 (credentials 미설정 시 no-op)
- langfuse_session: 세션/메타데이터 컨텍스트 매니저
- is_langfuse_enabled: credentials 유효성 검증
"""

import os
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

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

    사용 예시:
        from src.core.observe import langfuse_session

        llm, callbacks = LLMFactory.create('google', 'gemma-3-27b-it')

        with langfuse_session(session_id="session_123", user_id="user_456"):
            response = llm.invoke(prompt, config={'callbacks': callbacks})
    """
    if is_langfuse_enabled():
        from langfuse import propagate_attributes

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