"""
Langfuse observe 래퍼 - credentials 미설정/유효하지 않을 때 no-op 사용
401 등 Langfuse 연결 오류를 방지
"""

import os
from typing import Callable, Optional

_observe_fn: Optional[Callable] = None


def is_langfuse_enabled() -> bool:
    """Langfuse 사용 가능 여부 (llm_factory 등에서 공유)"""
    pk = (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    sk = (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    if not pk or not sk or "..." in pk or "..." in sk or len(pk) < 20 or len(sk) < 20:
        return False
    return True


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
    return _get_observe()(name)