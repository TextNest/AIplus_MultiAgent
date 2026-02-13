"""
Langfuse 세션 추적 테스트용 미니 그래프.
2개 노드가 각각 LLM 호출 → 동일 session_id로 Langfuse에 그룹핑되는지 확인.

실행: python test_session.py
확인: Langfuse 대시보드 → Sessions 탭에서 session_id 검색
"""

import os
import uuid
from typing import TypedDict, Optional, List, Annotated

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from src.core.llm_factory import LLMFactory
from src.core.observe import langfuse_session, is_langfuse_enabled


# --- State ---
def merge_lists(left: list, right: list) -> list:
    return (left or []) + (right or [])


class TestState(TypedDict):
    session_id: str
    node1_result: Optional[str]
    node2_result: Optional[str]
    logs: Annotated[List[str], merge_lists]


# --- Node 1: 간단한 LLM 호출 ---
def node_greeting(state: TestState, config: RunnableConfig) -> dict:
    session_id = config.get("configurable", {}).get("session_id", "unknown")

    llm, callbacks = LLMFactory.create(
        provider="google",
        model="gemma-3-27b-it",
        temperature=0,
    )

    with langfuse_session(
        session_id=session_id,
        user_id="test_user",
        tags=["test", "node_greeting"],
    ) as lf_metadata:
        response = llm.invoke(
            "Say 'Hello from Node 1!' in one short sentence.",
            config={"callbacks": callbacks, "metadata": lf_metadata},
        )

    content = response.content
    if isinstance(content, list):
        content = "".join([str(c) for c in content])

    print(f"  [Node1] session_id={session_id} | response={content[:80]}")
    return {
        "node1_result": content,
        "logs": [f"[Node1] OK (session={session_id})"],
    }


# --- Node 2: 또 다른 LLM 호출 (같은 session_id) ---
def node_summary(state: TestState, config: RunnableConfig) -> dict:
    session_id = config.get("configurable", {}).get("session_id", "unknown")
    prev = state.get("node1_result", "N/A")

    llm, callbacks = LLMFactory.create(
        provider="google",
        model="gemma-3-27b-it",
        temperature=0,
    )

    prompt = f"Summarize this in 10 words or less: '{prev}'"

    with langfuse_session(
        session_id=session_id,
        user_id="test_user",
        tags=["test", "node_summary"],
    ) as lf_metadata:
        response = llm.invoke(
            prompt,
            config={"callbacks": callbacks, "metadata": lf_metadata},
        )

    content = response.content
    if isinstance(content, list):
        content = "".join([str(c) for c in content])

    print(f"  [Node2] session_id={session_id} | response={content[:80]}")
    return {
        "node2_result": content,
        "logs": [f"[Node2] OK (session={session_id})"],
    }


# --- Graph ---
def create_test_graph():
    wf = StateGraph(TestState)
    wf.add_node("greeting", node_greeting)
    wf.add_node("summary", node_summary)
    wf.set_entry_point("greeting")
    wf.add_edge("greeting", "summary")
    wf.add_edge("summary", END)
    return wf.compile()


# --- Main ---
if __name__ == "__main__":
    print("=" * 60)
    print("Langfuse Session Tracking Test")
    print("=" * 60)

    # Langfuse 상태 확인
    enabled = is_langfuse_enabled()
    print(f"Langfuse enabled: {enabled}")
    if not enabled:
        print("WARNING: Langfuse credentials not set. Traces won't be recorded.")
        print("  Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL in .env")

    # 고유 session_id 생성
    test_session_id = f"test-session-{uuid.uuid4().hex[:8]}"
    print(f"Session ID: {test_session_id}")
    print()

    # 그래프 생성 & 실행
    app = create_test_graph()

    initial_state: TestState = {
        "session_id": test_session_id,
        "node1_result": None,
        "node2_result": None,
        "logs": [],
    }

    thread = {
        "configurable": {
            "thread_id": test_session_id,
            "session_id": test_session_id,
            "user_id": "test_user",
        }
    }

    print("Running graph...")
    result = app.invoke(initial_state, thread)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Node1: {result.get('node1_result', 'N/A')[:100]}")
    print(f"Node2: {result.get('node2_result', 'N/A')[:100]}")
    print(f"Logs:  {result.get('logs', [])}")
    print()

    if enabled:
        base_url = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        print(f"Check Langfuse dashboard:")
        print(f"  {base_url}")
        print(f"  Sessions tab -> search: {test_session_id}")
        print(f"  Expected: 2 traces grouped under this session")
    else:
        print("Langfuse disabled - no traces to check.")

    print()
    print("Done.")
