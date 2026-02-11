"""
AIplus MultiAgent - 단순화된 Streamlit 앱
데이터 입력 → 결과 표시 흐름

실행: streamlit run webapp/app.py
"""

# === STEP 1: .env 로드 (다른 import보다 먼저!) ===
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(dotenv_path=project_root / ".env")

# === STEP 2: 이제 안전하게 import ===
import streamlit as st
import os
import tempfile
import uuid
import pandas as pd
from typing import Generator, cast
from langchain_core.runnables import RunnableConfig
from src.agent.state import AgentState


# === 페이지 설정 ===
st.set_page_config(
    page_title="AI 데이터 분석",
    page_icon="📊",
    layout="wide"
)


# === LangGraph 워크플로우 ===
@st.cache_resource
def get_graph():
    """LangGraph 그래프 캐싱"""
    from src.graph import create_graph
    return create_graph()


def run_workflow(file_path: str, thread_id: str) -> Generator:
    """워크플로우 스트리밍 실행 (HITL 자동 승인)"""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "file_path": file_path,
        "session_id": thread_id,
        "steps_log": [],
        "analysis_results": [],
        "retry_count": 0
    }
    
    # 워크플로우 실행
    for event in graph.stream(initial_state, config):
        for node_name, node_output in event.items():
            yield node_name, node_output
    
    # human_review에서 중단된 경우 자동 승인
    snapshot = graph.get_state(config)
    while snapshot.next and "human_review" in snapshot.next:
        graph.update_state(config, {"human_feedback": "APPROVE"})
        for event in graph.stream(None, config):
            for node_name, node_output in event.items():
                yield node_name, node_output
        snapshot = graph.get_state(config)


def get_final_report(thread_id: str) -> str:
    """최종 보고서 조회"""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        snapshot = graph.get_state(config)
        return snapshot.values.get("final_report", "보고서를 찾을 수 없습니다.")
    except Exception:
        return "보고서를 찾을 수 없습니다."


# === 세션 상태 초기화 ===
def init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "workflow_completed" not in st.session_state:
        st.session_state.workflow_completed = False
    if "result_approved" not in st.session_state:
        st.session_state.result_approved = False

init_session()


# === UI 렌더링 ===
def main():
    st.title("📊 AI 데이터 분석")
    
    # 재분석 요청 → 바로 분석 실행
    if st.session_state.get("reanalysis_requested"):
        st.session_state.reanalysis_requested = False
        start_analysis()
        return
    
    # 워크플로우 완료됨 → 결과 표시
    if st.session_state.workflow_completed:
        render_result()
        return
    
    # 새 분석 시작 모드
    render_upload()


def render_upload():
    """파일 업로드 UI"""
    
    st.markdown("CSV 파일을 업로드하면 AI가 자동으로 분석합니다.")
    st.divider()

    uploaded_file = st.file_uploader(
        "CSV 파일 업로드",
        type=["csv"],
        help="분석할 CSV 파일을 선택하세요"
    )
    
    if uploaded_file:
        # 임시 파일로 저장
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.uploaded_file_path = temp_path
        
        # 데이터 미리보기
        df = pd.read_csv(temp_path)
        st.markdown("### 데이터 미리보기")
        st.dataframe(df.head(10), width= 'stretch')
        st.caption(f"총 {len(df)} 행, {len(df.columns)} 열")
        
        st.divider()
        
        if st.button("🚀 분석 시작", type="primary"):
            start_analysis()


def start_analysis():
    """분석 워크플로우 시작"""
    
    file_path = st.session_state.uploaded_file_path
    thread_id = st.session_state.thread_id
    
    with st.spinner("🤖 AI 에이전트가 분석 중입니다..."):
        for node_name, node_output in run_workflow(file_path, thread_id):
            pass  # 스트리밍만 소비
    
    st.session_state.workflow_completed = True
    st.rerun()


def render_result():
    """결과 표시 UI"""
    
    thread_id = st.session_state.thread_id
    report = get_final_report(thread_id)
    
    # 승인 여부에 따라 다른 UI 표시
    if st.session_state.result_approved:
        # 승인됨 → 다운로드 가능
        st.success("✅ 결과가 승인되었습니다!")
        st.divider()
        
        st.markdown("### 📄 최종 보고서")
        with st.container(border=True):
            st.markdown(report)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📥 보고서 다운로드",
                data=report,
                file_name="analysis_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            if st.button("🔄 새 분석 시작", use_container_width=True):
                reset_workflow()
    else:
        # 미승인 → 승인/거절 버튼 표시
        st.info("🔍 분석이 완료되었습니다. 결과를 확인하고 승인 또는 거절해주세요.")
        st.divider()
        
        st.markdown("### 📄 분석 결과")
        with st.container(border=True):
            st.markdown(report)
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ 승인", type="primary", use_container_width=True):
                st.session_state.result_approved = True
                st.rerun()
        
        with col2:
            if st.button("❌ 거절 (재분석)", use_container_width=True):
                rerun_analysis()
        
        with col3:
            if st.button("🔄 새 분석 시작", use_container_width=True):
                reset_workflow()


def rerun_analysis():
    """재분석 실행 (거절 시) - 같은 파일로 바로 재분석"""
    st.session_state.thread_id = str(uuid.uuid4())  # 새 thread로 깨끗하게 재실행
    st.session_state.workflow_completed = False
    st.session_state.result_approved = False
    st.session_state.reanalysis_requested = True  # 바로 분석 시작 플래그
    st.rerun()


def reset_workflow():
    """워크플로우 상태 리셋"""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.uploaded_file_path = None
    st.session_state.workflow_completed = False
    st.session_state.result_approved = False
    st.rerun()


if __name__ == "__main__":
    main()
