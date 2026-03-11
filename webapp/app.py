"""
AIplus MultiAgent - Streamlit POC
Refactored for 3-Column Layout & HITL Support
"""

import sys
import os
import io
from pathlib import Path
from dotenv import load_dotenv

# === 1. 환경 설정 및 경로 추가 ===
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(dotenv_path=project_root / ".env")

import streamlit as st
import pandas as pd
import uuid
import time
from typing import Any, Generator, Optional, cast
from langchain_core.runnables import RunnableConfig

# === 2. 모듈 임포트 ===
from src.Orc_agent.Graph.Main_graph import create_main_graph
from webapp.graph_visualizer import generate_highlighted_graph
from src.Orc_agent.core.streamlit_callback import StreamlitAgentCallback

# === 3. 페이지 설정 ===
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (선택 사항: 레이아웃 미세 조정)
st.markdown("""
    <style>
    .block-container {
        padding-top: 5rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# === 4. 세션 상태 초기화 ===
def init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "df_preview" not in st.session_state:
        st.session_state.df_preview = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {} # {key: insight_text}
    if "figure_list" not in st.session_state:
        st.session_state.figure_list = []
    if "final_report" not in st.session_state:
        st.session_state.final_report = ""
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    
    # HITL Control States
    if "hitl_active" not in st.session_state:
        st.session_state.hitl_active = False # True if paused
    if "hitl_type" not in st.session_state:
        st.session_state.hitl_type = None # "sub" or "main"
    if "hitl_snapshot" not in st.session_state:
        st.session_state.hitl_snapshot = None
    if "resume_mode" not in st.session_state:
        st.session_state.resume_mode = False
    if "resume_target" not in st.session_state:
        st.session_state.resume_target = None

init_session()

# === 5. 그래프 캐싱 및 로드 ===
@st.cache_resource
def get_graph():
    return create_main_graph()

# === 6. UI 레이아웃 구성 ===
def main():
    # 3단 컬럼 구성 (좌: 1, 중: 2, 우: 1)
    col_left, col_center, col_right = st.columns([1, 2, 1])

    # --- [Left Column] 입력 및 제어 ---
    with col_left:
        st.subheader("🛠️ 설정 및 제어")
        
        # 1. 초기 입력값
        with st.container(border=True):
            st.markdown("**1. 기본 설정**")
            user_query = st.text_input("사용자 질문", value="데이터의 전반적인 추세를 분석하고 시각화해줘")
            
            uploaded_file = st.file_uploader("파일 업로드 (CSV)", type=["csv"])
            if uploaded_file:
                # 파일 저장 및 세션 업데이트
                if st.session_state.uploaded_file_path is None or uploaded_file.name != os.path.basename(st.session_state.uploaded_file_path):
                    file_path = f"temp_{uploaded_file.name}" 
    
     
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.session_state.uploaded_file_path = file_path
                    
                    # 미리보기 데이터 로드
                    df = pd.read_csv(file_path)
                    st.session_state.df_preview = df
            
            if st.button("🚀 분석 시작", type="primary"):
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.is_running = True
                st.session_state.hitl_active = False
                st.session_state.hitl_type = None
                st.session_state.hitl_snapshot = None
                st.session_state.resume_mode = False
                st.session_state.resume_target = None
                st.session_state.analysis_results = {}
                st.session_state.figure_list = []
                st.session_state.final_report = ""
                st.session_state.logs = []
                st.rerun()

        # 2. 서브 에이전트 HITL 피드백 (조건부 표시)
        if st.session_state.hitl_active and st.session_state.hitl_type == "sub":
            with st.container(border=True):
                st.error("🛑 서브 에이전트 피드백 요청")
                st.info("분석 과정에서 사람의 확인이 필요합니다.")
                
                with st.form("sub_hitl_form"):
                    action = st.radio("행동 선택", ["완료 (Approve)", "수정 (Modify)", "추가 (Add)"])
                    feedback_text = st.text_area("피드백 내용", placeholder="수정 또는 추가 시 내용을 입력하세요.")
                    
                    if st.form_submit_button("전송"):
                        handle_sub_feedback(action, feedback_text)

        # 3. 메인 에이전트 피드백 (조건부 표시)
        if st.session_state.hitl_active and st.session_state.hitl_type == "main":
            with st.container(border=True):
                st.warning("🛑 분석 결과 검토 요청")
                st.info("생성된 시각화와 인사이트를 검토한 뒤, 보고서 생성을 결정해주세요.")
                
                with st.form("main_hitl_form"):
                    format_choice = st.multiselect("보고서 파일 형태", ["Markdown", "PDF", "PPTX", "HTML"], default=["Markdown"])
                    style_choice = st.selectbox("보고서 유형 선택", ["AI 자동 판단 (추천)", "일반 리포트", "의사 결정 리포트", "마케팅 예산 분배 리포트"], index=0)
                    action = st.radio("검토 결과", ["승인 (Approve)", "거절 (Reject)"])                
                    feedback_text = st.text_area("피드백 내용", placeholder="거절 시 수정 요청 사항을 입력하세요.")
                    
                    if st.form_submit_button("결정 전송"):
                        handle_main_feedback(action, feedback_text, format_choice, style_choice)

        # 4. 로그 출력 (간단히)
        with st.expander("📝 실행 로그", expanded=True):
            log_container = st.empty()
            # 세션에 저장된 로그가 있다면 다시 표시
            with log_container.container():
                for log in st.session_state.logs:
                    st.text(log)
        
        # 5. [NEW] 보고서 다운로드 (좌측 컬럼)
        if st.session_state.final_report:
            render_download_buttons()

    # --- [Right Column] 그래프 시각화 ---
    with col_right:
        st.subheader("🕸️ 에이전트 상태")
        graph_placeholder = st.empty()
        
        # 항상 마지막 그래프 상태 표시
        if "last_graph_dot" in st.session_state:
             graph_placeholder.graphviz_chart(st.session_state["last_graph_dot"], width='stretch')
        elif not st.session_state.is_running:
            dot = generate_highlighted_graph("Start") 
            graph_placeholder.graphviz_chart(dot, width='stretch')


    # --- [Center Column] 결과 디스플레이 ---
    with col_center:
        st.subheader("📊 분석 결과 및 보고서")

        report_ready = bool(st.session_state.final_report)
        tabs = st.tabs(
            ["📋 데이터 미리보기", "💡 시각화 및 인사이트", "📄 최종 보고서"]
            if report_ready
            else ["📋 데이터 미리보기", "💡 시각화 및 인사이트"]
        )

        with tabs[0]:
            if st.session_state.df_preview is not None:
                st.dataframe(st.session_state.df_preview.head(20), width='stretch')
            else:
                st.info("파일을 업로드하면 데이터가 표시됩니다.")

        with tabs[1]:
            render_visualization_tab()

            if st.session_state.analysis_results and not report_ready:
                st.caption("분석 피드백이 완료되면 최종 보고서 탭이 나타납니다.")

        if report_ready:
            with tabs[2]:
                render_markdown_with_images(st.session_state.final_report)


    # === Auto-Run Logic ===
    if st.session_state.is_running:
        run_engine(log_container, graph_placeholder, user_query)


def render_markdown_with_images(markdown_text):
    """
    Markdown 텍스트 내의 로컬 이미지 경로를 파싱하여 st.image로 렌더링
    Format: ![alt](path)
    """
    import re
    
    # 이미지 패턴 찾기: ![alt](path)
    pattern = r'!\[(.*?)\]\((.*?)\)'
    parts = re.split(pattern, markdown_text)
    
    # parts 구조: [text, alt, path, text, alt, path, ...]
    # len(parts)는 1 (이미지 없음) 또는 1 + 3*N (N개 이미지)
    
    for i in range(0, len(parts), 3):
        text_segment = parts[i]
        if text_segment.strip():
            st.markdown(text_segment)
        
        if i + 2 < len(parts):
            alt_text = parts[i+1]
            img_path = parts[i+2]
            
            # 이미지 경로 정리 (절대 경로 -> 상대 경로 시도 또는 그대로 사용)
            # Streamlit은 st.image에 로컬 절대 경로를 허용함
            if os.path.exists(img_path):
                st.image(img_path, caption=alt_text)
            else:
                st.warning(f"이미지를 찾을 수 없습니다: {img_path}")


# === 7. 실행 엔진 ===
def run_engine(log_container, graph_placeholder, user_query):
    graph, sub_apps = get_graph()
    analyze_app = sub_apps['analyze']
    
    config: RunnableConfig = {
        "configurable": {"thread_id": st.session_state.thread_id, "user_id": "streamlit_user"},
    }
    
    # Callback setup (Graph Placeholder 전달)
    st_callback = StreamlitAgentCallback(log_container, graph_placeholder)
    config["callbacks"] = [st_callback]

    resume_target = st.session_state.get("resume_target")

    # 초기 실행인지, 재개하는 것인지 확인
    if st.session_state.get("resume_mode", False) and resume_target == "sub":
        sub_config: RunnableConfig = {
            "configurable": {
                "thread_id": f"{st.session_state.thread_id}_sub",
                "user_id": "streamlit_user",
            }
        }

        st.session_state.resume_mode = False
        st.session_state.resume_target = None

        try:
            for chunk in analyze_app.stream(None, config=sub_config, stream_mode="values"):
                dot = generate_highlighted_graph("Analysis")
                st.session_state["last_graph_dot"] = dot
                graph_placeholder.graphviz_chart(dot, width='stretch')

                if "result_img_paths" in chunk:
                    st.session_state.figure_list = chunk.get("result_img_paths", [])
                if "final_insight" in chunk:
                    st.session_state.analysis_results = chunk.get("final_insight", {})

            sub_snapshot = analyze_app.get_state(sub_config)
            if sub_snapshot.next:
                st.session_state.is_running = False
                st.session_state.hitl_active = True
                st.session_state.hitl_type = "sub"
                if hasattr(sub_snapshot, "values"):
                    st.session_state.hitl_snapshot = sub_snapshot.values
                st.rerun()
                return

            st.session_state.is_running = False
            st.session_state.hitl_active = True
            st.session_state.hitl_type = "main"
            st.session_state.hitl_snapshot = None
            st.rerun()
            return
        except Exception as e:
            st.error(f"실행 중 오류 발생: {e}")
            st.session_state.is_running = False
            return
    elif st.session_state.get("resume_mode", False):
        # [Bugfix] 재개 모드라면 입력값 없이 실행 (이전 상태 유지)
        input_data = None
        st.session_state.resume_mode = False # 사용 후 초기화
        st.session_state.resume_target = None
    elif not st.session_state.hitl_active:
        # 처음 시작
        initial_state: dict[str, Any] = {
            "file_path": st.session_state.uploaded_file_path,
            "user_query": user_query
        }
        input_data = initial_state
    else:
        input_data = None

    try:
        # 스트리밍 실행
        stream_input = cast(Optional[Any], input_data)
        for event in graph.stream(stream_input, config=config):
            for key, value in event.items():
                # 로그 저장
                msg = f"Completed Node: {key}"

                if key in {"File_type", "File_analysis", "Preprocessing", "Analysis", "Wait", "Final_report"}:
                    dot = generate_highlighted_graph(key)
                    st.session_state["last_graph_dot"] = dot
                    graph_placeholder.graphviz_chart(dot, width='stretch')
                
                # 분석 결과 저장 (실시간 업데이트)
                if key == "Analysis" and "analysis_results" in value:
                    st.session_state.analysis_results = value["analysis_results"]
                    st.session_state.figure_list = value["figure_list"]
                
                if key == "Final_report" and "final_report" in value:
                    st.session_state.final_report = value["final_report"]

        # Sub Agent의 상태 확인 (Analysis 노드 내부 Interrupt)
        sub_config = cast(RunnableConfig, cast(object, {**config}))
        sub_config["configurable"] = {
            **cast(dict[str, Any], config.get("configurable", {})),
            "thread_id": f"{st.session_state.thread_id}_sub",
        }
        sub_snapshot = analyze_app.get_state(sub_config)
        
        if sub_snapshot.next:
             st.session_state.is_running = False
             st.session_state.hitl_active = True
             st.session_state.hitl_type = "sub"
             
             # *** 중요 *** : 중간 결과(이미지 등)를 저장하여 피드백 시 보여줌
             if hasattr(sub_snapshot, "values"):
                 st.session_state.hitl_snapshot = sub_snapshot.values
                 
             st.rerun()
             return

        # 스트림 루프 종료 후 상태 체크 (Interrupt 확인)
        snapshot = graph.get_state(config)

        if snapshot.next:
             if snapshot.next[0] == "Wait":
                 st.session_state.is_running = False
                 st.session_state.hitl_active = True
                 st.session_state.hitl_type = "main"
                 st.rerun()
                 return

        # 완전히 끝남
        st.session_state.is_running = False
        st.session_state.hitl_active = False
        st.balloons()
        
        # [NEW] 다운로드 버튼 표시 (Rerun to show buttons)
        st.rerun()
        
    except Exception as e:
        error_msg = str(e)
        if "서브그래프" in error_msg and "멈췄습니다" in error_msg:
             st.session_state.is_running = False
             st.session_state.hitl_active = True
             st.session_state.hitl_type = "sub"
             # 에러 발생 시에도 sub snapshot 확인 시도
             try:
                sub_config = cast(RunnableConfig, cast(object, {**config}))
                sub_config["configurable"] = {
                    **cast(dict[str, Any], config.get("configurable", {})),
                    "thread_id": f"{st.session_state.thread_id}_sub",
                }
                sub_snapshot = analyze_app.get_state(sub_config)
                if hasattr(sub_snapshot, "values"):
                    st.session_state.hitl_snapshot = sub_snapshot.values
             except:
                 pass
             st.rerun()
        elif "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
            st.error(
                "LLM 호출이 quota 한도에 걸렸습니다. 현재 설정된 Google Gemini 키로는 분석을 계속할 수 없습니다. "
                "유효한 다른 provider 키를 넣거나, Gemini quota가 남은 키로 교체한 뒤 다시 시도해주세요."
            )
            st.session_state.is_running = False
        elif "invalid_api_key" in error_msg.lower() or "incorrect api key" in error_msg.lower() or "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            st.error(
                "LLM API 키 인증에 실패했습니다. 현재 설정된 provider 키가 더미이거나 유효하지 않습니다. "
                "실사용 가능한 API 키를 `.env`에 설정한 뒤 다시 시도해주세요."
            )
            st.session_state.is_running = False
        else:
            st.error(f"실행 중 오류 발생: {e}")
            st.session_state.is_running = False

# === 8. 서브함수 (피드백 처리) ===
def handle_sub_feedback(action, text):
    _, sub_apps = get_graph()
    analyze_app = sub_apps['analyze']
    
    sub_config: RunnableConfig = {
        "configurable": {"thread_id": f"{st.session_state.thread_id}_sub"}
    }
    
    mapping = {"완료 (Approve)": "완료", "수정 (Modify)": "수정", "추가 (Add)": "추가"}
    val = mapping[action]
    
    # 상태 업데이트
    analyze_app.update_state(sub_config, {
        "user_choice": val,
        "feed_back": [text]
    }, as_node="Wait")
    
    st.session_state.hitl_active = False
    st.session_state.is_running = True # 다시 실행
    st.session_state.hitl_snapshot = None # 초기화
    st.session_state.resume_mode = True # [Bugfix] 재개 모드 활성화
    st.session_state.resume_target = "sub"
    st.rerun()

def handle_main_feedback(action, text, format_choice, style_choice):
    graph, sub_apps = get_graph()
    
    config: RunnableConfig = {
        "configurable": {"thread_id": st.session_state.thread_id}
    }
    
    val = "APPROVE" if "Approve" in action else "REJECT"

    if val == "APPROVE":
        snapshot = graph.get_state(config)
        values = snapshot.values if hasattr(snapshot, "values") else {}
        report_app = sub_apps["report"]

        sub_input = {
            "analysis_results": values.get("analysis_results") or st.session_state.analysis_results,
            "figure_list": values.get("figure_list") or st.session_state.figure_list,
            "file_path": values.get("file_path", st.session_state.uploaded_file_path or ""),
            "report_format": format_choice or ["Markdown"],
            "report_style": style_choice,
            "clean_data": values.get("clean_data"),
        }

        result = report_app.invoke(sub_input, config=config)
        final_report = result.get("final_report", "")

        st.session_state.final_report = final_report
        st.session_state.hitl_active = False
        st.session_state.hitl_type = None
        st.session_state.is_running = False
        st.session_state.resume_mode = False
        st.session_state.resume_target = None
        st.balloons()
        st.rerun()
        return

    graph.update_state(config, {
        "human_feedback": val,
        "feedback": text
    }, as_node="Wait")

    st.session_state.hitl_active = False
    st.session_state.is_running = True
    st.session_state.resume_mode = True # [Bugfix] 재개 모드 활성화
    st.session_state.resume_target = "main"
    st.rerun()

# === 9. 시각화 및 인사이트 렌더링 (Pagination & Pairing) ===
def render_visualization_tab():
    # 1. 데이터 소스 결정
    # 기본은 Session State의 결과
    results = st.session_state.analysis_results or {} # dict: {'overall':..., 'img.png':...}
    figures = st.session_state.figure_list or []      # list: ['path/to/img.png', ...]
    
    # HITL 중이라면 Snapshot 데이터가 우선할 수 있음 (또는 최신 상태 반영)
    if st.session_state.hitl_active and st.session_state.hitl_type == "sub" and st.session_state.hitl_snapshot:
        snapshot = st.session_state.hitl_snapshot
        # 스냅샷에 있는 데이터로 덮어쓰기 (있다면)
        if snapshot.get("result_img_paths"):
            figures = snapshot.get("result_img_paths")
        if snapshot.get("final_insight"):
            results = snapshot.get("final_insight")
    
    # 데이터가 없으면 안내
    if not results and not figures:
        st.info("시각화 또는 분석 결과가 없습니다.")
        return

    # 2. 아이템 구성 (Pairing Logic)
    # 목표: [ {title, image_path, insight_text}, ... ]
    imgs = [] 
    texts = []
    items = []
    
    # (1) Overall Insight(s)
    for key, value in results.items():
        if not isinstance(key, str) or not key.startswith("overall"):
            continue
        overall_text = value.get("insight", "") if isinstance(value, dict) else str(value)
        if overall_text:
            items.append({
                "type": "overall",
                "title": f"📊 종합 인사이트: {key}",
                "text": overall_text
            })
            
    # (2) Image + Insight Pairs
    # figures 리스트를 순회하며 매칭되는 insight를 찾음
    # final_insight 키는 보통 파일명(basename)임
    
    # 매칭된 인사이트 키를 추적하여 나중에 남은 것 처리
    matched_keys = {key for key in results.keys() if isinstance(key, str) and key.startswith("overall")}
    
    for fig_path in figures:
        if not os.path.exists(fig_path):
            continue
            
        file_name = os.path.basename(fig_path)
        insight_data = results.get(file_name, {})
        insight_text = insight_data.get("insight", "") if isinstance(insight_data, dict) else ""
        if file_name in results:
            matched_keys.add(file_name)
        
        # if file_name in results:
        #     matched_keys.add(file_name)
            
        imgs.append(fig_path)
        
    # (3) Orphan Insights (이미지 없이 텍스트만 있는 경우)
    for key, val in results.items():
        if key in matched_keys:
            continue
        txt = val.get("insight", "") if isinstance(val, dict) else str(val)
        if txt:
            items.append({
                "type": "text",
                "title": f"📝 추가 인사이트: {key}",
                "text": txt
            })

    # zip으로 묶어서 추가 (이미지와 텍스트 순서가 보장되어야 함 - 현재 로직은 단순 순서 매칭이라 위험할 수 있음)
    # 하지만 사용자 의도대로 리스트를 합침
    # text 리스트가 img 리스트와 길이가 다를 수 있으므로 주의
    
    # 텍스트 리스트 구성 (이미지 순서에 맞춰야 함)
    text = []
    for fig_path in figures:
        if not os.path.exists(fig_path): continue
        file_name = os.path.basename(fig_path)
        insight_data = results.get(file_name, {})
        insight_text = insight_data.get("insight", "") if isinstance(insight_data, dict) else ""
        text.append(insight_text)

    for i,t in zip(imgs,text):
        items.append({
            "type": "chart",
            "title": f"📈 분석 차트: {os.path.basename(i)}",
            "image": i,
            "text": t
        })
    
    if not items:
         st.warning("표시할 항목이 없습니다.")
         return

    # 3. Pagination 구현
    if "viz_page" not in st.session_state:
        st.session_state.viz_page = 0
    
    # 페이지 범위 안전 장치
    total_pages = len(items)
    if st.session_state.viz_page >= total_pages:
        st.session_state.viz_page = total_pages - 1
    if st.session_state.viz_page < 0:
        st.session_state.viz_page = 0

    # 네비게이션 버튼 (상단)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("⬅️ 이전", key="viz_prev", disabled=st.session_state.viz_page <= 0):
            st.session_state.viz_page -= 1
            st.rerun()
    with c3:
        if st.button("다음 ➡️", key="viz_next", disabled=st.session_state.viz_page >= total_pages - 1):
            st.session_state.viz_page += 1
            st.rerun()
            
    # 현재 페이지 렌더링
    current_item = items[st.session_state.viz_page]
    
    with st.container(border=True):
        st.markdown(f"### {current_item['title']}")
        
        if current_item.get("type") == "chart":
            st.image(current_item["image"], width='stretch')
            if current_item["text"]:
                st.info(current_item["text"])
            else:
                st.caption("해당 차트에 대한 상세 인사이트가 아직 생성되지 않았습니다.")
                
        elif current_item.get("type") == "overall":
            st.success(current_item["text"])
            
        else:
            st.info(current_item["text"])
        
    st.caption(f"Page {st.session_state.viz_page + 1} / {total_pages}")


def render_download_buttons():
    """
    생성된 보고서 파일(PDF, HTML, PPTX, Markdown) 다운로드 버튼 렌더링
    """
    st.divider()
    st.subheader("📥 보고서 다운로드")
    
    # 1. 파일 경로 설정 (output 디렉토리 기준)
    output_dir = "output"
    files = {
        "PDF 보고서": "report.pdf",
        "HTML 보고서": "report.html",
        "PPTX 보고서": "report.pptx"
    }
    
    # 좌측 컬럼용 수직 레이아웃
    
    # (1) Markdown 다운로드 (항상 가능)
    if st.session_state.final_report:
        st.download_button(
            label="📄 Markdown 다운로드",
            data=st.session_state.final_report,
            file_name="report.md",
            mime="text/markdown",
            use_container_width=True
        )
        
    # (2) 생성된 파일 다운로드
    for label, filename in files.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                file_data = f.read()
                
            st.download_button(
                label=f"📑 {label}",
                data=file_data,
                file_name=filename,
                mime="application/octet-stream",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
