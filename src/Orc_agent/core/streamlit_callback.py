from typing import Any, Dict, Optional
import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
# graph_visualizer 모듈 임포트 (경로 주의: app.py와 같은 위치라고 가정하거나 path 추가 필요)
try:
    from webapp.graph_visualizer import generate_highlighted_graph
except ImportError:
    # 절대 경로 등으로 import 시도 또는 예외 처리
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
    from webapp.graph_visualizer import generate_highlighted_graph

class StreamlitAgentCallback(BaseCallbackHandler):
    """
    Streamlit UI에 에이전트의 사고 과정과 중간 결과를 실시간으로 렌더링하는 콜백 핸들러
    + 우측 그래프 시각화 업데이트 기능 추가
    """
    def __init__(self, log_container, graph_container=None):
        self.log_container = log_container
        self.graph_container = graph_container
        self.current_step = None

    def __deepcopy__(self, memo):
        # Streamlit DeltaGenerator 객체는 _thread.lock을 포함하여 deepcopy 불가.
        # LangGraph가 서브그래프 config 전달 시 deepcopy를 시도하므로,
        # 동일 인스턴스를 반환하여 모든 그래프/서브그래프가 같은 콜백을 공유하도록 함.
        return self

    def __getstate__(self):
        # pickle 호환: 직렬화 불가능한 Streamlit 컨테이너 제외
        return {"current_step": self.current_step}

    def __setstate__(self, state):
        self.current_step = state.get("current_step")
        self.log_container = None
        self.graph_container = None

    def on_chain_start(self, serialized: Optional[Dict[str, Any]], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        # 체인 이름이 있는 경우 (Graph의 Node 이름 등)
        if serialized:
            chain_name = serialized.get("name", "Agent Step")
        else:
            chain_name = kwargs.get("name", "Agent Step")
        
        # 1. 그래프 업데이트 (Main Agent Node인 경우만)
        if self.graph_container and chain_name in ["File_type", "File_analysis", "Preprocessing", "Analysis", "Final_report", "Wait"]:
            try:
                dot = generate_highlighted_graph(chain_name)
                st.session_state["last_graph_dot"] = dot
                with self.graph_container:
                    st.graphviz_chart(dot, width='stretch')
            except Exception as e:
                pass # 그래프 업데이트 실패해도 로그는 남기도록

        # 2. 로그 메세지 출력
        # Main Agent Nodes
        if chain_name == "File_type":
            self.log_container.info("📂 [Main] 파일 타입을 확인하고 있습니다...")
        elif chain_name == "File_analysis":
            self.log_container.info("📜 [Main] 파일 내용을 분석하고 있습니다...")
        elif chain_name == "Preprocessing":
            self.log_container.info("🧹 [Main] 데이터 전처리를 수행하고 있습니다...")
        elif chain_name == "Analysis":
            self.log_container.success("🤖 [Main] 데이터 분석 서브 에이전트를 호출합니다...")
        elif chain_name == "Final_report":
            self.log_container.success("📝 [Main] 최종 리포트 생성 서브 에이전트를 호출합니다...")
        elif chain_name == "Wait":
             self.log_container.warning("⏳ [Main] 사용자의 최종 검토를 기다리고 있습니다...")

        # Sub Agent Nodes (Analysis)
        elif chain_name in ["Plan", "Make", "Run", "Insight"]:
            if chain_name == "Plan":
                self.log_container.info("  📅 [Sub] 상세 분석 계획을 수립하고 있습니다...")
            elif chain_name == "Make":
                self.log_container.info("  💻 [Sub] 분석 코드를 작성하고 있습니다...")
            elif chain_name == "Run":
                self.log_container.info("  🚀 [Sub] 코드를 실행하고 데이터를 시각화합니다...")
            elif chain_name == "Insight":
                self.log_container.info("  💡 [Sub] 결과를 분석하여 인사이트를 도출합니다...")
                
            # Analysis 노드가 활성화된 상태에서 내부 상태 업데이트
            if self.graph_container:
                try:
                    # Main Node는 여전히 'Analysis'
                    dot = generate_highlighted_graph("Analysis", sub_status=chain_name)
                    st.session_state["last_graph_dot"] = dot
                    with self.graph_container:
                        st.graphviz_chart(dot, width='stretch')
                except Exception:
                    pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        # LLM 응답이 끝났을 때 (디버깅용 로그 또는 결과 표시)
        # 텍스트가 너무 길면 접어서 보여줍니다.
        # pass
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        pass
    
    def render_code(self, code: str):
        with self.log_container.expander("생성된 파이썬 코드", expanded=False):
            st.code(code, language="python")

    def render_image(self, img_path: str):
        import os
        if os.path.exists(img_path):
            self.log_container.image(img_path, caption=os.path.basename(img_path))
        else:
            self.log_container.warning(f"이미지 파일을 찾을 수 없습니다: {img_path}")

    def render_insight(self, insight_text: str):
        self.log_container.success("분석 결과 도출됨")
        # self.log_container.markdown(insight_text) # 로그에는 간단히 표시

