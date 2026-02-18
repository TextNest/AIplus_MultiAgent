import graphviz

def get_base_graph():
    """
    Orc_agent의 Main_graph 구조를 정의하는 Graphviz 객체 생성
    - 가로 폭을 줄이고 세로로 길게 배치 (Vertical Layout)
    - 복잡한 클러스터 제거 및 심플한 디자인 적용
    """
    # format='png' -> Streamlit graphviz_chart uses its own rendering but dot source is what matters
    dot = graphviz.Digraph(comment='Orc_agent Flow')
    
    # === 그래프 전역 설정 ===
    dot.attr(rankdir='TB')      # Top to Bottom
    dot.attr(splines='ortho')   # 직각 선 (깔끔하게)
    dot.attr(nodesep='0.3')     # 노드 간격 (좌우)
    dot.attr(ranksep='0.4')     # 계층 간격 (상하) - 줄임
    
    # 노드 스타일
    dot.attr('node', shape='rect', style='filled,rounded', 
             fillcolor='white', fontname="Malgun Gothic", 
             height='0.4', width='1.2', fixedsize='false', fontsize='10')
    
    # === 노드 정의 ===
    dot.node('Start', 'Start', shape='circle', style='filled', fillcolor='#E0E0E0', width='0.5', height='0.5', fontsize='9')
    dot.node('End', 'End', shape='circle', style='filled', fillcolor='#E0E0E0', width='0.5', height='0.5', fontsize='9')
    
    dot.node('File_type', '📂 파일 타입 확인')
    dot.node('File_analysis', '📄 문서 분석')
    
    # Tabular processing nodes
    dot.node('Preprocessing', '🧹 전처리')
    dot.node('Analysis', '🤖 데이터 분석 에이전트')
    dot.node('Final_report', '📝 리포트 생성 에이전트')
    dot.node('Wait', '👤 피드백 대기', shape='diamond', style='filled', fillcolor='#FFE0B2', height='0.6', fontsize='10')

    # === 엣지 정의 (흐름) ===
    
    # 1. Start -> Check
    dot.edge('Start', 'File_type')
    
    # 2. Branching (Tabular vs Document)
    # File_type -> Preprocessing (Main Flow)
    # File_type -> File_analysis (Side Flow)
    dot.edge('File_type', 'Preprocessing', label='tabular')
    dot.edge('File_type', 'File_analysis', label='document')
    
    # 3. Tabular Flow (Vertical)
    dot.edge('Preprocessing', 'Analysis')
    dot.edge('Analysis', 'Final_report')
    dot.edge('Final_report', 'Wait')
    
    # 4. Human Review Loop & End
    dot.edge('Wait', 'End', label='APPROVE')

    dot.edge('Wait', 'Analysis', label='REJECT', color='red', style='dashed', constraint='true', tailport='e', headport='e')
    
    # 5. Document Flow End
    dot.edge('File_analysis', 'End')
    
    # === 레이아웃 조정 (Ranking) ===
    # Preprocessing, Analysis, Final_report, Wait를 수직으로 정렬 (메인 파이프라인)
    # File_analysis는 옆으로 빠지게
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Preprocessing')
        # s.node('File_analysis') # 같은 레벨에 두면 가로로 넓어질 수 있음. 
        # 대신 File_analysis를 Preprocessing과 같은 랭크에 두되, 순서를 제어

    return dot

def generate_highlighted_graph(current_node: str, sub_status: str = None):
    """
    현재 실행 중인 노드를 강조 표시한 Graphviz 객체 반환
    sub_status: Analysis 노드 내부의 진행 상황 (예: Plan, Make...)
    """
    dot = get_base_graph()
    
    # 노드 이름 매핑 (LangGraph 노드 이름 -> Graphviz 노드 이름)
    node_map = {
        "File_type": "File_type",
        "File_analysis": "File_analysis",
        "Preprocessing": "Preprocessing",
        "Analysis": "Analysis",
        "Wait": "Wait",
        "Final_report": "Final_report",
        "END": "End"
    }
    
    target_node = node_map.get(current_node)
    
    if target_node:
        # 강조 스타일 적용 (진한 테두리 + 밝은 노란 배경)
        dot.node(target_node, color='#FF4B4B', penwidth='3.0', fillcolor='#FFF9C4')
        
        # Analysis 노드인 경우 서브 상태 표시 (움직이는 느낌을 주기 위해)
        if current_node == "Analysis" and sub_status:
            # 라벨 변경: "Analysis Agent\n(Running: Plan)"
            new_label = f"🤖 데이터 분석 에이전트\n(진행중: {sub_status})"
            dot.node('Analysis', label=new_label, color='#FF4B4B', penwidth='3.0', fillcolor='#FFF9C4')
    
    return dot
