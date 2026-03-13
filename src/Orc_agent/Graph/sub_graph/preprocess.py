from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from ...State.state import preprocessState
from ...Node.sub_node import preprocess


def preprocess_graph(CheckPoint=None):
    """Build the 15-node preprocessing pipeline graph."""
    preprocess_workflow = StateGraph(preprocessState)

    # --- Nodes ---
    preprocess_workflow.add_node("intake",                         preprocess.intake_node)
    preprocess_workflow.add_node("raw_data_preprocessing",         preprocess.raw_data_preprocessing_node)
    preprocess_workflow.add_node("categorical_standardization",    preprocess.categorical_standardization_node)
    preprocess_workflow.add_node("duplicate_cleanup",              preprocess.duplicate_cleanup_node)
    preprocess_workflow.add_node("date_integrity",                 preprocess.date_integrity_node)
    preprocess_workflow.add_node("data_state_awareness",           preprocess.data_state_awareness_node)
    preprocess_workflow.add_node("measurement_reconstruction",     preprocess.measurement_reconstruction_node)
    preprocess_workflow.add_node("metric_derivation",              preprocess.metric_derivation_node)
    preprocess_workflow.add_node("reliability_signals",            preprocess.reliability_signals_node)
    preprocess_workflow.add_node("semantic_cleanup",               preprocess.semantic_cleanup_node)
    preprocess_workflow.add_node("funnel_leakage",                 preprocess.funnel_leakage_node)
    preprocess_workflow.add_node("context_enrichment",             preprocess.context_enrichment_node)
    preprocess_workflow.add_node("final_assembly",                 preprocess.final_assembly_node)
    preprocess_workflow.add_node("output_formatting",              preprocess.output_formatting_node)
    preprocess_workflow.add_node("quality_gate",                   preprocess.quality_gate_node)

    # --- Edges ---
    preprocess_workflow.add_edge(START,                            "intake")
    preprocess_workflow.add_edge("intake",                         "raw_data_preprocessing")
    preprocess_workflow.add_edge("raw_data_preprocessing",         "categorical_standardization")
    preprocess_workflow.add_edge("categorical_standardization",    "duplicate_cleanup")
    preprocess_workflow.add_edge("duplicate_cleanup",              "date_integrity")
    preprocess_workflow.add_edge("date_integrity",                 "data_state_awareness")
    preprocess_workflow.add_edge("data_state_awareness",           "measurement_reconstruction")
    preprocess_workflow.add_edge("measurement_reconstruction",     "metric_derivation")
    preprocess_workflow.add_edge("metric_derivation",              "reliability_signals")
    preprocess_workflow.add_edge("reliability_signals",            "semantic_cleanup")
    preprocess_workflow.add_edge("semantic_cleanup",               "funnel_leakage")
    preprocess_workflow.add_edge("funnel_leakage",                 "context_enrichment")
    preprocess_workflow.add_edge("context_enrichment",             "final_assembly")
    preprocess_workflow.add_edge("final_assembly",                 "output_formatting")
    preprocess_workflow.add_edge("output_formatting",              "quality_gate")

    # --- Quality gate loop ---
    preprocess_workflow.add_conditional_edges(
        "quality_gate",
        preprocess.route_after_quality_gate,
        {"data_state_awareness": "data_state_awareness", END: END},
    )

    preprocess_app = preprocess_workflow.compile(checkpointer=CheckPoint)
    return preprocess_app
