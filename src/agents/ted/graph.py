from langgraph.graph import StateGraph, START, END
from agents.ted.nodes import after_approval, after_critique, approval, count_words, critique_script, plan_talk, gather_context, revise, write_talk, synthesize_audio 
from agents.ted.state import TedState
from langgraph.types import RetryPolicy
from pathlib import Path



def build_ted_graph(checkpointer=None):
    graph = StateGraph(TedState)
    graph.add_node(plan_talk)
    graph.add_node(gather_context)
    graph.add_node(write_talk)
    graph.add_node(count_words)
    graph.add_node(critique_script)
    graph.add_node(revise)
    graph.add_node(approval)
    graph.add_node(synthesize_audio,retry_policy=RetryPolicy(max_attempts=3))
    graph.add_edge(START,"plan_talk")
    graph.add_edge("plan_talk","gather_context")
    graph.add_edge("gather_context","write_talk")
    graph.add_edge("write_talk","count_words")
    graph.add_edge("count_words","critique_script")
    graph.add_edge("revise", "count_words")
    graph.add_conditional_edges("critique_script", after_critique, {"approval": "approval", "revise": "revise"})
    graph.add_conditional_edges("approval", after_approval, {"revise": "revise", "synthesize_audio": "synthesize_audio"})
    graph.add_edge("synthesize_audio", END)
    return graph.compile(checkpointer=checkpointer)

def build_ted_graph_for_server():
    """Zero-arg factory for langgraph dev/Studio, which injects values into any
    recognized parameter (e.g. checkpointer) and provides its own persistence."""
    # Studio runs in its own process: the app's in-memory sources aren't here,
    # and the Netfree SSL workaround from serve.py hasn't run.
    from netfree_unstrict_ssl import unstrict_ssl
    unstrict_ssl()

    from core.store import store
    if not store.list():
        for path in Path("data/studio_sources").glob("*.*"):
            store.add(name=path.name, content=path.read_text(encoding="utf-8"))
    return build_ted_graph()

