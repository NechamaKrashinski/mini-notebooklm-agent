from langgraph.graph import StateGraph, START, END
from agents.ted.nodes import after_approval, after_critique, approval, count_words, critique_script, plan_talk, gather_context, revise, write_talk
from agents.ted.state import TedState



def build_ted_graph(checkpointer=None):
    graph = StateGraph(TedState)
    graph.add_node(plan_talk)
    graph.add_node(gather_context)
    graph.add_node(write_talk)
    graph.add_node(count_words)
    graph.add_node(critique_script)
    graph.add_node(revise)
    graph.add_node(approval)
    graph.add_edge(START,"plan_talk")
    graph.add_edge("plan_talk","gather_context")
    graph.add_edge("gather_context","write_talk")
    graph.add_edge("write_talk","count_words")
    graph.add_edge("count_words","critique_script")
    graph.add_edge("revise", "count_words")
    graph.add_conditional_edges("critique_script", after_critique, {"approval": "approval", "revise": "revise"})
    graph.add_conditional_edges("approval", after_approval, {"revise": "revise", "done": END})
    return graph.compile(checkpointer=checkpointer)


