from langgraph.graph import StateGraph, START, END
from agents.ted.nodes import plan_talk, gather_context, write_talk
from agents.ted.state import TedState



def build_ted_graph():
    graph = StateGraph(TedState)
    graph.add_node(plan_talk)
    graph.add_node(gather_context)
    graph.add_node(write_talk)
    graph.add_edge(START,"plan_talk")
    graph.add_edge("plan_talk","gather_context")
    graph.add_edge("gather_context","write_talk")
    graph.add_edge("write_talk",END)
    
    return graph.compile()