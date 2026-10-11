from langgraph.graph import StateGraph, START, END
from agents.ted.nodes import plan_talk
from agents.ted.state import TedState



def build_ted_graph():
    graph = StateGraph(TedState).add_node(plan_talk).add_edge(START,"plan_talk").add_edge("plan_talk",END)
    return graph.compile()