from functools import lru_cache
from langgraph.types import Command
from langgraph.graph.state import CompiledStateGraph

@lru_cache
def _graph() -> CompiledStateGraph:
    # Imported here, not at module level: `python -m agents.ted` imports this
    # package's __init__.py before __main__.py runs load_dotenv(), and this
    # import chain (graph -> nodes -> core.store) needs COHERE_API_KEY to
    # already be set.
    from agents.ted.graph import build_ted_graph
    from agents.ted.checkpoint import get_saver

    return build_ted_graph(checkpointer=get_saver())

def _config(job_id: str) -> dict:
    return {"configurable":{"thread_id": job_id},
            "recursion_limit":25,
            "run_name":"ted_talk",
            "metadata":{"job_id":job_id}
        }

def run_start(job_id: str):
    result = _graph().invoke({"job_id": job_id}, _config(job_id))
    return result["__interrupt__"][0].value

def run_resume(job_id: str, resume: dict) -> dict:
    """Resume a paused talk with the user's decision.

    Returns the graph's result dict: it contains "__interrupt__" when the talk
    is back at the approval gate (after a revision), or "audio_path" when it
    ran through to END.
    """
    graph = _graph()
    config = _config(job_id)
    if "approval" in graph.get_state(config).next:
        return graph.invoke(Command(resume=resume), config)
    # A previous approve already consumed the interrupt but crashed in
    # synthesize_audio; invoking with no input retries the pending node.
    return graph.invoke(None, config)

def get_values(job_id:str):
    return _graph().get_state(_config(job_id)).values

