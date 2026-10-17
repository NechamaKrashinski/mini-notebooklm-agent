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

def run_resume(job_id: str, resume:dict):
    return _graph().invoke(Command(resume=resume),_config(job_id=job_id))

def get_values(job_id:str):
    return _graph().get_state(_config(job_id)).values

