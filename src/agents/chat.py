from dataclasses import dataclass

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

from core.store import SourceStore, store
from core.sources import format_docs

@dataclass
class Answer:
    text: str
    sources: list[str]


MODEL = "anthropic:claude-sonnet-4-6"
SYSTEM_PROMPT = (
    "You are an assistant for a notebook of source documents. "
    "When the user asks for information about available sources, use the tools. "
    "Use `list_sources` to return the list of available sources. "
    "Use `get_source` when the user asks for details about a specific source by id. "
    "Use `search_sources` to find relevant passages in the active sources for the user's query. "
    "Answer only using the information from the sources and tools, and do not make up facts. "
    "If the answer cannot be found in the provided sources, say that the information is not available in the sources."
)


def _make_tools(store: SourceStore):

    @tool
    def search_sources(query: str) -> str:
        """Find passages in the active sources that are relevant to a query"""
        docs = store.search(query=query)
        if not docs:
            return "No relevant documents found in the active sources"
        return format_docs(docs)

    @tool
    def list_sources() -> list[dict]:
        """Return a list of sources currently available in the store."""
        return [
            {"id": source.id, "name": source.name, "active": source.active}
            for source in store.list()
        ]

    @tool
    def get_source(source_id: str) -> dict | str:
        """Return a source by its identifier."""
        source = store.get(source_id)
        if source is None:
            return f"Source with id '{source_id}' not found."
        return {
            "id": source.id,
            "name": source.name,
            "active": source.active,
            "content": source.content,
        }

    return [search_sources, list_sources, get_source]


def answer(question: str, thread_id: str) -> Answer:
    agent = create_agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        tools=_make_tools(store),
    )

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config=config
    )

    text = result["messages"][-1].text
    # TODO add sources to answer
    return Answer(text=text, sources=[])