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

def answer(question: str, thread_id: str) -> Answer:
    last_sources: list[str] = []

    def make_tools(store: SourceStore):
        @tool
        def search_sources(query: str) -> str:
            """Find relevant passages in the active sources and return them in a readable format."""
            nonlocal last_sources

            docs = store.search(query=query)
            if not docs:
                last_sources = []
                return "No relevant documents found in the active sources"

            last_sources = []
            for doc in docs:
                source_id = doc.metadata.get("source_id")
                if source_id and source_id not in last_sources:
                    last_sources.append(source_id)

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

    agent = create_agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        tools=make_tools(store),
    )

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    final_message = result["messages"][-1]
    text = getattr(final_message, "text", final_message.content)

    return Answer(text=text, sources=last_sources)