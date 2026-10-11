from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_source(source_id:str, name:str, content:str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True
    )
    doc = Document(page_content=content, metadata={"source":name, "source_id":source_id})
    return splitter.split_documents([doc])

def format_docs(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def build_sources_overview(store) -> str:
    parts = []
    for s in store.list():
        if not s.active:
            continue
        summary = s.summary or s.content[:500]
        parts.append(f"### {s.name}\n{summary}")
    return "\n\n".join(parts)  