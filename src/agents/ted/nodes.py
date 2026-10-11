from agents.ted.state import TedState
from agents.ted.prompts import PLAN_TALK_SYSTEM,PLAN_TALK_USER 
from agents.ted.models import get_model
from core.sources import build_sources_overview, format_docs
from core.store import store
from agents.ted.schemas import TalkBrief


def plan_talk(state: TedState) -> dict:
    overview = build_sources_overview(store=store)
    if not overview:
        raise ValueError("No Sources Please add Sources")
    llm = get_model("planning").with_structured_output(TalkBrief)
    brief = llm.invoke([
        {"role": "system", "content": PLAN_TALK_SYSTEM},
        {"role": "user", "content": PLAN_TALK_USER.format(source_overview=overview)}
    ])
    return {"brief": brief, "revision_count": 0}

def gather_context(state: TedState) -> dict:
    docs, seen = [], set()
    for key in state["brief"].key_points:
        for doc in store.search(key, k=4):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)

    context = format_docs(docs) if docs else "לא נמצאו קטעים רלוונטיים במקורות."
    return {"context": context}
