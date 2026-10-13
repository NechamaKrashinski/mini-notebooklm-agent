import re

from agents.ted.state import MAX_REVISIONS, TedState,HARD_CAP, WORD_MAX, WORD_MIN
from agents.ted.prompts import PLAN_TALK_SYSTEM,PLAN_TALK_USER,WRITE_TALK_SYSTEM,WRITE_TALK_USER,CRITIQUE_SYSTEM,CRITIQUE_USER,REVISE_SYSTEM,REVISE_USER
from agents.ted.models import get_model
from core.sources import build_sources_overview, format_docs
from core.store import store
from agents.ted.schemas import CritiqueResult, TalkBrief
from agents.ted.state import TARGET_WORDS, WORD_MIN, WORD_MAX

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


def write_talk(state:TedState) -> dict:
    context = state["context"]
    brief = state["brief"]
    llm = get_model("writing")
    write = llm.invoke([
        {"role": "system", "content": WRITE_TALK_SYSTEM.format(target_words=TARGET_WORDS,word_min=WORD_MIN,word_max=WORD_MAX)},
        {"role": "user", "content": WRITE_TALK_USER.format(topic=state["brief"].topic,hook=state["brief"].hook,key_points="\n".join(f"- {p}" for p in brief.key_points),context=context)}
    ])
    return {"script_he": write.text.strip()}


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


def count_words(state: TedState) -> dict:
    script = state["script_he"]
    if len(script.split()) <= HARD_CAP:
        return {"word_count":len(script.split())}
    kept, used = [], 0
    for sentence in _sentences(script):
        n = len(sentence.split())
        if kept and used + n > HARD_CAP:
            break
        kept.append(sentence)
        used += n
    trimmed = " ".join(kept)
    return {"script_he": trimmed, "word_count": len(trimmed.split())}

def critique_script(state: TedState) -> dict:
    brief = state["brief"]
    llm = get_model("critique").with_structured_output(CritiqueResult)
    critique = llm.invoke([
        {"role": "system", "content": CRITIQUE_SYSTEM.format(word_min=WORD_MIN,word_max=WORD_MAX)},
        {"role": "user", "content": CRITIQUE_USER.format(topic=brief.topic,hook=brief.hook,key_points="\n".join(f"- {p}" for p in brief.key_points),word_count=state["word_count"],script=state["script_he"])}
    ])
    return {"critique":critique}

def after_critique(state: TedState) -> dict:
    if state["critique"].passed or state["revision_count"] >= MAX_REVISIONS:
        return "approval"
    return "revise"

def revise(state: TedState) -> dict:
    critique =state.get("critique")
    issues = "\n".join(f"- {issue}" for issue in critique.issues) if critique and critique.issues else "אין"
    llm = get_model("writing")
    response = llm.invoke([
        {"role": "system", "content": REVISE_SYSTEM.format(target_words=TARGET_WORDS, word_min=WORD_MIN, word_max=WORD_MAX)},
        {"role": "user", "content": REVISE_USER.format(
            script=state["script_he"],
            word_count=state["word_count"],
            issues=issues,
            human_feedback="אין",
            context=state["context"],
        )}
    ])
    return {
        "script_he": response.text.strip(),
        "revision_count": state["revision_count"] + 1,
    }
