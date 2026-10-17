"""Dev runner: uv run python -m agents.ted notes.txt article.md"""

# Must run before anything that calls an LLM (store.add() -> summarize()
# included) — same workaround serve.py uses for the network's SSL interception.
from netfree_unstrict_ssl import unstrict_ssl
unstrict_ssl()

from dotenv import load_dotenv
load_dotenv()  # ANTHROPIC_API_KEY / COHERE_API_KEY must be set before any model is built

import json
import sys
from pathlib import Path

from core.store import store
from agents.ted import run_start, run_resume

# Each CLI arg is a path to a text file; load it as a source so plan_talk
# has something to plan from (the store is in-memory, so a fresh process
# starts empty).
for arg in sys.argv[1:]:
    path = Path(arg)
    store.add(name=path.name, content=path.read_text(encoding="utf-8"))
    print(f"loaded source: {path.name}")

JOB_ID = "test"

payload = run_start(JOB_ID)

# The graph pauses at the approval node and reports back via __interrupt__.
# Loop so the human can ask for more than one round of changes.
while True:
    print("--- script ---")
    print(payload["script_he"])
    print(f"\n({payload['word_count']} מילים, {payload['revision_count']} תיקונים)")
    if payload["critique_notes"]:
        print("הערות המבקר האחרונות:", payload["critique_notes"])

    answer = input("\nלאשר? (y = אישור / כל טקסט אחר = משוב לשינוי): ").strip()
    resume = {"action": "approve"} if answer.lower() == "y" else {"action": "revise", "feedback": answer}
    result = run_resume(JOB_ID, resume)
    if "__interrupt__" not in result:
        break
    payload = result["__interrupt__"][0].value

print("\nההרצאה אושרה — הגרף הסתיים.")

# result["brief"] is a validated TalkBrief (Pydantic); model_dump() turns it
# into a plain dict so json.dumps can print it readably in Hebrew.
print(json.dumps(result["brief"].model_dump(), indent=2, ensure_ascii=False))

print("--- final script ---")
print(result["script_he"])
print(f"\n({result['word_count']} מילים)")
print("revisions:", result["revision_count"])
