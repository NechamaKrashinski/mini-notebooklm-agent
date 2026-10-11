from langchain.chat_models import init_chat_model


def summarize(name:str, content: str) -> str | None:
    try:
        model = init_chat_model(model="anthropic:claude-sonnet-5")
        SUMMARY_PROMPT = (
            "Summarize the following source in 5-10 concise sentences, in the same "
            "language as the source. Focus on what the document is about and its "
            "key points.\n\nTitle: {name}\n\nContent:\n{content}"
        )

        result = model.invoke(SUMMARY_PROMPT.format(name=name,content=content))
        
        
        return result.text.strip()
        
    except Exception:

        return None

