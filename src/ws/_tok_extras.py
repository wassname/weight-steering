"""Tiny tokenizer utilities with no ws imports (avoids circular deps)."""


def chat_template_extras(tok) -> dict:
    """Extra kwargs for apply_chat_template that vary by model family.

    Gemma 4 family is identified by <|think|>/<|channel> in the Jinja template.
    Pass enable_thinking=False explicitly so outputs skip the thought channel
    even if the model would otherwise default to thinking mode.
    Qwen3 and Gemma 3 have no such kwarg and return {}.
    """
    template = tok.chat_template or ""
    if "<|think|>" in template or "<|channel>" in template:
        return {"enable_thinking": False}
    return {}
