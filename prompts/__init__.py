from .prompts import GENERATE_NEW_STRATEGY_PROMPT, GENERATE_STRATEGY_PROMPT

def format_string(template, **kwargs):
    return template.format(**kwargs)
