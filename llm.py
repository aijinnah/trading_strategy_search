import os
import time

import openai
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

from utils import setup_logger
from vars import LOG_FILE

from prompts import (
    GENERATE_NEW_STRATEGY_PROMPT,
    GENERATE_STRATEGY_PROMPT,
    format_string,
)

logger = setup_logger("Logger1", LOG_FILE)

load_dotenv(".env")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION")

oai_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def generate_new_strategy_code(
    sampled_strategies,
    provider="openai",
    google_model="gemini-2.0-flash-001",
):
    """
    Creates a new 'generate_orders' function by prompting an LLM with
    performance logs and prior strategy code. Returns the code as a string.
    """

    sampled_strategies_str = ""
    for i, strat_dict in enumerate(sampled_strategies):
        strat_code = strat_dict["current"]
        strat_trade_log = strat_dict["trade_log"]

        sampled_strategies_str += f"generate_orders {i+1}:\n"
        sampled_strategies_str += f"```python\n{strat_code}\n```\n\n"
        for win in strat_dict["window_scores"]:
            win_trade_log = strat_dict["window_scores"][win]["trade_log"]
            sampled_strategies_str += (
                f"generate_orders {i+1} Window {win} Performance:\n"
                f"{win_trade_log}\n\n"
            )

    prompt = format_string(
        GENERATE_STRATEGY_PROMPT, sampled_strategies_str=sampled_strategies_str
    )

    if provider == "openai":
        try:
            openai.api_key = OPENAI_API_KEY
            response = oai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=0.8,
            )
            new_code = response.choices[0].message.content.strip()
        except Exception as e:
            logger.info(f"[OpenAI] Error generating new strategy: {str(e)}")
            # fallback
            return generate_new_strategy_code_openai(
                model=google_model, provider="google"
            )
    else:
        # Example usage with Google GenAI
        try:
            client = genai.Client(
                vertexai=True, project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION
            )
            contents = [
                types.Content(role="user", parts=[types.Part.from_text(prompt)])
            ]
            config = types.GenerateContentConfig(
                temperature=1.2, max_output_tokens=8000, response_modalities=["TEXT"]
            )
            result = client.models.generate_content(
                model=google_model, contents=contents, config=config
            )
            new_code = result.candidates[0].content.parts[0].text
        except Exception as e:
            logger.info(f"[Google GenAI] Error generating new strategy: {str(e)}")

            if "resource_exhausted" in str(e).lower():
                print(f"Sleeping....")
                time.sleep(60)

            return generate_new_strategy_code_openai(
                model=google_model, provider="google"
            )

    # Extract code from triple backticks if present
    if "```python" in new_code:
        parts = new_code.split("```python")
        if len(parts) > 1:
            segment = parts[1].split("```")[0].strip()
            new_code = segment
    elif "```" in new_code:
        new_code = new_code.split("```")[1].strip()

    # Basic safety check
    try:
        compile(new_code, "<string>", "exec")
    except Exception as e:
        logger.info(
            f"Generated code failed to compile, reverting to BASE_STRATEGY_CODE: {e}"
        )
        return generate_new_strategy_code_openai(model=google_model, provider="google")

    return new_code


def generate_new_strategy_code_openai(model="gpt-4", provider="openai"):
    """
    Single-shot request to produce a 'generate_orders' function that returns a
    series and a dictionary, no print statements inside.
    """

    try:
        if provider == "openai":
            openai.api_key = OPENAI_API_KEY
            response = oai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": GENERATE_NEW_STRATEGY_PROMPT}],
                max_tokens=8192,
                temperature=0.8,
            )
            new_code = response.choices[0].message.content.strip()
        else:
            client = genai.Client(
                vertexai=True, project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION
            )
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(GENERATE_NEW_STRATEGY_PROMPT)],
                )
            ]
            config = types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=2000,
                response_modalities=["TEXT"],
            )
            result = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            new_code = result.candidates[0].content.parts[0].text

        if "```python" in new_code:
            new_code = new_code.split("```python")[1].split("```")[0].strip()
        elif "```" in new_code:
            new_code = new_code.split("```")[1].strip()

        compile(new_code, "<string>", "exec")
        return new_code
    except Exception as e:
        logger.info(f"Error generating new strategy: {str(e)}")
        # Fallback - just return a trivial strategy
        return generate_new_strategy_code_openai(
            model="gemini-2.0-flash-001", provider="google"
        )
