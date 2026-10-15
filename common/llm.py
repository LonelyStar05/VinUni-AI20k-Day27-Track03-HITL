"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter."""

import os

from langchain_openai import ChatOpenAI


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    # Gemini via its OpenAI-compatible endpoint — used when GEMINI_API_KEY is set.
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        return ChatOpenAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
            base_url=GEMINI_BASE_URL,
            api_key=gemini_key,
            max_retries=2,  # free tier = 20 req/day per model; keep retries low
            temperature=temperature,
        )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — copy .env.example to .env")
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        temperature=temperature,
    )
