"""
Generation (LLM layer)
=======================
Concept: This is the final step -- take the prompt (question + context)
and have a language model write an answer grounded in that context.

Why this file is structured as a "provider" abstraction:
The project asked for a configurable LLM layer so you can start free/local
and later switch to OpenAI, Ollama, Gemini, etc. by changing ONE line in
.env (LLM_PROVIDER=...), without touching any other file.

Providers included:
- "mock":   No API key, no internet, no GPU needed. Extracts the most
            relevant sentence from the context using simple heuristics.
            Great for testing the whole pipeline (API, telemetry, evals)
            for free before you hook up a real LLM.
- "ollama": Free, runs a real open-source LLM (e.g. llama3.1) on your own
            machine via https://ollama.com. No API key needed.
- "openai": Uses OpenAI's API. Needs OPENAI_API_KEY.
- "gemini": Uses Google's Gemini API. Needs GEMINI_API_KEY.
"""

import re
from dataclasses import dataclass
from typing import Optional

import requests

from app.config import settings


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model: str = ""


class BaseLLM:
    name = "base"

    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "as", "and",
    "or", "but", "if", "do", "does", "did", "have", "has", "had", "it",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "what", "when", "where", "who", "how", "which", "can", "will", "would",
    "my", "your", "our", "their", "me", "us", "them",
}


class MockLLM(BaseLLM):
    """
    A zero-dependency 'LLM' so the whole pipeline works out of the box
    with no API key. It picks the sentence in the context most similar
    (by meaningful-word overlap) to the question -- crude, but
    deterministic and free, and it still respects "don't know if not
    present in context". Stopwords (the, of, is, ...) are excluded from
    the overlap count so common words don't cause false-positive matches.
    """

    name = "mock"

    def generate(self, prompt: str) -> LLMResponse:
        # Pull the context and question back out of the prompt we built.
        context_match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.DOTALL)
        question_match = re.search(r"Question: (.*?)\n\nAnswer:", prompt, re.DOTALL)
        context = context_match.group(1) if context_match else ""
        question = question_match.group(1) if question_match else ""

        # Strip our own "[Source N: ...]" tags before splitting into sentences,
        # otherwise a tag line with no sentence-ending punctuation gets fused
        # with the real sentence that follows it.
        context_no_tags = re.sub(r"\[Source \d+:.*?\]\n?", "", context)
        context_no_tags = re.sub(r"</?evidence[^>]*>\n?", "\n", context_no_tags)  # new evidence fences

        def meaningful_words(text: str) -> set:
            words = re.findall(r"\w+", text.lower())
            return {w for w in words if w not in STOPWORDS and len(w) > 1}

        question_words = meaningful_words(question)
        # Split on sentence ends AND line breaks, so section headings are not glued onto answers.
        sentences = re.split(r"(?<=[.!?])\s+|\n+", context_no_tags)

        best_sentence, best_overlap = "", 0
        for sentence in sentences:
            words = meaningful_words(sentence)
            overlap = len(words & question_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()

        # Require at least 2 meaningful overlapping words so a single
        # coincidental match (e.g. one shared noun) doesn't produce a
        # confident-looking but wrong answer.
        if not best_sentence or best_overlap < 2:
            answer = "I don't know based on the provided documents."
        else:
            answer = best_sentence

        return LLMResponse(
            text=answer,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(answer.split()),
            model="mock-extractive-v1",
        )


class OllamaLLM(BaseLLM):
    name = "ollama"

    def generate(self, prompt: str) -> LLMResponse:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=data.get("response", "").strip(),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            model=settings.ollama_model,
        )


class OpenAILLM(BaseLLM):
    name = "openai"

    def generate(self, prompt: str) -> LLMResponse:
        # Imported lazily so the package isn't a hard dependency unless used.
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = completion.choices[0].message.content
        usage = completion.usage
        return LLMResponse(
            text=choice.strip(),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            model=settings.openai_model,
        )


class GeminiLLM(BaseLLM):
    name = "gemini"

    def generate(self, prompt: str) -> LLMResponse:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
        )
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return LLMResponse(text=text.strip(), model=settings.gemini_model)


def get_llm() -> BaseLLM:
    """
    Factory function: reads LLM_PROVIDER from settings and returns the
    matching provider instance. This is the ONE place that decides which
    LLM class gets used -- everything else in the app just calls
    `llm.generate(prompt)` without knowing which provider it is.
    """
    provider = settings.llm_provider.lower()
    if provider == "mock":
        return MockLLM()
    elif provider == "ollama":
        return OllamaLLM()
    elif provider == "openai":
        return OpenAILLM()
    elif provider == "gemini":
        return GeminiLLM()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
