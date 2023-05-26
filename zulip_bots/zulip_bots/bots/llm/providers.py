from typing import Dict, Type, Protocol

import openai
from openai.error import OpenAIError


class LLMProviderError(Exception):
    pass

class LLMProvider(Protocol):
    """
    An LLM Provider is responsible for connecting to a provider
    and use the service to complete a set of predefined tasks like
    summarization with prompt engineering.
    """

    def initialize(self, config: Dict[str, str]) -> None:
        ...

    def summarize(self, text: str) -> str:
        ...


class OpenAIProvider:
    def initialize(self, config: Dict[str, str]) -> None:
        openai.api_key = config["openai_api_key"]

    def summarize(self, text: str) -> str:
        prompt = f"Summarize this conversation:\n{text}"
        print("PROMPT", prompt)
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "assistant",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=1000,
            )
        except OpenAIError as e:
            raise LLMProviderError(str(e))
        print("COMPLETION", completion)
        return completion.choices[0].message.content


class VicunaProvider:
    def initialize(self, config: Dict[str, str]) -> None:
        ...

    def summarize(self, text: str) -> str:
        ...


class EchoProvider:
    def initialize(self, config: Dict[str, str]) -> None:
        pass

    def summarize(self, text: str) -> str:
        return text


SUPPORTED_PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "vicuna": VicunaProvider,
    "debug": EchoProvider,
}

DEFAULT_PROVIDER = "debug"
