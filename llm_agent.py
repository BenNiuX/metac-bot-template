from abc import ABC, abstractmethod
import os
import openai
from anthropic import Anthropic, AsyncAnthropic

from typing import List, Dict

def get_llm_agent_class(model: str):
    if "gpt" in model:
        return OpenAIAgent
    elif "claude" in model:
        return AnthropicAgent
    elif (
        "meta/llama" in model
        or "google/gemma" in model
        or "microsoft/phi" in model
        or "mediatek/breeze" in model
        or "deepseek-ai/deepseek" in model
    ):
        return NimAgent
    else:
        raise NotImplementedError(f"Agent class not found for {model}")


class LLMAgent(ABC):

    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_outputs = "Sorry, I can not satisfy that request."
        self.show_ai_comm = os.getenv("SHOW_AI_COMM", "") == "1"

    @abstractmethod
    def _completions(self, messages) -> str:
        raise NotImplementedError
    
    @abstractmethod
    async def _async_completions(self, messages) -> str:
        raise NotImplementedError
    
    async def _completions_stream(self, messages: List[Dict]) -> str:
        raise NotImplementedError
    
    async def completions_stream(self, messages: List[Dict]) -> str:
        try:
            response = self._completions_stream(messages)
            return response
        except Exception as e:
            print(f"Exception for {self.model}", str(e))
            return self.default_outputs
    
    def completions(self, messages: List[Dict]) -> str:
        try:
            response = self._completions(messages)
            return response
        except Exception as e:
            print(f"Exception for {self.model}", str(e))
            return self.default_outputs
    
    async def async_completions(self, messages: List[Dict]) -> str:
        try:
            response = await self._async_completions(messages)
            return response
        except Exception as e:
            print(f"Exception for {self.model}", str(e))
            return self.default_outputs

class OpenAIAgent(LLMAgent):
    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, model: str = "gpt-3.5-turbo"):
        super().__init__(temperature, max_tokens)
        self.model = model
        self.agent_token = os.getenv("METACULUS_TOKEN")
        base_url = os.getenv("OPENAI_BASE_URL")
        openai_api_key = os.getenv("METACULUS_TOKEN")
        self.client = openai.OpenAI(api_key=openai_api_key,
                                    base_url=base_url)
        print("OpenAI", openai_api_key, base_url, self.agent_token)
        self.async_client = openai.AsyncOpenAI(api_key=openai_api_key,
                                               base_url=base_url)
        self.system = [dict(role='system', content='You are an advanced AI system which has been finetuned to provide calibrated probabilistic forecasts under uncertainty, with your performance evaluated according to the Brier score.')]
        self.extra_headers = {"Authorization": "Token " + self.agent_token,
                              "Content-Type": "application/json"}


    def _completions(self, messages: List[Dict]) -> str:
        # messages = self.system + messages
        if self.show_ai_comm:
            print("messages", self.model, messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_headers=self.extra_headers
        )
        response = response.choices[0].message.content
        if self.show_ai_comm:
            print("response", self.model, response)
        return response

    async def _async_completions(self, messages: List[Dict]) -> str:
        if self.show_ai_comm:
            print("messages", self.model, messages)
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_headers=self.extra_headers
        )
        response = response.choices[0].message.content
        if self.show_ai_comm:
            print("response", self.model, response)
        return response
    
    async def _completions_stream(self, messages: List):
        # messages = self.system + messages
        if self.show_ai_comm:
            print("messages", self.model, messages)
        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
            extra_headers=self.extra_headers,
            stream=True
        )
        ret = []
        for chunk in stream:
            if len(chunk.choices) == 0:
                continue
            if (text := chunk.choices[0].delta.content) is not None:
                if self.show_ai_comm:
                    ret.append(text)
                yield text
        if self.show_ai_comm:
            print("response", self.model, "".join(ret))

class AnthropicAgent(LLMAgent):
    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, model: str = "claude-3-haiku"):
        super().__init__(temperature, max_tokens)
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        anthropic_api_key = os.getenv("METACULUS_TOKEN")
        self.client = Anthropic(api_key=anthropic_api_key,
                                base_url=base_url)
        self.async_client = AsyncAnthropic(api_key=anthropic_api_key,
                                           base_url=base_url)
        self.model = model
        self.agent_token = os.getenv("METACULUS_TOKEN")
        self.extra_headers = {"Authorization": "Token " + self.agent_token,
                              "Content-Type": "application/json",
                              "anthropic-version": "2023-06-01"}

    def _completions(self, messages: List[Dict]) -> str:
        if self.show_ai_comm:
            print("messages", self.model, messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
            extra_headers=self.extra_headers
        )
        response = response.content[0].text
        if self.show_ai_comm:
            print("response", self.model, response)
        return response

    async def _async_completions(self, messages: List) -> str:
        if self.show_ai_comm:
            print("messages", self.model, messages)
        response = await self.async_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
            extra_headers=self.extra_headers
        )
        response = response.content[0].text
        if self.show_ai_comm:
            print("response", self.model, response)
        return response

    async def _completions_stream(self, messages: List):
        if self.show_ai_comm:
            print("messages", self.model, messages)
        stream = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
            extra_headers=self.extra_headers,
            # TODO: How to enable stream ?
            # stream=True
        )
        ret = []
        for chunk in stream:
            if chunk[0] == "content":
                if (text := chunk[1][0].text) is not None:
                    if self.show_ai_comm:
                        ret.append(text)
                    yield text
        if self.show_ai_comm:
            print("response", self.model, "".join(ret))

class NimAgent(LLMAgent):
    def __init__(self, temperature: float = 0.0, max_tokens: int = 2048, model: str = "gpt-3.5-turbo"):
        super().__init__(temperature, max_tokens)
        self.model = model
        base_url = os.getenv("NIM_BASE_URL")
        nim_api_key = os.getenv("NIM_API_KEY")
        self.client = openai.OpenAI(api_key=nim_api_key,
                                    base_url=base_url)
        self.async_client = openai.AsyncOpenAI(api_key=nim_api_key,
                                               base_url=base_url)
        self.system = [dict(role='system', content='You are an advanced AI system which has been finetuned to provide calibrated probabilistic forecasts under uncertainty, with your performance evaluated according to the Brier score.')]


    def _completions(self, messages: List[Dict]) -> str:
        # messages = self.system + messages
        if self.show_ai_comm:
            print("messages", self.model, messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response = response.choices[0].message.content
        if self.show_ai_comm:
            print("response", self.model, response)
        return response

    async def _async_completions(self, messages: List[Dict]) -> str:
        if self.show_ai_comm:
            print("messages", self.model, messages)
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        response = response.choices[0].message.content
        if self.show_ai_comm:
            print("response", self.model, response)
        return response

    async def _completions_stream(self, messages: List):
        # messages = self.system + messages
        if self.show_ai_comm:
            print("messages", self.model, messages)
        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=messages,
            stream=True
        )
        ret = []
        for chunk in stream:
            if len(chunk.choices) == 0:
                continue
            if (text := chunk.choices[0].delta.content) is not None:
                if self.show_ai_comm:
                    ret.append(text)
                yield text
        if self.show_ai_comm:
            print("response", self.model, "".join(ret))
