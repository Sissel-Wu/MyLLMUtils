from openai import OpenAI
import openai
from openai.types.chat import ChatCompletion
import tiktoken
from abc import ABC, abstractmethod


class Messages(ABC):
    """
    Abstract class for messages to be sent to OpenAI's (or compatible) chat API.
    """
    @abstractmethod
    def to_openai_form(self):
        """
        Convert the messages to the form that OpenAI's chat API expects.
        :return: A list of {"role": "system" | "assistant" | "user", "content": "xxx"}
        """
        pass


class ZeroShotMessages(Messages):
    """
    Messages for zero-shot chat completion, in the form of
    [{"role": "system", "content": system_message}, {"role": "user", "content": user_query}]
    """
    def __init__(self, user_query: str, system_message=None):
        self.system_message = system_message
        self.user_query = user_query

    def to_openai_form(self):
        rst = []
        if self.system_message is not None:
            rst.append({"role": "system", "content": self.system_message})
        rst.append({"role": "user", "content": self.user_query})
        return rst

    def __str__(self):
        return f"===System===\n{self.system_message}\n\n===User===\n{self.user_query}"


class LLMService:
    def __init__(self, base_url: str | None = None, api_key: str = "EMPTY"):
        """
        Initialize the LLM service.
        :param base_url: By default, the OpenAI API is used.
        If you want to use compatible LLMs, specify the base URL, e.g., "http://localhost:8000/v1/".
        """
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key,
                              base_url=base_url)

    def embed(self, document: str, method: str) -> openai.types.CreateEmbeddingResponse:
        """
        Embed a document using the specified method/model.
        The results are cached and if the document is already embedded, the cached result will be returned.
        :param document: The document to embed.
        :param method: The method/model to use. Now only supports "text-embedding-3-small" or "text-embedding-3-large".
        :return: The raw response from OpenAI.
        """
        if method.startswith("text-embedding-3"):
            encoding = tiktoken.get_encoding('cl100k_base')  # for 3rd-gen embedding models
            token_sizes = len(encoding.encode(document))
            print('tokens(tiktoken): ', token_sizes)
            raw_response = self._client.embeddings.create(input=[document], model=method)
            return raw_response
        else:
            raise ValueError(f"Unknown method: {method}")

    def chat_complete(self,
                      messages: Messages,
                      model: str,
                      temperature: float | None | openai.NotGiven) -> ChatCompletion:
        """
        Query the chat completion API with the given messages.
        :param messages: The messages to send.
        :param model: The model name.
        :param temperature: The temperature.
        :return: The raw response in OpenAI format.
        """
        response = self._client.chat.completions.create(messages=messages.to_openai_form(),
                                                        model=model,
                                                        temperature=temperature)
        return response

    def simple_chat(self, message: str, model: str) -> ChatCompletion:
        response = self._client.chat.completions.create(messages=ZeroShotMessages(message).to_openai_form(),
                                                        model=model)
        return response
