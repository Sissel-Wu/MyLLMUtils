from typing import Any

from openai import OpenAI
import openai
from openai.types.chat import ChatCompletion
import tiktoken
from abc import ABC, abstractmethod
from os import environ
import os
from datetime import datetime
import json
from myllmutils.output_utils import CacheHelper, ResponseHelper
import httpx


class Messages(ABC):
    """
    Abstract class for messages to be sent to OpenAI's (or compatible) chat API.
    """

    @abstractmethod
    def to_openai_form(self) -> list[dict[str, str]]:
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

    def __init__(self, user_query: str, system_message: str | None = None):
        self.system_message = system_message
        self.user_query = user_query

    def to_openai_form(self) -> list[dict[str, str]]:
        rst = []
        if self.system_message is not None:
            rst.append({"role": "system", "content": self.system_message})
        rst.append({"role": "user", "content": self.user_query})
        return rst

    def __str__(self):
        return f"===System===\n{self.system_message}\n\n===User===\n{self.user_query}"


class FewShotMessages(Messages):
    """
    Messages for few-shot chat completion, in the form of
    [{"role": "system", "content": system_message},
     {"role": "user", "content": query_example_1},
     {"role": "assistant", "content": assistant_message_1},
     ...,
     {"role": "user", "content": user_query}]
    """

    def __init__(self,
                 system_message: str,
                 shots: list[tuple[str, str]],
                 user_query: str):
        self.system_message = system_message
        self.shots = shots
        self.user_query = user_query

    def to_openai_form(self) -> list[dict[str, str]]:
        rst = []
        if self.system_message is not None:
            rst.append({"role": "system", "content": self.system_message})
        for q, a in self.shots:
            rst.append({"role": "user", "content": q})
            rst.append({"role": "assistant", "content": a})
        rst.append({"role": "user", "content": self.user_query})
        return rst

    def __str__(self):
        res = f"===System===\n{self.system_message}\n\n"
        for q, a in self.shots:
            res += f"===User===\n{q}\n\n===Assistant===\n{a}\n\n"
        res += f"===User===\n{self.user_query}"
        return res


class LLMService:
    def __init__(self,
                 base_url: str | None = None,
                 api_key: str | None = None,
                 output_dir: str | None = None,
                 disable_ssl_verify: bool = False):
        """
        Initialize the LLM service.
        :param base_url: If None, the env variable "MYLLM_URL" is used.
         If the env variable is not set, env variable "OPENAI_BASE_URL" is used.
         If you want to use compatible LLMs, specify the base URL, e.g., "http://localhost:8000/v1/".
        :param api_key: If None, the env variable "MYLLM_API_KEY" is used.
         If the env variable is not set, env variable "OPENAI_API_KEY" is used.
        :param output_dir: If set, the responses are dumped to the directory.
        :param disable_ssl_verify: If True, disable SSL verification. Usually only set when using corporate VPNs. Default is False.
        """
        if base_url is None:
            base_url = environ.get("MYLLM_URL")
        if base_url is None:
            base_url = environ.get("OPENAI_BASE_URL")
        if api_key is None:
            api_key = environ.get("MYLLM_API_KEY")
        if api_key is None:
            api_key = environ.get("OPENAI_API_KEY")

        self.output_dir = output_dir
        self.cache_helper = CacheHelper(self.output_dir)

        if disable_ssl_verify:
            http_client = httpx.Client(verify=False)
            self._client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        else:
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def set_output_dir(self, output_dir: str):
        """
        Set the output directory for the LLM service.
        :param output_dir: The output directory.
        """
        self.output_dir = output_dir

    def embed(self,
              document: str,
              method: str) -> openai.types.CreateEmbeddingResponse:
        """
        Embed a document using the specified method/model.
        The results are cached and if the document is already embedded, the cached result will be returned.
        :param document: The document to embed.
        :param method: The method/model to use. Now only supports "text-embedding-3-small" or "text-embedding-3-large".
        :return: The raw response from OpenAI.
        """
        if method.startswith("text-embedding-3"):
            encoding = tiktoken.get_encoding('cl100k_base')  # for 3rd-gen embedding models TODO: compatibility with tiktoken 0.7.0
            token_sizes = len(encoding.encode(document))
            print('tokens(tiktoken): ', token_sizes)
            raw_response = self._client.embeddings.create(input=[document], model=method)
            return raw_response
        else:
            raise ValueError(f"Unknown method: {method}")

    def chat_complete(self,
                      messages: Messages,
                      model: str,
                      temperature: float | None | openai.NotGiven = openai.NOT_GIVEN,
                      return_str: bool = True,
                      title: str | None = None,
                      use_cache: bool = False,
                      **kwargs) -> str | list[str] | ChatCompletion | ResponseHelper:
        """
        Query the chat completion API with the given messages. For params not listed here, see OpenAI's API doc.
        :param messages: The messages to send.
        :param model: The model name.
        :param temperature: The temperature.
        :param return_str: If True (default), return the response as a string. Otherwise, return the raw response.
        :param title: The title for the files to dump.
        Note that the files are dumped only if self.output_dir is set.
        May overwrite the existing files.
        :param use_cache: False is default
        :return: The raw response in OpenAI format, or string if return_str is True, or ResponseHelper if return_str is False and use_cache and hit.
        """
        params = {"model": model, "temperature": temperature, **kwargs}
        response = None
        query = messages.to_openai_form()
        if use_cache:
            print("Searching for cached response...")
            response_helper = self.cache_helper.get_by_query(query, params)
            if response_helper is None:
                print("Not in cache.")
            else:
                print("Hit.")
                if return_str:
                    return response_helper.content()
                else:
                    return response_helper
        if not response:
            response = self._client.chat.completions.create(messages=query,
                                                            model=model,
                                                            temperature=temperature,
                                                            **kwargs)
            self.cache_helper.add(query, ResponseHelper(json.loads(response.model_dump_json())), params)
        return self._process_response(messages, params, response, title, return_str)

    def chat_complete_greedy(self,
                             messages: Messages,
                             model: str,
                             return_str: bool = True,
                             title: str | None = None,
                             use_cache: bool = False,
                             **kwargs) -> str | ChatCompletion:
        """
        Query the chat completion API with the given messages with the greedy sampling by setting top_p=0.
        As suggested by OpenAI, we do not set both top_p and temperature.
        Params are the same as chat_complete.
        """
        kwargs["top_p"] = 0.00000001
        return self.chat_complete(messages, model, return_str=return_str, title=title, use_cache=use_cache, **kwargs)

    def simple_chat(self,
                    message: str,
                    system_message: str | None = None,
                    model: str = "gpt-5-nano",
                    return_str: bool = True,
                    title: str | None = None,
                    use_cache: bool = False) -> str | ChatCompletion:
        """
        A simple chat function, by default returning the single response as a string.
        :param message: The message (string) to send.
        :param system_message: The system message (string) to send.
        :param model: The model name, e.g., "gpt-5-nano".
        :param return_str: If True (default), return the response as a string. Otherwise, return the raw response.
        :param title: The title for the files to dump.
        :param use_cache: See chat_complete.
        :return: A string or the raw response.
        """
        messages = ZeroShotMessages(message, system_message)
        return self.chat_complete(messages, model, return_str=return_str, title=title, use_cache=use_cache)

    @staticmethod
    def check_prompt(message: Messages) -> str:
        """
        Check the prompt without sending it to the chat completion API.
        :param message: The message to send.
        :return: a string representing the prompt.
        """
        return str(message)

    def _process_response(self,
                          messages: Messages,
                          params: dict[str, Any],
                          response: ChatCompletion,
                          title: str | None,
                          return_str: bool) -> str | list[str] | ChatCompletion:
        """
        Process the response from the chat completion API. Process includes:
        - If self.output_dir is set, save the response to the directory.
        - If return_str is True, return the first choice as a string.
        :param messages: The messages.
        :param params: The query parameters, e.g., model, temperature, top_p.
        :param response: The response.
        :param title: The title for the files to dump. If more than one response, the file titles under "output_dir/str" are appended with "-index".
        :return: The processed response.
        """
        num_choices = len(response.choices)
        index_suffices = [""] if num_choices == 1 else [f"-{i}" for i in range(num_choices)]
        if self.output_dir:
            os.makedirs(f"{self.output_dir}/raw", exist_ok=True)
            os.makedirs(f"{self.output_dir}/str", exist_ok=True)
            datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

            if title:
                raw_file = f"{self.output_dir}/raw/{title}.json"
                str_files = [f"{self.output_dir}/str/{title}{suffix}.txt" for suffix in index_suffices]
            else:
                raw_file = f"{self.output_dir}/raw/chat-{datetime_now}.json"
                str_files = [f"{self.output_dir}/str/chat-{datetime_now}{suffix}.txt" for suffix in index_suffices]

            parsed_json = json.loads(response.model_dump_json())
            with open(raw_file, "w") as f:
                obj = {"query": messages.to_openai_form(), "params": params, "response": parsed_json}
                f.write(json.dumps(obj, indent=2))

            for i in range(num_choices):
                str_file = str_files[i]
                with open(str_file, "w", encoding="utf-8") as f:
                    query_str = str(messages)
                    message = parsed_json["choices"][i]["message"]
                    resp_content = message["content"]
                    combined_str = f"{query_str}\n\n===Response===\n{resp_content}"
                    if "reasoning_content" in message:
                        combined_str += f"\n\n===Reasoning===\n{message['reasoning_content']}"
                    f.write(combined_str)

        if return_str:
            all_content = [response.choices[i].message.content for i in range(num_choices)]
            return all_content[0] if num_choices == 1 else all_content
        return response
