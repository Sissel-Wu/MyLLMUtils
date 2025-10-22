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
import concurrent.futures
from threading import Lock
import queue


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


class CompletionMessages(Messages):
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

    def to_openai_form(self) -> list[dict[str, str]]:
        return [{"completion": self.prompt}]

    def __str__(self):
        return f"===Prompt===\n{self.prompt}\n\n"


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
    

class LLMClientPool:
    def __init__(self, pool_size, base_url, api_key, timeout, disable_ssl_verify=False):
        self.pool_size = pool_size
        self._pool = queue.Queue(maxsize=pool_size)
        self._lock = Lock()

        for _ in range(pool_size):
            if disable_ssl_verify:
                http_client = httpx.Client(verify=False)
                client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client, timeout=timeout)
            else:
                client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            self._pool.put(client)

    def acquire(self) -> OpenAI:
        return self._pool.get()
    
    def release(self, client: OpenAI):
        self._pool.put(client)


class LLMService:
    def __init__(self,
                 base_url: str | None = None,
                 api_key: str | None = None,
                 output_dir: str | None = None,
                 timeout: float | openai.Timeout | None | openai.NotGiven = openai.not_given,
                 disable_ssl_verify: bool = False,
                 parallels: int = 1,
                 ignore_cache_params: list[str] | None = None):
        """
        Initialize the LLM service.
        :param base_url: If None, the env variable "MYLLM_URL" is used.
         If the env variable is not set, env variable "OPENAI_BASE_URL" is used.
         If you want to use compatible LLMs, specify the base URL, e.g., "http://localhost:8000/v1/".
        :param api_key: If None, the env variable "MYLLM_API_KEY" is used.
         If the env variable is not set, env variable "OPENAI_API_KEY" is used.
        :param output_dir: If set, the responses are dumped to the directory.
        :param disable_ssl_verify: If True, disable SSL verification. Usually only set when using corporate VPNs. Default is False.
        :param parallels: If >1 (default), use that number of clients to send queries in parallel.
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
        self.cache_helper = CacheHelper(self.output_dir, ignore_cache_params)

        self._clients = LLMClientPool(pool_size=parallels,
                                      base_url=base_url,
                                      api_key=api_key,
                                      timeout=timeout,
                                      disable_ssl_verify=disable_ssl_verify)
        self.parallels = parallels

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
            client = self._clients.acquire()
            try:
                raw_response = client.embeddings.create(input=[document], model=method)  # TODO parallels
            finally:
                self._clients.release(client)
            return raw_response
        else:
            raise ValueError(f"Unknown method: {method}")
        
    
    @staticmethod
    def _send_chat_complete(client_pool: LLMClientPool, n_per_query: int, messages: Messages, model: str, temperature: float | None | openai.NotGiven = openai.NOT_GIVEN, **kwargs) -> openai.types.chat.ChatCompletion:
        client = client_pool.acquire()
        try:
            response = client.chat.completions.create(messages=messages.to_openai_form(),
                                                      model=model,
                                                      temperature=temperature,
                                                      n=n_per_query,
                                                      **kwargs)
        finally:
            client_pool.release(client)
        return response


    def chat_complete(self,
                      messages: Messages,
                      model: str,
                      temperature: float | None | openai.NotGiven = openai.NOT_GIVEN,
                      return_str: bool = True,
                      title: str | None = None,
                      use_cache: bool = False,
                      n: int = 1,
                      n_limit_per_query: int = 0,
                      **kwargs) -> str | list[str] | ResponseHelper:
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
        :param n: Number of samples to obtain. Default is 1.
        :param n_limit_per_query: If >0, limit the number of responses to n_limit_per_query per unique query (messages). Default is 0 (no limit).
        :return: The raw response in OpenAI format, or string if return_str is True, or ResponseHelper if return_str is False and use_cache and hit.
        """
        params = {"model": model, "temperature": temperature, "n": n, **kwargs}
        query = messages.to_openai_form()
        if use_cache:
            print(f"Searching for cached response ({query}) ({params})...")
            response_helper = self.cache_helper.get_by_query(query, params)
            if response_helper is None:
                print("Not in cache.")
            else:
                print("Hit.")
                if return_str:
                    return response_helper.content()
                else:
                    return response_helper

        # send the request in batches
        n_per_time = min(n, n_limit_per_query) if n_limit_per_query > 0 else n
        responses = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallels) as executor:
            while n > 0:
                n_sent = min(n, n_per_time)
                future = executor.submit(self._send_chat_complete,
                                         self._clients,
                                         n_sent,
                                         messages,
                                         model,
                                         temperature,
                                         **kwargs)
                futures.append(future)
                n -= n_sent
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                responses.append(response)
            
        rh_obj = ResponseHelper([json.loads(resp.model_dump_json()) for resp in responses])
        if use_cache:
            self.cache_helper.add(query, rh_obj, params)
        return self._process_response(messages, params, rh_obj, title, return_str)

    def complete(self, prompt: str,
                 model: str,
                 temperature: float | None | openai.NotGiven = openai.NOT_GIVEN,
                 return_str: bool = True,
                 title: str | None = None,
                 use_cache: bool = False,
                 n: int = 1,
                 n_limit_per_query: int = 0,
                 **kwargs) -> str | list[str] | ResponseHelper:
        params = {"api": "complete", "model": model, "temperature": temperature, "n": n,
                  "n_limit_per_query": n_limit_per_query, **kwargs}
        messages = CompletionMessages(prompt)
        if use_cache:
            print("Searching for cached response...")
            query = messages.to_openai_form()
            response_helper = self.cache_helper.get_by_query(query, params)
            if response_helper is None:
                print("Not in cache.")
            else:
                print("Hit.")
                if return_str:
                    return response_helper.content()
                else:
                    return response_helper

            # send the request in batches
        n_per_time = min(n, n_limit_per_query) if n_limit_per_query > 0 else n
        responses = []
        while n > 0:
            n_sent = min(n, n_per_time)
            # TODO parallels
            client = self._clients.acquire()
            try:
                response = self._clients[0].completions.create(prompt=prompt,
                                                       model=model,
                                                       temperature=temperature,
                                                       n=n_sent,
                                                       **kwargs)
            finally:
                self._clients.release(client)
            responses.append(response)
            n -= n_per_time
        rh_obj = ResponseHelper([json.loads(resp.model_dump_json()) for resp in responses])
        if use_cache:
            self.cache_helper.add(query, rh_obj, params)
        return self._process_response(messages, params, rh_obj, title, return_str)

    def chat_complete_greedy(self,
                             messages: Messages,
                             model: str,
                             return_str: bool = True,
                             title: str | None = None,
                             use_cache: bool = False,
                             n: int = 1,
                             n_limit_per_query: int = 0,
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
                          resp_helper: ResponseHelper,
                          title: str | None,
                          return_str: bool) -> str | list[str] | ResponseHelper:
        """
        Process the response from the chat completion API. Process includes:
        - If self.output_dir is set, save the response to the directory.
        - If return_str is True, return the first choice as a string.
        :param messages: The messages.
        :param params: The query parameters, e.g., model, temperature, top_p.
        :param resp_helper: The response helper.
        :param title: The title for the files to dump. If more than one response, the file titles under "output_dir/str" are appended with "-index".
        :return: The processed response.
        """
        num_choices = len(resp_helper.choices)
        index_suffices = [""] if num_choices == 1 else [f"-{i}" for i in range(num_choices)]
        if self.output_dir:
            os.makedirs(f"{self.output_dir}/raw", exist_ok=True)
            os.makedirs(f"{self.output_dir}/str", exist_ok=True)
            datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

            if title:
                raw_file = f"{self.output_dir}/raw/{title}.json"
                str_files = [f"{self.output_dir}/str/{title}{suffix}.txt" for suffix in index_suffices]
            else:
                raw_file = f"{self.output_dir}/raw/chat-{datetime_now}-{resp_helper.chat_id()}.json"
                str_files = [f"{self.output_dir}/str/chat-{datetime_now}{suffix}-{resp_helper.chat_id()}.txt" for suffix in index_suffices]

            with open(raw_file, "w") as f:
                obj = {"query": messages.to_openai_form(), "params": params, "response": resp_helper.raw_response}
                f.write(json.dumps(obj, indent=2))

            for i in range(num_choices):
                str_file = str_files[i]
                with open(str_file, "w", encoding="utf-8") as f:
                    query_str = str(messages)
                    message = resp_helper.choices[i]["message"]
                    resp_content = message["content"]
                    combined_str = f"{query_str}\n\n===Response===\n{resp_content}"
                    if "reasoning_content" in message:
                        combined_str += f"\n\n===Reasoning===\n{message['reasoning_content']}"
                    f.write(combined_str)

        if return_str:
            return resp_helper.content()
        return resp_helper
