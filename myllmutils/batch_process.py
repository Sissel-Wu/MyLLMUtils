"""
A standalone, multi-threaded/async, resumable batch processor for LLM APIs.

This script reads queries from one or more JSONL file, sends them to an LLM API
concurrently, and saves the results to the corresponding output JSONL files.

It supports:
- Resuming progress (skips queries already in the output files)
- Multi-threading (sync mode) or async/await (async mode) for concurrent API calls
- Error handling with exponential backoff for server-side errors
- Caching results as they are completed
- Assembling streamed responses (including text, tool calls, and reasoning)

Requirements:
- httpx: `pip install httpx[socks]` (for both sync and async modes)

Input JSONL Format:
Each line must be a valid JSON object with a unique "custom_id" field.
All other fields will be passed to the API (after modification by `build_payload`).
Example:
{"custom_id": "req-1", "model": "gpt-5-nano", "messages": [{"role": "user", "content": "Hello"}]}
{"custom_id": "req-2", "body": {"messages": [{"role": "user", "content": "How are you?"}]}}

Output JSONL Format:
Each line is a JSON object containing the original "custom_id" and the
"response" from the API.
Example:
{"custom_id": "req-1", "response": {"id": "...", "choices": [...], ...}, "query": {...}}
{"custom_id": "req-2", "response": {"id": "...", "choices": [...], ...}, "query": {...}}

Usage (Sync mode - default):
python batch_processor.py \
    --input_file queries.jsonl \
    --output_file results.jsonl \
    --api_config model_config.json \
    --max_workers 16 \
    --stream

Usage (Async mode - better concurrency):
python batch_processor.py \
    --input_file queries.jsonl \
    --output_file results.jsonl \
    --api_config model_config.json \
    --max_workers 100 \
    --async \
    --stream

OR

python batch_processor.py \
    --io mapping.json \
    --api_config model_config.yaml \
    --max_workers 16 \
    --stream

NEW in async mode (--async flag):
- Uses asyncio instead of ThreadPoolExecutor
- Better concurrency for I/O-bound workloads (3-5x throughput improvement)
- Lower memory footprint (coroutines vs threads)
- Recommended for >50 concurrent workers
- Requires httpx: pip install 'httpx[socks]'
"""

import argparse
import json
import logging
import os
import sys
import time
import threading
import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Any, Optional, Set, Tuple, Type, Union
from collections import defaultdict

# Import httpx for both sync and async HTTP requests
try:
    import httpx
except ImportError:
    print("Error: The 'httpx' library is not installed.", file=sys.stderr)
    print("Please install it by running: pip install 'httpx[socks]'", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


# --- Protocol Abstraction ---


class BaseProtocol(ABC):
    """Abstract base class for LLM API protocols."""

    @abstractmethod
    def get_endpoint(self, base_url: str, model: str) -> str:
        """Build the API endpoint URL."""
        pass

    @abstractmethod
    def get_default_auth_header(self) -> str:
        """Return the default header key for this protocol (e.g., 'Authorization')."""
        pass

    @abstractmethod
    def format_default_auth_value(self, api_key: str) -> str:
        """Return the formatted default header value for this protocol (e.g., 'Bearer {key}')."""
        pass

    def get_headers(self, api_key: str, auth_header: Optional[str] = None, auth_value: Optional[str] = None) -> Dict[str, str]:
        """Build request headers.

        Args:
            api_key: The API key.
            auth_header: Optional custom header key. If None, uses protocol default.
            auth_value: Optional custom header value. Supports {api_key} placeholder.
                        If None, uses protocol default.
        """
        if auth_header is None:
            auth_header = self.get_default_auth_header()
        if auth_value is None:
            auth_value = self.format_default_auth_value(api_key)
        else:
            # Support {api_key} placeholder in custom auth_value
            auth_value = auth_value.replace("{api_key}", api_key)

        return {
            auth_header: auth_value,
            "Content-Type": "application/json"
        }

    @abstractmethod
    async def process_stream(
        self,
        response,
        max_stream_tokens: int,
        token_counter,
        is_async: bool = False,
    ) -> Dict[str, Any]:
        """Process a streamed response and assemble the final result.

        Args:
            response: httpx.Response object
            max_stream_tokens: Maximum tokens to allow in assembled response
            token_counter: TokenCounter instance for token limits
            is_async: True for async responses (aiter_lines), False for sync (iter_lines)

        Returns:
            Assembled response dict mimicking non-streaming format.
        """
        pass


class OpenAIProtocol(BaseProtocol):
    """OpenAI-compatible chat completions protocol."""

    def get_endpoint(self, base_url: str, model: str) -> str:
        return f"{base_url.rstrip('/')}/chat/completions"

    def get_default_auth_header(self) -> str:
        return "Authorization"

    def format_default_auth_value(self, api_key: str) -> str:
        return f"Bearer {api_key}"

    async def process_stream(
        self,
        response,
        max_stream_tokens: int,
        token_counter,
        is_async: bool,
    ) -> Dict[str, Any]:
        """Process OpenAI-compatible SSE stream and assemble response.

        *** ASSUMES OPENAI-COMPATIBLE SSE FORMAT ***
        This now handles:
        - Content deltas (e.g., data: {"id": ..., "choices": [{"delta": {"content": ...}}]})
        - Tool call deltas (e.g., data: {"id": ..., "choices": [{"delta": {"tool_calls": [...]}}]})
        - Reasoning deltas (e.g., data: {"id": ..., "choices": [{"delta": {"reasoning_content": ...}}]})
        - A final `data: [DONE]`

        Args:
            is_async: True for async responses (aiter_lines), False for sync (iter_lines)
        """
        full_content = ""
        full_reasoning = ""
        tool_call_chunks = {}
        model_info = ""
        response_id = ""
        finish_reason = "stop"            

        try:
            if is_async:
                async for line in response.aiter_lines():
                    if line and line.startswith('data: '):
                        full_content, full_reasoning, tool_call_chunks, response_id, model_info, finish_reason = await self._process_stream_line(
                            line, full_content, full_reasoning, tool_call_chunks,
                            response_id, model_info, finish_reason,
                            max_stream_tokens, token_counter, response
                        )
                        # Break if token limit reached
                        if finish_reason == 'length':
                            break
            else:
                for line in response.iter_lines():                    
                    if line.startswith('data: '):
                        full_content, full_reasoning, tool_call_chunks, response_id, model_info, finish_reason = await self._process_stream_line(
                            line, full_content, full_reasoning, tool_call_chunks,
                            response_id, model_info, finish_reason,
                            max_stream_tokens, token_counter, response
                        )
                        # Break if token limit reached
                        if finish_reason == 'length':
                            break

        except Exception:
            logging.exception("Error while processing stream")
            raise
        finally:
            if is_async:
                await response.aclose()
            else:
                response.close()

        # Convert dict of chunks to a final list
        assembled_tool_calls = []
        if tool_call_chunks:
            sorted_indices = sorted(tool_call_chunks.keys())
            assembled_tool_calls = [tool_call_chunks[i] for i in sorted_indices]

        # Build the final message object
        message = {"role": "assistant"}
        if full_content:
            message["content"] = full_content
        else:
            message["content"] = None if assembled_tool_calls else ""

        if assembled_tool_calls:
            message["tool_calls"] = assembled_tool_calls

        if full_reasoning:
            message["reasoning_content"] = full_reasoning

        return {
            "id": response_id,
            "model": model_info,
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
        }

    async def _process_stream_line(
        self,
        line: str,
        full_content: str,
        full_reasoning: str,
        tool_call_chunks: dict,
        response_id: str,
        model_info: str,
        finish_reason: str,
        max_stream_tokens: int,
        token_counter,
        response,
    ):
        """Process a single SSE line from a streaming response."""
        data_str = line[len('data: '):]
        if data_str.strip() == '[DONE]':
            return (full_content, full_reasoning, tool_call_chunks, response_id, model_info, finish_reason)

        try:
            chunk = json.loads(data_str)
            if not response_id:
                response_id = chunk.get('id', '')
                model_info = chunk.get('model', '')

            choices = chunk.get('choices', [])
            if not choices:
                return (full_content, full_reasoning, tool_call_chunks, response_id, model_info, finish_reason)

            delta = choices[0].get('delta', {})
            if not delta:
                return (full_content, full_reasoning, tool_call_chunks, response_id, model_info, finish_reason)

            # Process Content Deltas
            if delta.get('content'):
                full_content += delta['content']

            # Process Reasoning Deltas
            if delta.get('reasoning_content'):
                full_reasoning += delta['reasoning_content']
            elif delta.get('reasoning'):
                full_reasoning += delta['reasoning']

            # Enforce max stream token limit
            if token_counter is not None and max_stream_tokens and max_stream_tokens > 0:
                try:
                    combined_text = (full_content or "") + ("\n" + full_reasoning if full_reasoning else "")
                    loop = asyncio.get_event_loop()
                    total_tokens = await loop.run_in_executor(
                        None,
                        token_counter.count_tokens,
                        combined_text
                    )
                    if total_tokens > max_stream_tokens:
                        finish_reason = 'length'
                        # Return accumulated state with finish_reason='length' to preserve content
                        return (full_content, full_reasoning, tool_call_chunks, response_id, model_info, finish_reason)
                except Exception:
                    logging.warning("Token counting failed during streaming; continuing without enforcement.")

            # Process Tool Call Deltas
            if delta.get('tool_calls'):
                for tool_delta in delta['tool_calls']:
                    index = tool_delta.get('index')
                    if index is None:
                        continue

                    if index not in tool_call_chunks:
                        tool_call_chunks[index] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        }

                    if tool_delta.get('id'):
                        tool_call_chunks[index]['id'] = tool_delta['id']
                    if tool_delta.get('type'):
                        tool_call_chunks[index]['type'] = tool_delta['type']

                    if 'function' in tool_delta:
                        if tool_delta['function'].get('name'):
                            tool_call_chunks[index]['function']['name'] = tool_delta['function']['name']
                        if tool_delta['function'].get('arguments'):
                            tool_call_chunks[index]['function']['arguments'] += tool_delta['function']['arguments']

            # Capture Finish Reason
            if choices[0].get('finish_reason'):
                finish_reason = choices[0].get('finish_reason')

        except json.JSONDecodeError:
            logging.warning("Failed to decode JSON chunk: %s", data_str)

        return (full_content, full_reasoning, tool_call_chunks, response_id, model_info, finish_reason)


class GeminiProtocol(BaseProtocol):
    """Google Gemini 3 generateContent protocol."""

    def get_endpoint(self, base_url: str, model: str) -> str:
        return f"{base_url.rstrip('/')}/models/{model}:generateContent"

    def get_default_auth_header(self) -> str:
        return "x-goog-api-key"

    def format_default_auth_value(self, api_key: str) -> str:
        return api_key

    async def process_stream(
        self,
        response,
        max_stream_tokens: int,
        token_counter,
        is_async: bool
    ) -> Dict[str, Any]:
        """Process Gemini stream.

        TODO: Implement Gemini-specific stream processing.
        For now, returns raw response to avoid breaking existing code.
        """
        logging.warning("Gemini stream processing not yet implemented. Returning raw response.")
        # Return the raw response for now - this will be handled differently
        return {"error": "Gemini stream processing not implemented"}


# Protocol registry
PROTOCOLS: Dict[str, Type[BaseProtocol]] = {
    "openai": OpenAIProtocol,
    "gemini": GeminiProtocol,
    "openai-compatible": OpenAIProtocol,  # Alias
}


def get_protocol(protocol_name: str) -> BaseProtocol:
    """Factory function to get protocol instance by name."""
    protocol_cls = PROTOCOLS.get(protocol_name.lower())
    if protocol_cls is None:
        raise ValueError(f"Unknown protocol: {protocol_name}. Available protocols: {list(PROTOCOLS.keys())}")
    return protocol_cls()


# --- Request Preparation ---


@dataclass
class RequestContext:
    """Pre-processed request data shared by sync and async implementations."""
    custom_id: str
    payload: Dict[str, Any]
    endpoint: str
    headers: Dict[str, str]
    stream: bool
    protocol: BaseProtocol  # Protocol instance for processing requests


def _prepare_request_context(
    task_data: Dict[str, Any],
    api_config: str | Dict[str, Any],
    mask_input_fields: str,
    stream: bool,
) -> Union[RequestContext, str]:
    """
    Common preparation logic for both sync and async.

    Validates input, loads config, builds payload, resolves API key,
    and builds endpoint/headers using protocol.

    Args:
        task_data: The JSON object with custom_id and body.
        api_config: Path to config file or config dict.
        mask_input_fields: Comma-separated fields to ignore from input.
        stream: Whether to enable streaming mode.

    Returns:
        RequestContext: Prepared request data.
        str: Error message if preparation failed.
    """
    custom_id = task_data.get("custom_id")
    if not custom_id:
        logging.error("Task data missing 'custom_id'. Skipping: %s", task_data)
        return f"Task data missing 'custom_id'. Skipping: {task_data}"

    api_config = load_api_config(api_config)
    payload = build_payload(task_data, api_config, mask_input_fields)

    if 'model' not in payload:
        logging.error("No 'model' specified in api_config or task_data for task %s.", custom_id)
        return f"No 'model' specified in api_config or task_data for task {custom_id}."

    if stream:
        payload['stream'] = True

    # Get protocol instance
    protocol_name = api_config.get("protocol", "openai")
    protocol = get_protocol(protocol_name)

    base_url = api_config.get('base_url', None)
    api_key = api_config.get('api_key', None)

    if not base_url:
        logging.error("No 'base_url' specified in the config.")
        return "No 'base_url' specified in the config."

    if not api_key:
        logging.warning("No 'api_key' specified in the config.")
        api_key = "NONE"

    if api_key.startswith("env::"):
        env_key = api_key[len("env::"):]
        api_key = os.getenv(env_key)
        if not api_key:
            logging.error("The environment %s after 'env::' is not set.", env_key)
            return f"The environment {env_key} after 'env::' is not set."

    # Build endpoint and headers using protocol
    model = payload.get('model', '')
    endpoint = protocol.get_endpoint(base_url, model)

    # Support custom auth header/value from config
    auth_header = api_config.get('auth_header', None)
    auth_value = api_config.get('auth_value', None)
    headers = protocol.get_headers(api_key, auth_header, auth_value)

    return RequestContext(
        custom_id=custom_id,
        payload=payload,
        endpoint=endpoint,
        headers=headers,
        stream=stream,
        protocol=protocol
    )


# --- API Call Logic ---

def _replace_local_url(messages):
    import base64
    from io import BytesIO
    from PIL import Image

    def encode_pil_image(pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format='PNG')
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    res = messages.copy()
    for msg in res:
        if "content" in msg:
            content = msg["content"]
            content = [content] if type(content) == str else content
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if not url.startswith("http") and not url.startswith("data"):
                        # Replace local file path with a placeholder URL
                        with Image.open(url) as img:
                            base64_encoded = encode_pil_image(img)
                            item["image_url"]["url"] = f"data:image/png;base64,{base64_encoded}"
    return res


def build_payload(task_data: Dict[str, Any],
                  api_config: Dict[str, Any],
                  mask_input_fields: str
                  ) -> Dict[str, Any]:
    """
    Prepares the payload to be sent to the API.
    The input `task_data` is the full JSON object from the input file.
    This function should return the dictionary to be sent as the API request body.

    By default, it assumes the payload is the "body" field of the task data.
    """
    if "body" in task_data:
        payload = task_data["body"]
    else:
        raise ValueError("The json does not contain a 'body'.")

    if not isinstance(payload, dict):
        raise ValueError("The 'body' field must be a dict (JSON object).")
    if "messages" not in payload:
        raise ValueError("The 'body' field must contain a 'messages' field.")
    payload["messages"] = _replace_local_url(payload["messages"])

    mask_input_fields = mask_input_fields.split(",")
    for field in mask_input_fields:
        field = field.strip()
        if field and field in payload:
            payload.pop(field)

    # Remove reserved keys that should not be sent to the API
    default_params = api_config.copy()
    reserved_keys = {"base_url", "api_key", "protocol", "auth_header", "auth_value"}
    for key in reserved_keys:
        default_params.pop(key, None)

    # Merge default params from config into payload if not already present
    # Use the values from payload if conflicting
    for k, v in payload.items():
        if k in default_params:
            logging.warning("The '%s' parameter in config is replaced as '%s'.", k, v)
        default_params[k] = v

    return default_params


def load_api_config(api_config: str | Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads model configuration from a JSON or YAML file.

    Args:
        api_config: Path to the config file, or the config itself.
    Returns:
        A dictionary with model configuration.
    """
    if type(api_config) == dict:
        config = api_config
    else:
        assert type(api_config) == str, "The 'api_config' must be a path to the config file or a Dict."
        file_path = api_config
        if not os.path.exists(file_path):
            logging.error("Model config file not found: %s", file_path)
            raise FileNotFoundError(f"Model config file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                config = json.load(f)
            elif file_path.endswith(('.yml', '.yaml')):
                try:
                    import yaml  # Local import to avoid dependency if not used
                except ImportError:
                    logging.error("PyYAML is not installed. Please install it to use YAML config files, or use json instead.")
                    raise
                config = yaml.safe_load(f)
            else:
                logging.error("Unsupported config file format: %s", file_path)
                raise ValueError("Unsupported config file format. Use .json or .yaml/.yml")

    # Set default protocol if not specified
    if "protocol" not in config:
        config["protocol"] = "openai"

    return config


def load_io_mapping(file_path: str) -> Dict[str, Any]:
    """
    Loads input-output mapping from a JSON or YAML file.
    """
    if not os.path.exists(file_path):
        logging.error("IO mapping file not found: %s", file_path)
        raise FileNotFoundError(f"IO mapping file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        elif file_path.endswith(('.yml', '.yaml')):
            try:
                import yaml
            except ImportError:
                logging.error("PyYAML is not installed. Please install it to use YAML config files, or use json instead.")
                raise Exception("PyYAML is not installed.")
            return yaml.safe_load(f)
        else:
            logging.error("Unsupported IO mapping file format: %s", file_path)
            raise Exception("Unsupported IO mapping file format. Use .json or .yaml/.yml")


def simplify_images(payload):
    res = payload.copy()
    for msg in res["messages"]:
        if "content" in msg:
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if url.startswith("data:image/"):
                        item["image_url"]["url"] = item["image_url"]["url"][-40:]
    return res


async def _execute_with_retry(
    context: RequestContext,
    client_or_factory,  # httpx.Client or httpx.AsyncClient, or a factory function
    max_retries: int,
    max_stream_tokens: int,
    token_counter,
    is_async: bool = False,
    timeout: int = 60,
    no_verify: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Unified retry loop for both sync and async requests.

    Args:
        context: Prepared request context.
        client_or_factory: For sync: httpx.Client or None (creates temp client).
                           For async: httpx.AsyncClient (required).
        max_retries: Maximum number of retries.
        max_stream_tokens: Token limit for streaming.
        token_counter: TokenCounter instance.
        is_async: True for async mode, False for sync mode.
        timeout: Request timeout in seconds (sync only, when client is None).
        no_verify: Disable SSL verification (sync only, when client is None).

    Returns:
        Tuple of (success, result).
    """
    retries = 0
    backoff_factor = 1.0
    last_error_msg = ""
    temp_client = None

    def after_error(error_msg):
        logging.error(error_msg)
        return False, error_msg

    while retries < max_retries:
        try:
            # Create client if not provided (sync only)
            if not is_async and client_or_factory is None:
                verify = not no_verify
                timeout_config = httpx.Timeout(timeout, read=None if context.stream else timeout)
                temp_client = httpx.Client(verify=verify, timeout=timeout_config)
                actual_client = temp_client
            else:
                actual_client = client_or_factory

            # Make the HTTP request
            if is_async:
                response = await actual_client.post(
                    context.endpoint,
                    headers=context.headers,
                    json=context.payload,
                )
            else:
                response = actual_client.post(
                    context.endpoint,
                    headers=context.headers,
                    json=context.payload,
                )
            
            if temp_client and not context.stream:
                temp_client.close()
                temp_client = None

            # --- Handle HTTP Status Codes ---

            # 200 OK: Success!
            if response.status_code == 200:
                if not context.stream:
                    # Non-streaming: just return the JSON body
                    return True, {
                        "custom_id": context.custom_id,
                        "response": response.json(),
                        "query": simplify_images(context.payload)
                    }
                else:
                    # Streaming: process the stream and assemble the full response
                    assembled_response = await context.protocol.process_stream(
                        response,
                        max_stream_tokens=max_stream_tokens,
                        token_counter=token_counter,
                        is_async=is_async
                    )
                    if temp_client:
                        temp_client.close()

                    return True, {
                        "custom_id": context.custom_id,
                        "response": assembled_response,
                        "query": simplify_images(context.payload)
                    }

            # 5xx Server Errors & 429 Rate Limit: Transient errors.
            elif response.status_code in [429] or response.status_code >= 500:
                response_text = response.text
                logging.warning(
                    "Server error %d for %s: %s. Retrying (%d/%d)...",
                    response.status_code, context.custom_id, response_text, retries + 1, max_retries
                )
                last_error_msg = f"Server error {response.status_code}: {response_text}."

            # 4xx Client Errors: Bad request, auth error, etc.
            elif 400 <= response.status_code < 500:
                response_text = response.text
                return after_error(
                    f"Client error {response.status_code} for {context.custom_id}: {response_text}. Skipping."
                )

            # Other unexpected codes
            else:
                response_text = response.text
                logging.warning(
                    "Unexpected status code %d for %s: %s. Retrying (%d/%d)...",
                    response.status_code, context.custom_id, response_text, retries + 1, max_retries
                )
                last_error_msg = f"Unexpected status code {response.status_code}: {response_text}."

        except httpx.TimeoutException:
            logging.warning(
                "Request timed out for %s. Retrying (%d/%d)...",
                context.custom_id, retries + 1, max_retries
            )
            last_error_msg = "Request timed out."

        except httpx.RequestError as e:
            logging.warning(
                "Request exception for %s: %s. Retrying (%d/%d)...",
                context.custom_id, e, retries + 1, max_retries
            )
            last_error_msg = f"Request exception: {e}."

        except Exception as e:
            # Catch any other exceptions (e.g., from stream processing)
            # This is more common in async mode
            logging.warning(
                "Unexpected exception for %s: %s. Retrying (%d/%d)...",
                context.custom_id, e, retries + 1, max_retries
            )
            last_error_msg = f"Unexpected exception: {e}."

        # Exponential backoff
        retries += 1
        if retries < max_retries:
            sleep_time = backoff_factor * (2 ** retries)
            logging.info("Waiting %.2f seconds before retrying %s...", sleep_time, context.custom_id)
            if is_async:
                await asyncio.sleep(sleep_time)
            else:
                time.sleep(sleep_time)

    # Clean up temp client if still open
    if temp_client is not None:
        temp_client.close()

    return after_error(
        f"Max retries ({max_retries}) exceeded for {context.custom_id}. Skipping. Last error message: {last_error_msg}"
    )


def process_single_query(
    task_data: Dict[str, Any],
    api_config: str | Dict[str, Any],
    client: Optional[httpx.Client] = None,
    mask_input_fields: str = "",
    stream: bool = False,
    max_retries: int = 5,
    timeout: int = 60,
    no_verify: bool = False,
    max_stream_tokens: int = 0,
    token_counter=None,
) -> Tuple[bool, Dict[str, Any]] | Tuple[bool, str]:
    """
    Sends a single query to the API and handles retries.

    Args:
        task_data: The JSON object like {"custom_id": ..., "body": { "messages": ..., "temperature": ... }}.
        api_config: Path to the config file, or the config itself.
        client: Optional httpx.Client. If None, creates a temporary client.
                Recommended for batch processing to reuse connections.
        mask_input_fields: Comma-separated fields to ignore from input JSON.
        stream: Whether to enable streaming mode. No need to set 'stream' in api_config.
        max_retries: Maximum number of retries for server errors.
        timeout: Request timeout in seconds.
        no_verify: Whether to disable SSL verification.
        max_stream_tokens: Maximum tokens to allow when assembling streamed responses.
        token_counter: TokenCounter instance for token limits.

    Returns:
        A tuple of (success, result) where success is a boolean indicating success.
        result is a dictionary in the format {"custom_id": ..., "response": ...} on
        success, or a string for error message on failure.
    """

    def after_error(error_msg):
        logging.error(error_msg)
        return False, error_msg

    # Prepare request context (common logic with async)
    context = _prepare_request_context(task_data, api_config, mask_input_fields, stream)
    if isinstance(context, str):
        return after_error(context)

    # Run the unified async retry function in a new event loop
    # This is safe because process_single_query is called from ThreadPoolExecutor threads,
    # each of which doesn't have its own event loop
    return asyncio.run(_execute_with_retry(
        context, client, max_retries, max_stream_tokens, token_counter,
        is_async=False, timeout=timeout, no_verify=no_verify
    ))


async def process_single_query_async(
    task_data: Dict[str, Any],
    api_config: str | Dict[str, Any],
    client: httpx.AsyncClient,  # Required for async
    mask_input_fields: str = "",
    stream: bool = False,
    max_retries: int = 5,
    max_stream_tokens: int = 0,
    token_counter=None,
) -> Tuple[bool, Dict[str, Any]] | Tuple[bool, str]:
    """
    Async version of process_single_query.
    Sends a single query to the API and handles retries using async/await.

    Args:
        task_data: The JSON object like {"custom_id": ..., "body": { "messages": ..., "temperature": ... }}.
        api_config: Path to the config file, or the config itself.
        client: httpx.AsyncClient instance (managed by caller).
        mask_input_fields: Comma-separated fields to ignore from input JSON.
        stream: Whether to enable streaming mode. No need to set 'stream' in api_config.
        max_retries: Maximum number of retries for server errors.
        max_stream_tokens: Maximum tokens to allow when assembling streamed responses.
        token_counter: TokenCounter instance for token limits.

    Returns:
        A tuple of (success, result) where success is a boolean indicating success.
        result is a dictionary in the format {"custom_id": ..., "response": ...} on
        success, or a string for error message on failure.
    """

    def after_error(error_msg):
        logging.error(error_msg)
        return False, error_msg

    # Prepare request context (common logic with sync)
    context = _prepare_request_context(task_data, api_config, mask_input_fields, stream)
    if isinstance(context, str):
        return after_error(context)

    return await _execute_with_retry(
        context, client, max_retries, max_stream_tokens, token_counter,
        is_async=True
    )


# --- Main Processing Logic ---

def load_processed_ids(output_file: str) -> Set[tuple[str, str]]:
    """
    Reads the output file to find which custom_ids have already been processed.
    """
    processed_ids = set()
    if not os.path.exists(output_file):
        return processed_ids

    logging.info("Loading processed IDs from %s...", output_file)
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'custom_id' in data:
                        processed_ids.add((output_file, data['custom_id']))
                except json.JSONDecodeError:
                    logging.warning("Skipping corrupted line in output file: %s", line)
    except Exception as e:
        logging.error("Error reading output file %s: %s", output_file, e)
        # We can still proceed, just might re-process some work.

    logging.info("Found %d processed IDs.", len(processed_ids))
    return processed_ids


def load_tasks(input_file: str, output_file: str, processed_ids: Set[tuple[str, str]], mask_ids: str) -> list:
    """
    Loads all tasks from the input file, skipping those already processed.
    """

    mask_patterns = [p for p in mask_ids.split(',') if p.strip()]
    def match_masked_id(cid: str) -> bool:
        from fnmatch import fnmatch
        for pattern in mask_patterns:
            if fnmatch(cid, pattern):
                return True
        return False

    tasks_to_process = []
    logging.info("Loading tasks from %s...", input_file)
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    task_data = json.loads(line)
                    custom_id = task_data.get('custom_id')
                    if not custom_id:
                        logging.warning(f"Task on {input_file} line {i+1} missing 'custom_id'. Skipping.")
                        continue
                    if match_masked_id(custom_id):
                        logging.info(f"Task with custom_id '{custom_id}' is masked. Skipping.")
                        continue
                    if (output_file, custom_id) in processed_ids:
                        continue  # Skip already processed task
                    tasks_to_process.append((output_file, task_data))
                except json.JSONDecodeError:
                    logging.warning("Skipping corrupted line in input file: %s", line)
    except FileNotFoundError:
        logging.critical("Input file not found: %s", input_file)
        raise
    except Exception as e:
        logging.critical("Error reading input file %s: %s", input_file, e)
        raise

    logging.info(
        "Loaded %d new tasks to process (skipped %d existing).",
        len(tasks_to_process), len(processed_ids)
    )
    return tasks_to_process


class TokenCounter:
    def __init__(self, model_name: str, kind: str = "huggingface"):
        """
        Initializes the TokenCounter with the specified model and tokenizer kind.
        :param model_name: If kind is "huggingface", this is the model name in HuggingFace transformers.
        :param kind: Currently only "huggingface" is supported.
        """
        self.model = model_name
        if kind not in ["huggingface"]:
            raise ValueError("kind must be 'huggingface' now")
        self.kind = kind
        if kind == "huggingface":
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except ImportError:
                logging.error("Transformers library is not installed. Please install it to use TokenCounter with 'huggingface' kind.")
                raise
            # Make the tokenizer usage thread-safe by guarding encodings with a lock.
            self._lock = threading.Lock()
        else:  # TODO: other tokenizers
            raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        if self.kind == "huggingface":
            with self._lock:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
        else:
            raise NotImplementedError


def main():
    """
    Main function to parse arguments and orchestrate the batch processing.
    """
    parser = argparse.ArgumentParser(
        description="Multi-threaded, resumable LLM API batch processor.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the output JSONL file (for results and caching)."
    )
    parser.add_argument(
        "--io",
        type=str,
        default=None,
        help="Path to a json/yaml file in which each key-value pair specifies an input-output pair."
    )
    parser.add_argument(
        "--api_config",
        type=str,
        required=True,
        help="File path to the api config file (json or yaml) including base_url, api_key, model, "
             "and api params, e.g., temperature, top_p, and platform-specific args, e.g., thinking_budget (siliconflow)."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of concurrent processing threads (default: 1)."
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max retries for server errors (default: 5)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)."
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode. The script will still save the *assembled* "
             "full response (including tool calls) to maintain resumability."
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Disable SSL verification. Useful for self-signed certificates."
    )
    parser.add_argument(
        "--mask_input_fields",
        type=str,
        default="",
        help="Ignore these comma-separated fields from the input JSON when building the payload."
    )
    parser.add_argument(
        "--mask_ids",
        type=str,
        default="",
        help="Comma-separated custom_ids to skip processing. Support wildcards (*)."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run only 2 tasks for quick verification."
    )

    parser.add_argument(
        "--max_stream_tokens",
        type=int,
        default=0,
        help="Maximum tokens (content+reasoning) to allow when assembling streamed responses. 0 means no limit."
    )

    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default=None,
        help="HuggingFace model name used by TokenCounter to count tokens when enforcing --max_stream_tokens."
    )

    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async/await with asyncio for better concurrency. "
             "Recommended for high-concurrency workloads (>50 workers). "
             "Requires httpx to be installed."
    )

    args = parser.parse_args()

    # Dispatch to appropriate implementation
    if args.use_async:
        if httpx is None:
            logging.critical(
                "httpx is required for async mode but not installed. "
                "Please install it: pip install 'httpx[socks]'"
            )
            sys.exit(1)

        # Run async implementation
        asyncio.run(main_async_impl(args))
    else:
        # Run existing sync implementation
        main_sync_impl(args)


def main_sync_impl(args):
    """
    Synchronous implementation of main processing loop.
    Uses ThreadPoolExecutor for parallel processing.
    """
    # --- Start Processing ---

    # 0. Handle input-output mappings
    # 1. Find tasks that are already done
    # 2. Load new tasks to process
    io_mapping_file = args.io
    if not io_mapping_file:
        if not args.output_file or not args.input_file:
            logging.error("Must specify either --io or --input_file and --output_file.")
            sys.exit(1)
        processed_ids = load_processed_ids(args.output_file)
        tasks = load_tasks(args.input_file, args.output_file, processed_ids, args.mask_ids)
    else:
        io_mapping = load_io_mapping(io_mapping_file)
        processed_ids = set()
        tasks = []
        for input_file, output_file in io_mapping.items():
            curr_processed = load_processed_ids(output_file)
            tasks.extend(load_tasks(input_file, output_file, curr_processed, args.mask_ids))
            processed_ids.update(curr_processed)

    if not tasks:
        logging.info("No new tasks to process. Exiting.")
        return

    # If max_stream_tokens is requested, require a tokenizer model and enable streaming.
    token_counter = None
    if args.max_stream_tokens and args.max_stream_tokens > 0:
        if not args.tokenizer_model:
            logging.critical("--max_stream_tokens requires --tokenizer_model to be set. Exiting.")
            sys.exit(1)
        # Turn on stream mode automatically when enforcing stream token limits
        args.stream = True
        try:
            token_counter = TokenCounter(model_name=args.tokenizer_model)
        except Exception as e:
            logging.critical("Failed to initialize TokenCounter for '%s': %s", args.tokenizer_model, e)
            sys.exit(1)

    # 3. For testing, limit to 2 tasks
    if args.test:
        tasks = tasks[:2]

    # 4. Process new tasks using a thread pool
    tasks_completed = 0
    total_tasks = len(tasks)

    try:
        with ThreadPoolExecutor(
                max_workers=args.max_workers,
                thread_name_prefix="Processor"
        ) as executor:
            # Submit all tasks to the pool
            future_to_task = {
                executor.submit(
                    process_single_query,
                    task,
                    args.api_config,
                    None,  # client - create temporary client per request
                    args.mask_input_fields,
                    args.stream,
                    args.max_retries,
                    args.timeout,
                    args.no_verify,
                    args.max_stream_tokens,
                    token_counter
                ): (output_file, task)
                for output_file, task in tasks
            }

            logging.info(
                "Submitted %d tasks to %d workers.", total_tasks, args.max_workers
            )

            # Process results as they come in
            # We open the output file in 'append' mode. This is safe for
            # multiple threads *as long as* we only write from the main thread.
            # The `as_completed` iterator allows us to do this.
            for future in as_completed(future_to_task):
                output_file, task = future_to_task[future]
                custom_id = task.get("custom_id", "UNKNOWN")

                try:
                    success, result = future.result()
                    # If result is not None, it was successful
                    if success:
                        # Ensure output directory exists
                        dirpath = os.path.dirname(output_file)
                        if dirpath:
                            os.makedirs(dirpath, exist_ok=True)

                        # Open the target output file per-result (append)
                        with open(output_file, 'a', encoding='utf-8') as f_out:
                            json.dump(result, f_out)
                            f_out.write('\n')
                            f_out.flush()  # Ensure it's written immediately
                        tasks_completed += 1
                except Exception as e:
                    # Handle unexpected errors from the worker function itself
                    logging.error(
                        "Error processing task %s: %s", custom_id, e
                    )
                logging.info(
                    "Progress: %d / %d tasks completed.",
                    tasks_completed, total_tasks
                )
    except Exception as e:
        logging.critical("A fatal error occurred: %s", e)
        # If the file-writing or thread pool fails, we stop.
        # Progress up to this point is saved.
    logging.info("Batch processing complete. Total successful: %d/%d",
                 tasks_completed, total_tasks)


async def main_async_impl(args):
    """
    Async implementation of main processing loop.
    Uses asyncio instead of ThreadPoolExecutor for better concurrency.
    """

    # Load tasks (sync - happens once)
    io_mapping_file = args.io
    if not io_mapping_file:
        if not args.output_file or not args.input_file:
            logging.error("Must specify either --io or --input_file and --output_file.")
            sys.exit(1)
        processed_ids = load_processed_ids(args.output_file)
        tasks = load_tasks(args.input_file, args.output_file, processed_ids, args.mask_ids)
    else:
        io_mapping = load_io_mapping(io_mapping_file)
        processed_ids = set()
        tasks = []
        for input_file, output_file in io_mapping.items():
            curr_processed = load_processed_ids(output_file)
            tasks.extend(load_tasks(input_file, output_file, curr_processed, args.mask_ids))
            processed_ids.update(curr_processed)

    if not tasks:
        logging.info("No new tasks to process. Exiting.")
        return

    # Initialize token counter if needed
    token_counter = None
    if args.max_stream_tokens and args.max_stream_tokens > 0:
        if not args.tokenizer_model:
            logging.critical(
                "--max_stream_tokens requires --tokenizer_model to be set. Exiting."
            )
            sys.exit(1)
        args.stream = True
        try:
            token_counter = TokenCounter(model_name=args.tokenizer_model)
        except Exception as e:
            logging.critical(
                "Failed to initialize TokenCounter for '%s': %s",
                args.tokenizer_model, e
            )
            sys.exit(1)

    # For testing, limit to 2 tasks
    if args.test:
        tasks = tasks[:2]

    tasks_completed = 0
    total_tasks = len(tasks)

    # Create file locks for concurrent writes
    file_locks = defaultdict(asyncio.Lock)

    # Create AsyncClient with connection pooling
    timeout_config = httpx.Timeout(
        timeout=args.timeout,
        read=None if args.stream else args.timeout
    )

    limits = httpx.Limits(
        max_connections=args.max_workers * 2,
        max_keepalive_connections=args.max_workers
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout_config,
            verify=not args.no_verify,
            limits=limits
        ) as client:

            # Create semaphore for bounded concurrency
            semaphore = asyncio.Semaphore(args.max_workers)

            async def bounded_process(output_file, task_data):
                async with semaphore:
                    success, result = await process_single_query_async(
                        task_data,
                        args.api_config,
                        client,
                        args.mask_input_fields,
                        args.stream,
                        args.max_retries,
                        args.max_stream_tokens,
                        token_counter
                    )
                    return success, result, output_file, task_data

            # Create all coroutines
            coroutines = [
                bounded_process(output_file, task_data)
                for output_file, task_data in tasks
            ]

            logging.info(
                "Submitted %d tasks with max %d concurrent workers.",
                total_tasks, args.max_workers
            )

            # Process results as they complete
            for coro in asyncio.as_completed(coroutines):
                try:
                    success, result, output_file, task_data = await coro
                    custom_id = task_data.get("custom_id", "UNKNOWN")

                    if success:
                        # Write result with file lock
                        async with file_locks[output_file]:
                            # Ensure output directory exists
                            dirpath = os.path.dirname(output_file)
                            if dirpath:
                                os.makedirs(dirpath, exist_ok=True)

                            # Write to file (sync, but fast)
                            with open(output_file, 'a', encoding='utf-8') as f_out:
                                json.dump(result, f_out)
                                f_out.write('\n')
                                f_out.flush()

                        tasks_completed += 1

                except Exception as e:
                    custom_id = task_data.get("custom_id", "UNKNOWN") if 'task_data' in locals() else "UNKNOWN"
                    logging.error("Error processing task %s: %s", custom_id, e)

                logging.info(
                    "Progress: %d / %d tasks completed.",
                    tasks_completed, total_tasks
                )

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Shutting down gracefully...")
        # Cancel all pending tasks
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()

    except Exception as e:
        logging.critical("A fatal error occurred: %s", e)

    logging.info(
        "Batch processing complete. Total successful: %d/%d",
        tasks_completed, total_tasks
    )


if __name__ == "__main__":
    main()
