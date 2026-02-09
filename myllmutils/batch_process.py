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
- requests: `pip install requests` (for sync mode)
- httpx: `pip install httpx[socks]` (for async mode)

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Set, Tuple
from collections import defaultdict

# Try to import requests, fail gracefully if not installed.
try:
    import requests
except ImportError:
    print("Error: The 'requests' library is not installed.", file=sys.stderr)
    print("Please install it by running: pip install requests", file=sys.stderr)
    sys.exit(1)

# Try to import httpx for async support
try:
    import httpx
except ImportError:
    httpx = None  # Will error at runtime if --async is used without httpx

# --- Configuration ---

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


# --- API Call Logic ---

def _process_streamed_response(response: requests.Response, max_stream_tokens: int = 0, token_counter=None) -> Dict[str, Any]:
    """
    Processes a streamed API response (SSE) and assembles a single
    JSON object mimicking the non-streaming format.

    *** ASSUMES OPENAI-COMPATIBLE SSE FORMAT ***
    This now handles:
    - Content deltas (e.g., data: {"id": ..., "choices": [{"delta": {"content": ...}}]})
    - Tool call deltas (e.g., data: {"id": ..., "choices": [{"delta": {"tool_calls": [...]}}]})
    - Reasoning deltas (e.g., data: {"id": ..., "choices": [{"delta": {"reasoning_content": ...}}]})
    - A final `data: [DONE]`

    Modify this function if your API's stream format is different.
    """
    full_content = ""
    full_reasoning = ""  # For separate reasoning/log streams
    tool_call_chunks = {}  # Stores partial tool calls by their index
    model_info = ""
    response_id = ""
    finish_reason = "stop"  # Default

    # `token_counter` may be provided by the caller (recommended) to avoid
    # creating tokenizer instances inside worker threads. If None, streaming
    # token limits are disabled.

    try:
        for line_bytes in response.iter_lines():
            if line_bytes:
                line = line_bytes.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[len('data: '):]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        if not response_id:  # Capture first ID/model
                            response_id = chunk.get('id', '')
                            model_info = chunk.get('model', '')

                        choices = chunk.get('choices', [])
                        if not choices:
                            continue

                        delta = choices[0].get('delta', {})
                        if not delta:
                            continue

                        # --- 1. Process Content Deltas ---
                        if delta.get('content'):
                            full_content += delta['content']

                        # --- 2. Process Reasoning Deltas ---
                        if delta.get('reasoning_content'):
                            full_reasoning += delta['reasoning_content']
                        elif delta.get('reasoning'):
                            full_reasoning += delta['reasoning']

                        # --- Enforce max stream token limit (content + reasoning) ---
                        if token_counter is not None and max_stream_tokens and max_stream_tokens > 0:
                            try:
                                combined_text = (full_content or "") + ("\n" + full_reasoning if full_reasoning else "")
                                total_tokens = token_counter.count_tokens(combined_text)
                                if total_tokens > max_stream_tokens:
                                    finish_reason = 'length'  # Indicate cut-off due to token limit
                                    try:
                                        response.close()
                                    except Exception:
                                        pass
                                    break
                            except Exception:
                                # If counting fails for any reason, continue without enforcing
                                logging.warning("Token counting failed during streaming; continuing without enforcement.")

                        # --- 3. Process Tool Call Deltas ---
                        if delta.get('tool_calls'):
                            for tool_delta in delta['tool_calls']:
                                index = tool_delta.get('index')
                                if index is None:
                                    continue  # Invalid tool delta

                                # Initialize storage for this tool call index
                                if index not in tool_call_chunks:
                                    tool_call_chunks[index] = {
                                        "id": "",
                                        "type": "function",  # Default type
                                        "function": {"name": "", "arguments": ""}
                                    }

                                # Merge data
                                if tool_delta.get('id'):
                                    tool_call_chunks[index]['id'] = tool_delta['id']
                                if tool_delta.get('type'):
                                    tool_call_chunks[index]['type'] = tool_delta['type']

                                if 'function' in tool_delta:
                                    if tool_delta['function'].get('name'):
                                        tool_call_chunks[index]['function']['name'] = tool_delta['function']['name']
                                    if tool_delta['function'].get('arguments'):
                                        tool_call_chunks[index]['function']['arguments'] += tool_delta['function'][
                                            'arguments']

                        # --- 4. Capture Finish Reason ---
                        # Capture finish reason from the *last* chunk that has one
                        if choices[0].get('finish_reason'):
                            finish_reason = choices[0].get('finish_reason')

                    except json.JSONDecodeError:
                        logging.warning("Failed to decode JSON chunk: %s", data_str)
                        continue

    except Exception:
        logging.exception("Error while processing stream")
        raise

    # --- Assemble the final response object ---

    # Convert dict of chunks to a final list
    assembled_tool_calls = []
    if tool_call_chunks:
        # Sort by index to ensure correct order
        sorted_indices = sorted(tool_call_chunks.keys())
        assembled_tool_calls = [tool_call_chunks[i] for i in sorted_indices]

    # Build the final message object
    message = {"role": "assistant"}
    if full_content:
        message["content"] = full_content
    else:
        # Per OpenAI spec, content is null if tool_calls are present
        message["content"] = None if assembled_tool_calls else ""

    if assembled_tool_calls:
        message["tool_calls"] = assembled_tool_calls

    if full_reasoning:
        # Add the assembled reasoning content as a separate field
        message["reasoning_content"] = full_reasoning

    # This structure is designed to mimic the OpenAI non-streaming
    # chat completions response, with the addition of `reasoning_content`.
    return {
        "id": response_id,
        "model": model_info,
        "object": "chat.completion",  # Mock object type
        "created": int(time.time()),  # Mock timestamp
        "choices": [
            {
                "index": 0,
                "message": message,  # Use the assembled message
                "finish_reason": finish_reason
            }
        ],
        "usage": {  # Note: Usage data is often incomplete/missing in streams
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        }
    }


async def _process_streamed_response_async(response, max_stream_tokens: int = 0, token_counter=None) -> Dict[str, Any]:
    """
    Async version of _process_streamed_response.
    Processes a streamed API response (SSE) and assembles a single JSON object.

    *** ASSUMES OPENAI-COMPATIBLE SSE FORMAT ***
    This now handles:
    - Content deltas (e.g., data: {"id": ..., "choices": [{"delta": {"content": ...}}]})
    - Tool call deltas (e.g., data: {"id": ..., "choices": [{"delta": {"tool_calls": [...]}}]})
    - Reasoning deltas (e.g., data: {"id": ..., "choices": [{"delta": {"reasoning_content": ...}}]})
    - A final `data: [DONE]`

    Modify this function if your API's stream format is different.
    """
    full_content = ""
    full_reasoning = ""  # For separate reasoning/log streams
    tool_call_chunks = {}  # Stores partial tool calls by their index
    model_info = ""
    response_id = ""
    finish_reason = "stop"  # Default

    # `token_counter` may be provided by the caller (recommended) to avoid
    # creating tokenizer instances inside worker threads. If None, streaming
    # token limits are disabled.

    try:
        async for line in response.aiter_lines():
            if line and line.startswith('data: '):
                data_str = line[len('data: '):]
                if data_str.strip() == '[DONE]':
                    break
                try:
                    chunk = json.loads(data_str)
                    if not response_id:  # Capture first ID/model
                        response_id = chunk.get('id', '')
                        model_info = chunk.get('model', '')

                    choices = chunk.get('choices', [])
                    if not choices:
                        continue

                    delta = choices[0].get('delta', {})
                    if not delta:
                        continue

                    # --- 1. Process Content Deltas ---
                    if delta.get('content'):
                        full_content += delta['content']

                    # --- 2. Process Reasoning Deltas ---
                    if delta.get('reasoning_content'):
                        full_reasoning += delta['reasoning_content']
                    elif delta.get('reasoning'):
                        full_reasoning += delta['reasoning']

                    # --- Enforce max stream token limit (content + reasoning) ---
                    if token_counter is not None and max_stream_tokens and max_stream_tokens > 0:
                        try:
                            combined_text = (full_content or "") + ("\n" + full_reasoning if full_reasoning else "")
                            # Use run_in_executor to avoid blocking the event loop
                            loop = asyncio.get_event_loop()
                            total_tokens = await loop.run_in_executor(
                                None,
                                token_counter.count_tokens,
                                combined_text
                            )
                            if total_tokens > max_stream_tokens:
                                finish_reason = 'length'  # Indicate cut-off due to token limit
                                try:
                                    await response.aclose()
                                except Exception:
                                    pass
                                break
                        except Exception:
                            # If counting fails for any reason, continue without enforcing
                            logging.warning("Token counting failed during streaming; continuing without enforcement.")

                    # --- 3. Process Tool Call Deltas ---
                    if delta.get('tool_calls'):
                        for tool_delta in delta['tool_calls']:
                            index = tool_delta.get('index')
                            if index is None:
                                continue  # Invalid tool delta

                            # Initialize storage for this tool call index
                            if index not in tool_call_chunks:
                                tool_call_chunks[index] = {
                                    "id": "",
                                    "type": "function",  # Default type
                                    "function": {"name": "", "arguments": ""}
                                }

                            # Merge data
                            if tool_delta.get('id'):
                                tool_call_chunks[index]['id'] = tool_delta['id']
                            if tool_delta.get('type'):
                                tool_call_chunks[index]['type'] = tool_delta['type']

                            if 'function' in tool_delta:
                                if tool_delta['function'].get('name'):
                                    tool_call_chunks[index]['function']['name'] = tool_delta['function']['name']
                                if tool_delta['function'].get('arguments'):
                                    tool_call_chunks[index]['function']['arguments'] += tool_delta['function'][
                                        'arguments']

                    # --- 4. Capture Finish Reason ---
                    # Capture finish reason from the *last* chunk that has one
                    if choices[0].get('finish_reason'):
                        finish_reason = choices[0].get('finish_reason')

                except json.JSONDecodeError:
                    logging.warning("Failed to decode JSON chunk: %s", data_str)
                    continue

    except Exception:
        logging.exception("Error while processing stream")
        raise

    # --- Assemble the final response object ---

    # Convert dict of chunks to a final list
    assembled_tool_calls = []
    if tool_call_chunks:
        # Sort by index to ensure correct order
        sorted_indices = sorted(tool_call_chunks.keys())
        assembled_tool_calls = [tool_call_chunks[i] for i in sorted_indices]

    # Build the final message object
    message = {"role": "assistant"}
    if full_content:
        message["content"] = full_content
    else:
        # Per OpenAI spec, content is null if tool_calls are present
        message["content"] = None if assembled_tool_calls else ""

    if assembled_tool_calls:
        message["tool_calls"] = assembled_tool_calls

    if full_reasoning:
        # Add the assembled reasoning content as a separate field
        message["reasoning_content"] = full_reasoning

    # This structure is designed to mimic the OpenAI non-streaming
    # chat completions response, with the addition of `reasoning_content`.
    return {
        "id": response_id,
        "model": model_info,
        "object": "chat.completion",  # Mock object type
        "created": int(time.time()),  # Mock timestamp
        "choices": [
            {
                "index": 0,
                "message": message,  # Use the assembled message
                "finish_reason": finish_reason
            }
        ],
        "usage": {  # Note: Usage data is often incomplete/missing in streams
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        }
    }


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
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if not url.startswith("http") and not url.startswith("data"):
                        # Replace local file path with a placeholder URL
                        base64_encoded = encode_pil_image(Image.open(url))
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

    default_params = api_config.copy()
    default_params.pop("base_url", None)
    default_params.pop("api_key", None)

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
        return api_config

    assert type(api_config) == str, "The 'api_config' must be a path to the config file or a Dict."
    file_path = api_config
    if not os.path.exists(file_path):
        logging.error("Model config file not found: %s", file_path)
        raise FileNotFoundError(f"Model config file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        elif file_path.endswith(('.yml', '.yaml')):
            try:
                import yaml  # Local import to avoid dependency if not used
            except ImportError:
                logging.error("PyYAML is not installed. Please install it to use YAML config files, or use json instead.")
                raise
            return yaml.safe_load(f)
        else:
            logging.error("Unsupported config file format: %s", file_path)
            raise ValueError("Unsupported config file format. Use .json or .yaml/.yml")


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


def process_single_query(
    task_data: Dict[str, Any],
    api_config: str | Dict[str, Any],
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
        mask_input_fields: Comma-separated fields to ignore from input JSON.
        stream: Whether to enable streaming mode. No need to set 'stream' in api_config.
        max_retries: Maximum number of retries for server errors.
        timeout: Request timeout in seconds.
        no_verify: Whether to disable SSL verification.

    Returns:
        A tuple of (success, result) where success is a boolean indicating success.
        result is a dictionary in the format {"custom_id": ..., "response": ...} on
        success, or a string for error message on failure.
    """

    def after_error(error_msg):
        logging.error(error_msg)
        return False, error_msg

    custom_id = task_data.get("custom_id")
    if not custom_id:
        return after_error(f"Task data missing 'custom_id'. Skipping: {task_data}")

    api_config = load_api_config(api_config)
    payload = build_payload(task_data, api_config, mask_input_fields)
    if 'model' not in payload:
        return after_error(f"No 'model' specified in api_config or task_data for task {custom_id}.")
    if stream:
        payload['stream'] = True

    base_url = api_config.get('base_url', None)
    api_key = api_config.get('api_key', None)
    if not base_url:
        return after_error(f"No 'base_url' specified in the config.")
    if not api_key:
        logging.warning("No 'api_key' specified in the config.")
        api_key = "NONE"

    if api_key.startswith("env::"):
        env_key = api_key[len("env::"):]
        api_key = os.getenv(env_key)
        if not api_key:
            return after_error(f"The environment {env_key} after 'env::' is not set.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    retries = 0
    backoff_factor = 1.0  # Initial backoff time in seconds

    last_error_msg = ""
    while retries < max_retries:
        try:
            verify = not no_verify
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
                verify=verify,
                stream=stream  # Enable streaming for the requests call
            )

            # --- Handle HTTP Status Codes ---

            # 200 OK: Success!
            if response.status_code == 200:
                if not stream:
                    # Non-streaming: just return the JSON body
                    return True, {"custom_id": custom_id, "response": response.json(), "query": simplify_images(payload)}
                else:
                    # Streaming: process the stream and assemble the full response
                    assembled_response = _process_streamed_response(response, max_stream_tokens=max_stream_tokens, token_counter=token_counter)
                    return True, {"custom_id": custom_id, "response": assembled_response, "query": simplify_images(payload)}

            # 5xx Server Errors & 429 Rate Limit: Transient errors.
            # These are worth retrying.
            elif response.status_code in [429] or response.status_code >= 500:
                # Read response text for logging
                if stream:
                    try:
                        response_text = response.text
                    except Exception:
                        response_text = "[Could not read response text]"
                else:
                    response_text = response.text

                logging.warning(
                    "Server error %d for %s: %s. Retrying (%d/%d)...",
                    response.status_code, custom_id, response_text, retries + 1, max_retries
                )
                last_error_msg = f"Server error {response.status_code}: {response_text}."

            # 4xx Client Errors: Bad request, auth error, etc.
            # These are unlikely to succeed on retry.
            elif 400 <= response.status_code < 500:
                # Read response text for logging, even if streaming was requested
                if stream:
                    # If we error'd early, we might be able to read text
                    try:
                        response_text = response.text
                    except Exception:
                        response_text = "[Could not read response text]"
                else:
                    response_text = response.text

                # Do not retry
                return after_error(f"Client error {response.status_code} for {custom_id}: {response_text}. Skipping.")

            # Other unexpected codes
            else:
                if stream:
                    try:
                        response_text = response.text
                    except Exception:
                        response_text = "[Could not read response text]"
                else:
                    response_text = response.text

                logging.warning(
                    "Unexpected status code %d for %s: %s. Retrying (%d/%d)...",
                    response.status_code, custom_id, response_text, retries + 1, max_retries
                )
                last_error_msg = f"Unexpected status code {response.status_code}: {response_text}."

        except requests.exceptions.Timeout:
            logging.warning(
                "Request timed out for %s. Retrying (%d/%d)...",
                custom_id, retries + 1, max_retries
            )
            last_error_msg = f"Request timed out."

        except requests.exceptions.RequestException as e:
            # Catch other requests-related errors (e.g., connection error)
            logging.warning(
                "Request exception for %s: %s. Retrying (%d/%d)...",
                custom_id, e, retries + 1, max_retries
            )
            last_error_msg = f"Request exception: {e}."

        # Exponential backoff
        retries += 1
        if retries < max_retries:
            sleep_time = backoff_factor * (2 ** retries)
            logging.info("Waiting %.2f seconds before retrying %s...", sleep_time, custom_id)
            time.sleep(sleep_time)

    return after_error(f"Max retries ({max_retries}) exceeded for {custom_id}. Skipping. Last error message: {last_error_msg}")


async def process_single_query_async(
    task_data: Dict[str, Any],
    api_config: str | Dict[str, Any],
    client,  # httpx.AsyncClient
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
        timeout: Request timeout in seconds (not used - client has timeout config).
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

    custom_id = task_data.get("custom_id")
    if not custom_id:
        return after_error(f"Task data missing 'custom_id'. Skipping: {task_data}")

    # Reuse sync functions for config/payload (they're fast)
    api_config = load_api_config(api_config)
    payload = build_payload(task_data, api_config, mask_input_fields)

    if 'model' not in payload:
        return after_error(f"No 'model' specified in api_config or task_data for task {custom_id}.")

    if stream:
        payload['stream'] = True

    base_url = api_config.get('base_url', None)
    api_key = api_config.get('api_key', None)

    if not base_url:
        return after_error("No 'base_url' specified in the config.")
    if not api_key:
        logging.warning("No 'api_key' specified in the config.")
        api_key = "NONE"

    if api_key.startswith("env::"):
        env_key = api_key[len("env::"):]
        api_key = os.getenv(env_key)
        if not api_key:
            return after_error(f"The environment {env_key} after 'env::' is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    retries = 0
    backoff_factor = 1.0
    last_error_msg = ""

    while retries < max_retries:
        try:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )

            # --- Handle HTTP Status Codes (same logic as sync) ---

            # 200 OK: Success!
            if response.status_code == 200:
                if not stream:
                    # Non-streaming: just return the JSON body
                    return True, {
                        "custom_id": custom_id,
                        "response": response.json(),
                        "query": simplify_images(payload)
                    }
                else:
                    # Streaming: process the stream and assemble the full response
                    assembled_response = await _process_streamed_response_async(
                        response,
                        max_stream_tokens=max_stream_tokens,
                        token_counter=token_counter
                    )
                    return True, {
                        "custom_id": custom_id,
                        "response": assembled_response,
                        "query": simplify_images(payload)
                    }

            # 5xx Server Errors & 429 Rate Limit: Transient errors. Retry.
            elif response.status_code in [429] or response.status_code >= 500:
                response_text = response.text
                logging.warning(
                    "Server error %d for %s: %s. Retrying (%d/%d)...",
                    response.status_code, custom_id, response_text, retries + 1, max_retries
                )
                last_error_msg = f"Server error {response.status_code}: {response_text}."

            # 4xx Client Errors: Bad request, auth error, etc. Don't retry.
            elif 400 <= response.status_code < 500:
                response_text = response.text
                return after_error(
                    f"Client error {response.status_code} for {custom_id}: {response_text}. Skipping."
                )

            # Other unexpected codes
            else:
                response_text = response.text
                logging.warning(
                    "Unexpected status code %d for %s: %s. Retrying (%d/%d)...",
                    response.status_code, custom_id, response_text, retries + 1, max_retries
                )
                last_error_msg = f"Unexpected status code {response.status_code}: {response_text}."

        except httpx.TimeoutException:
            logging.warning(
                "Request timed out for %s. Retrying (%d/%d)...",
                custom_id, retries + 1, max_retries
            )
            last_error_msg = "Request timed out."

        except httpx.RequestError as e:
            logging.warning(
                "Request exception for %s: %s. Retrying (%d/%d)...",
                custom_id, e, retries + 1, max_retries
            )
            last_error_msg = f"Request exception: {e}."

        except Exception as e:
            # Catch any other exceptions (e.g., from httpx or stream processing)
            logging.warning(
                "Unexpected exception for %s: %s. Retrying (%d/%d)...",
                custom_id, e, retries + 1, max_retries
            )
            last_error_msg = f"Unexpected exception: {e}."

        # Exponential backoff
        retries += 1
        if retries < max_retries:
            sleep_time = backoff_factor * (2 ** retries)
            logging.info("Waiting %.2f seconds before retrying %s...", sleep_time, custom_id)
            await asyncio.sleep(sleep_time)

    return after_error(f"Max retries ({max_retries}) exceeded for {custom_id}. Skipping. Last error message: {last_error_msg}")


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

    mask_patterns = mask_ids.split(',')
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
        sys.exit(1)
    except Exception as e:
        logging.critical("Error reading input file %s: %s", input_file, e)
        sys.exit(1)

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
                        args.timeout,
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
