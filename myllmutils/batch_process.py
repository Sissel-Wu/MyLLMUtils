"""
A standalone, multi-threaded, resumable batch processor for LLM APIs.

This script reads queries from one or more JSONL file, sends them to an LLM API
concurrently, and saves the results to the corresponding output JSONL files.

It supports:
- Resuming progress (skips queries already in the output files)
- Multi-threading for concurrent API calls
- Error handling with exponential backoff for server-side errors
- Caching results as they are completed
- Assembling streamed responses (including text, tool calls, and reasoning)

Requirements:
- requests: `pip install requests`

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

Usage:
python batch_processor.py \
    --input_file queries.jsonl \
    --output_file results.jsonl \
    --api_config model_config.json \
    --api_key "YOUR_API_KEY" \
    --max_workers 16 \
    --stream

OR

python batch_processor.py \
    --io mapping.json \
    --api_config model_config.yaml \
    --api_key "YOUR_API_KEY" \
    --max_workers 16 \
    --stream
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Set

# Try to import requests, fail gracefully if not installed.
try:
    import requests
except ImportError:
    print("Error: The 'requests' library is not installed.", file=sys.stderr)
    print("Please install it by running: pip install requests", file=sys.stderr)
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


# --- API Call Logic ---

def _process_streamed_response(response: requests.Response) -> Dict[str, Any]:
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
        raise Exception("The json does not contain a body.")

    assert isinstance(payload, dict), "The 'body' field must be a JSON object."
    assert "messages" in payload, "The 'body' must contain a 'messages' field."
    payload["messages"] = _replace_local_url(payload["messages"])

    mask_input_fields = mask_input_fields.split(",")
    for field in mask_input_fields:
        field = field.strip()
        if field and field in payload:
            payload.pop(field)

    default_params = api_config.copy()
    default_params.pop("base_url")
    default_params.pop("api_key")

    # Merge default params from config into payload if not already present
    # Use the values from payload if conflicting
    for k, v in payload.items():
        if k in default_params:
            logging.warning("The '%s' parameter in config is replaced as '%s'.", k, v)
        default_params[k] = v

    return default_params


def load_api_config(file_path: str) -> Dict[str, Any]:
    """
    Loads model configuration from a JSON or YAML file.

    Args:
        file_path: Path to the config file.
    Returns:
        A dictionary with model configuration.
    """
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
                raise Exception("PyYAML is not installed.")
            return yaml.safe_load(f)
        else:
            logging.error("Unsupported config file format: %s", file_path)
            raise Exception("Unsupported config file format. Use .json or .yaml/.yml")


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
                        item["image_url"]["url"] = item["image_url"]["url"][:40]
    return res


def process_single_query(
        task_data: Dict[str, Any],
        args: argparse.Namespace
) -> Optional[Dict[str, Any]]:
    """
    Sends a single query to the API and handles retries.

    Args:
        task_data: The JSON object for a single query from the input file.
        args: The arguments passed from command line.

    Returns:
        A dictionary in the format {"custom_id": ..., "response": ...} on
        success, or None on failure after retries.
    """
    custom_id = task_data.get("custom_id")
    if not custom_id:
        logging.error("Task data missing 'custom_id'. Skipping: %s", task_data)
        return None

    api_config = load_api_config(args.api_config) if args.api_config else None
    payload = build_payload(task_data, api_config, args.mask_input_fields)
    if 'model' not in payload:
        logging.error(
            "No model specified for task %s. Provide a model in the input data or via config.",
            custom_id
        )
        return None
    if args.stream:
        payload['stream'] = True

    base_url = api_config.get('base_url')
    api_key = api_config.get('api_key')
    if api_key.startswith("env::"):
        api_key = os.getenv(api_key[len("env::"):])
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    retries = 0
    backoff_factor = 1.0  # Initial backoff time in seconds

    while retries < args.max_retries:
        try:
            verify = not args.no_verify
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=args.timeout,
                verify=verify,
                stream=args.stream  # Enable streaming for the requests call
            )

            # --- Handle HTTP Status Codes ---

            # 200 OK: Success!
            if response.status_code == 200:
                if not args.stream:
                    # Non-streaming: just return the JSON body
                    return {"custom_id": custom_id, "response": response.json(), "query": simplify_images(payload)}
                else:
                    # Streaming: process the stream and assemble the full response
                    assembled_response = _process_streamed_response(response)
                    return {"custom_id": custom_id, "response": assembled_response, "query": simplify_images(payload)}

            # 4xx Client Errors: Bad request, auth error, etc.
            # These are unlikely to succeed on retry.
            elif 400 <= response.status_code < 500:
                # Read response text for logging, even if streaming was requested
                response_text = ""
                if args.stream:
                    # If we error'd early, we might be able to read text
                    try:
                        response_text = response.text
                    except Exception:
                        response_text = "[Could not read response text]"
                else:
                    response_text = response.text

                logging.error(
                    "Client error %d for %s: %s. Skipping.",
                    response.status_code, custom_id, response_text
                )
                return None  # Do not retry

            # 5xx Server Errors & 429 Rate Limit: Transient errors.
            # These are worth retrying.
            elif response.status_code in [429] or response.status_code >= 500:
                # Read response text for logging
                response_text = ""
                if args.stream:
                    try:
                        response_text = response.text
                    except Exception:
                        response_text = "[Could not read response text]"
                else:
                    response_text = response.text

                logging.warning(
                    "Server error %d for %s: %s. Retrying (%d/%d)...",
                    response.status_code, custom_id, response_text, retries + 1, args.max_retries
                )

            # Other unexpected codes
            else:
                response_text = ""
                if args.stream:
                    try:
                        response_text = response.text
                    except Exception:
                        response_text = "[Could not read response text]"
                else:
                    response_text = response.text

                logging.warning(
                    "Unexpected status code %d for %s: %s. Retrying (%d/%d)...",
                    response.status_code, custom_id, response_text, retries + 1, args.max_retries
                )

        except requests.exceptions.Timeout:
            logging.warning(
                "Request timed out for %s. Retrying (%d/%d)...",
                custom_id, retries + 1, args.max_retries
            )

        except requests.exceptions.RequestException as e:
            # Catch other requests-related errors (e.g., connection error)
            logging.warning(
                "Request exception for %s: %s. Retrying (%d/%d)...",
                custom_id, e, retries + 1, args.max_retries
            )

        # Exponential backoff
        retries += 1
        if retries < args.max_retries:
            sleep_time = backoff_factor * (2 ** retries)
            logging.info("Waiting %.2f seconds before retrying %s...", sleep_time, custom_id)
            time.sleep(sleep_time)

    logging.error(
        "Max retries (%d) exceeded for %s. Skipping.", args.max_retries, custom_id
    )
    return None


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
        "--mask-ids",
        type=str,
        default="",
        help="Comma-separated custom_ids to skip processing. Support wildcards (*)."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run only 2 tasks for quick verification."
    )

    args = parser.parse_args()

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
                    args
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
                    result = future.result()
                    # If result is not None, it was successful
                    if result:
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


if __name__ == "__main__":
    main()
