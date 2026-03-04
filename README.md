# MyLLMUtils
A lightweight library to ease LLM use.

Managed with `uv`.

## Installation

```bash
# Basic installation
uv pip install -e .

# With GPU support (for token counting)
uv pip install -e ".[gpu]"
```

## API Configuration

The `api_config` file (YAML or JSON) specifies how to connect to an LLM API. It supports:

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `base_url` | string | API endpoint base URL |
| `api_key` | string | API key. Use `env::VAR_NAME` to read from environment variable |
| `model` | string | Model name (can be overridden in individual requests) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `protocol` | string | Protocol type: `openai` (default) or `gemini` |
| `auth_header` | string | Custom authorization header key (overrides protocol default) |
| `auth_value` | string | Custom authorization header value (supports `{api_key}` placeholder) |
| `temperature` | number | Sampling temperature (0.0 - 2.0) |
| `max_tokens` | number | Maximum tokens in response |
| `top_p` | number | Nucleus sampling threshold |
| `*` | any | Additional parameters passed directly to the API |

**Note:** The following keys are reserved for configuration and will **NOT** be sent to the API as part of the request payload: `base_url`, `api_key`, `protocol`, `auth_header`, `auth_value`. All other fields in the config will be included in the API request.

### Authorization

By default, each protocol uses its standard authorization scheme:

| Protocol | Default Header | Default Value Format |
|----------|---------------|---------------------|
| `openai` | `Authorization` | `Bearer {api_key}` |
| `gemini` | `x-goog-api-key` | `{api_key}` |

You can override these using `auth_header` and `auth_value` in your config:

```yaml
# Example: Custom header with Bearer token
base_url: https://custom-api.example.com
api_key: env::MY_API_KEY
model: my-model
auth_header: X-API-Key
auth_value: Bearer {api_key}  # {api_key} gets replaced with the actual key

# Example: No Bearer prefix
base_url: https://another-api.example.com
api_key: env::MY_API_KEY
model: my-model
auth_header: Authorization
auth_value: {api_key}  # Direct key without "Bearer"
```

### Example Configs

**OpenAI-compatible:**
```yaml
base_url: https://api.openai.com/v1
api_key: env::OPENAI_API_KEY
model: gpt-4
temperature: 0.7
```

**Gemini:**
```yaml
base_url: https://generativelanguage.googleapis.com/v1beta
api_key: env::GOOGLE_API_KEY
model: gemini-2.0-flash-exp
protocol: gemini
```

**DeepSeek:**
```yaml
base_url: https://api.deepseek.com
api_key: env::DS_API_KEY
model: deepseek-chat
max_tokens: 100
```

## Batch Processing

The primary use case is batch processing queries from JSONL files:

```bash
python -m myllmutils.batch_process \
  --input_file queries.jsonl \
  --output_file results.jsonl \
  --api_config model.yaml \
  --max_workers 16 \
  --stream
```

**Input JSONL format:**
```jsonl
{"custom_id": "req-1", "body": {"messages": [{"role": "user", "content": "Hello"}]}}
{"custom_id": "req-2", "body": {"messages": [{"role": "user", "content": "How are you?"}]}}
```

**Output JSONL format:**
```jsonl
{"custom_id": "req-1", "response": {...}, "query": {...}}
{"custom_id": "req-2", "response": {...}, "query": {...}}
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--input_file` | Input JSONL file path |
| `--output_file` | Output JSONL file path |
| `--io` | JSON/YAML file mapping input/output pairs |
| `--api_config` | API config file (required) |
| `--max_workers` | Concurrent threads (default: 1) |
| `--max_retries` | Max retries for server errors (default: 5) |
| `--timeout` | Request timeout in seconds (default: 60) |
| `--stream` | Enable streaming mode |
| `--no_verify` | Disable SSL verification |
| `--async` | Use async/await for better concurrency |
| `--max_stream_tokens` | Token limit for streamed responses |
| `--tokenizer_model` | HuggingFace model for token counting |

### IO Mapping

Process multiple input/output pairs in one run:

```bash
python -m myllmutils.batch_process \
  --io mapping.json \
  --api_config model.yaml \
  --max_workers 8
```

**mapping.json:**
```json
{
  "input1.jsonl": "output1.jsonl",
  "input2.jsonl": "output2.jsonl"
}
```

## Determinism Testing

Test whether an LLM API produces deterministic outputs by sending the same prompts multiple times at various concurrency levels:

```bash
python -m myllmutils.test_determinism \
  --dataset tatsu-lab/alpaca_eval \
  --text_column instruction \
  --api_config model.yaml \
  --batch_sizes 1,2,4,8 \
  --num_repetitions 8 \
  --max_prompts 50
```

Each `batch_size` value controls the concurrency level (max_workers per round). The `--num_repetitions` parameter controls how many total responses are collected per prompt per batch setting — the same count for every batch_size, ensuring fair comparison. Requests are dispatched in rounds of `batch_size` concurrent workers.

### Metrics

- **Default (fast):** Exact match rate, longest common prefix (LCP) length and ratio — O(n)
- **Optional (`--edit_distance`):** Levenshtein edit distance, SequenceMatcher similarity — O(n*m), use for short responses

### Caching

Use `--cache_file` to save responses to a JSONL file. On restart, cached responses are reused automatically:

```bash
python -m myllmutils.test_determinism \
  --dataset prompts.json \
  --text_column prompt \
  --api_config model.yaml \
  --batch_sizes 1,4 \
  --num_repetitions 4 \
  --cache_file cache.jsonl \
  --output_file report.json
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--dataset` | Local file path or HuggingFace dataset identifier (required) |
| `--text_column` | Column name containing prompt text (required) |
| `--api_config` | API config file (required) |
| `--batch_sizes` | Comma-separated concurrency levels, e.g. `1,2,4,8` (required) |
| `--num_repetitions` | Responses per prompt per batch setting (required) |
| `--dataset_split` | Dataset split (default: `train`) |
| `--dataset_config` | HuggingFace dataset config/subset name |
| `--max_prompts` | Limit number of prompts (0 = all) |
| `--stream` | Enable streaming mode |
| `--cache_file` | JSONL cache file for response resumability |
| `--output_file` | Save detailed JSON report |
| `--edit_distance` | Enable expensive Levenshtein/SequenceMatcher metrics |
| `--max_retries` | Max retries for server errors (default: 5) |
| `--timeout` | Request timeout in seconds (default: 60) |
| `--no_verify` | Disable SSL verification |
