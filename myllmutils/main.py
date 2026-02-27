import myllmutils
from myllmutils import LLMService, ZeroShotMessages, FewShotMessages, ZeroShotVLMessages
from myllmutils import prepare_offline_inference, volcano_template
from myllmutils.batch_process import process_single_query, process_single_query_async, TokenCounter
import asyncio
import httpx
import os
import yaml


def example_zeroshot():
    # A zero-shot chat with an LLM
    chat_llm = LLMService(output_dir="llm_output")
    messages = ZeroShotMessages(user_query="How are you doing?",
                                system_message="You are a cool girl and talk in that vibe.")
    print(chat_llm.chat_complete(messages,
                                 model="gpt-5-nano",
                                 temperature=1.0,
                                 return_str=True,
                                 title="say_hi"))

def example_disable_ssl():
    # A zero-shot chat with an LLM
    chat_llm = LLMService(output_dir="llm_output", disable_ssl_verify=True)
    print(chat_llm.simple_chat("How are you doing?",
                               system_message="You are a cool girl and talk in that vibe.",
                               model="gpt-5-nano",
                               return_str=True,
                               title="say_hi_no_ssl"))

def example_fewshot():
    # A few-shot chat with an LLM
    chat_llm = LLMService(output_dir="llm_output")
    messages = FewShotMessages(system_message="Answer math questions.",
                               shots=[("What is 1+1?", "1+1=2"), ("What is 2+2?", "2+2=4")],
                               user_query="What is 2+4?")
    print(chat_llm.chat_complete(messages,
                                 model="gpt-5-nano",
                                 temperature=1.0,
                                 return_str=True,
                                 title="math"))

def example_logprobs():
    chat_llm = LLMService(output_dir="llm_output")
    messages = ZeroShotMessages(user_query="There are three balls (red, blue, blue) in a black box. Pick a random ball from the box, what is the color? Answer randomly.")
    print(chat_llm.chat_complete(messages,
                                 model="gpt-5-nano",
                                 temperature=1.0,
                                 return_str=True,
                                 title="random_color",
                                 logprobs=True,
                                 top_logprobs=10))

def example_sampling():
    chat_llm = LLMService(output_dir="llm_output")
    messages = ZeroShotMessages(user_query="Pick a random substring in \"woijroi23oijovjasoijweijrowjieorjowiejr\".")
    print(chat_llm.chat_complete(messages,
                                 model="gpt-4o-mini",
                                 temperature=1.0,
                                 return_str=True,
                                 title="random_substr",
                                 use_cache=True,
                                 n=5,
                                 n_limit_per_query=2))
    print(chat_llm.chat_complete_greedy(messages,
                                        model="gpt-4o-mini",
                                        temperature=0.01,
                                        return_str=True,
                                        title="random_substr_greedy",
                                        use_cache=True,
                                        n=5))

def example_deepseek():
    chat_llm = LLMService("https://api.deepseek.com", os.environ.get("DS_API_KEY"))
    chat_llm.set_output_dir("llm_output")
    print(chat_llm.chat_complete(ZeroShotMessages(user_query="What is the sum of 124 and 789?"),
                                 model="deepseek-reasoner",
                                 temperature=0.6,
                                 return_str=True,
                                 title="calc_reasoning",
                                 n=1))

def example_cache():
    chat_llm = LLMService(output_dir="llm_output")
    messages = ZeroShotMessages(user_query="How are you doing?",
                                system_message="You are a cool girl and talk in that vibe.")
    print(chat_llm.chat_complete(messages,
                                 model="gpt-5-nano",
                                 temperature=1.0,
                                 return_str=True,
                                 use_cache=True))
    
def example_parallel():
    chat_llm = LLMService(output_dir="llm_output", parallels=2)
    messages = ZeroShotMessages(user_query="How are you doing?")
    print(chat_llm.chat_complete(messages,
                                 model="gpt-5-nano",
                                 temperature=1.0,
                                 title="parallel",
                                 return_str=True,
                                 n=5,
                                 use_cache=True,
                                 n_limit_per_query=2))

def example_parallel_multiple():
    chat_llm = LLMService(output_dir="llm_output", parallels=3)
    messages1 = ZeroShotMessages(user_query="How are you doing?")
    messages2 = ZeroShotMessages(user_query="Hi, how should I call you?")
    responses = chat_llm.chat_complete_batch([messages1, messages2],
                                             model="gpt-5-nano",
                                             return_str=True,
                                             n=2,
                                             n_limit_per_query=1,
                                             use_cache=True)
    for resp in responses:
        print(resp)

def example_ignore_cache_params():
    # A few-shot chat with an LLM
    chat_llm_not_ignore = LLMService(output_dir="llm_output")
    chat_llm_ignore = LLMService(output_dir="llm_output", ignore_cache_params=["temperature"])
    messages = ZeroShotMessages(user_query="What is 2+4?")
    print(chat_llm_not_ignore.chat_complete(messages,
                                            model="gpt-5-nano",
                                            temperature=1.0,
                                            return_str=True, use_cache=True))
    print(chat_llm_ignore.chat_complete(messages,
                                        model="gpt-5-nano",
                                        temperature=0.87,
                                        return_str=True, use_cache=True))

def example_offline():
    messages1 = ZeroShotMessages(user_query="9 * 8 = ?")
    messages2 = ZeroShotMessages(user_query="1000 - 7 = ?")
    prepare_offline_inference("temp/vol_offline.jsonl",
                              volcano_template.copy(),
                              [messages1, messages2],
                              n=2)

def example_vl():
    user_query = [
        {"type": "text", "text": "Describe the image."},
        {"type": "image_url", "image_url": {"url": "https://pet-health-content-media.chewy.com/wp-content/uploads/2025/04/16185711/202503bec-201610how-to-slow-down-dog-eating-1024x615.jpg"}}
    ]
    messages1 = ZeroShotVLMessages(user_query=user_query)
    chat_llm = LLMService(output_dir="llm_output")
    print(chat_llm.chat_complete(messages1,"gpt-5-nano", return_str=True, use_cache=True))


def example_batch_single():
    query = {
        "custom_id": "example_1",
        "body": {
            "messages": [
                {
                    # "role": "user", "content": "How many 'l's are in the word 'lullaby'? Place the final answer in \\boxed{}"
                    "role": "user", "content": "12345 * 6789 = ?"
                }
            ],
        }
    }

    # test client-side error handling
    print("Client-side error test cases:")
    print(process_single_query(query, {}))
    print(process_single_query(query, {'model': 'gpt-5-nano'}))
    print(process_single_query(query, {'model': 'gpt-5-nano', 'base_url': 'https://api.openai.com/v1', 'api_key': "env::OPENAI_KEY"}))
    print(process_single_query(query, {'model': 'gpt-5-nano', 'base_url': 'https://api.openai.com/v1'}))
    
    api_config = {
        "base_url": "https://api.openai.com/v1",
        "api_key": "env::OPENAI_API_KEY",
        "model": "gpt-5-nano",
    }
    # test retries error message
    print("Retries test case:")
    print(process_single_query(query, api_config, timeout=1, max_retries=2))

    # normal case
    print("Normal case:")
    print(process_single_query(query, api_config))
    # normal streaming
    print("Streaming case:")
    print(process_single_query(query, api_config, stream=True))

    # async case    
    async def async_test():
        async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as async_client:
            print("Async normal case:")
            success, result = await process_single_query_async(query, api_config, client=async_client)            
            print(success, result)
            # streaming
            print("Async streaming case:")
            success, result = await process_single_query_async(query, api_config, client=async_client, stream=True)
            print(success, result)
    asyncio.run(async_test())    


def example_count_tokens():
    from myllmutils.batch_process import TokenCounter
    counter = TokenCounter(model_name="deepseek-ai/DeepSeek-V3.2")
    print(f"tokens: {counter.count_tokens("Hello, how are you?")}")


def example_stream_max_tokens():
    from myllmutils.batch_process import TokenCounter

    query = {
        "custom_id": "stream_example_1",
        "body": {
            "messages": [{"role": "user", "content": "Tell a short, multi-sentence story about a robot who learns to paint."}],
        }
    }

    api_config = {
        "base_url": "https://api.deepseek.com",
        "api_key": "env::DS_API_KEY",
        "model": "deepseek-chat",
    }

    # TokenCounter used to count tokens; choose an installed tokenizer name.
    tc = TokenCounter(model_name="deepseek-ai/DeepSeek-V3.2")

    # Allow up to 30 tokens across content+reasoning in the stream
    success, result = process_single_query(query, api_config, stream=True, max_stream_tokens=30, token_counter=tc)
    print(success, result)

    async def async_test():
        async with httpx.AsyncClient() as async_client:
            success, result = await process_single_query_async(query, api_config, client=async_client, stream=True, max_stream_tokens=30, token_counter=tc)
            print(success, result)
    asyncio.run(async_test())


def example_gemini():
    """
    Example demonstrating Gemini protocol with both streaming and non-streaming modes.
    Tests both sync and async implementations.
    """
    query = {
        "custom_id": "gemini_example_1",
        "body": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Explain quantum entanglement in one sentence."}
                    ]
                }
            ],
        }
    }

    with open("example_api_configs/gemini-flash.yaml") as f:
        api_config = yaml.load(f, Loader=yaml.FullLoader)

    # Test 1: Non-streaming sync mode
    print("=" * 60)
    print("Test 1: Gemini Non-Streaming (Sync)")
    print("=" * 60)
    success, result = process_single_query(query, api_config, stream=False)
    if success:
        print(f"Success: {success}")
        print(f"Response: {result['response']}")
    else:
        print(f"Error: {result}")

    # Test 2: Streaming sync mode
    print("\n" + "=" * 60)
    print("Test 2: Gemini Streaming (Sync)")
    print("=" * 60)
    success, result = process_single_query(query, api_config, stream=True)
    if success:
        print(f"Success: {success}")
        print(f"Response: {result['response']}")
    else:
        print(f"Error: {result}")

    # Test 3: Async modes
    async def async_test():
        async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as async_client:
            # Test 3a: Non-streaming async mode
            print("\n" + "=" * 60)
            print("Test 3: Gemini Non-Streaming (Async)")
            print("=" * 60)
            success, result = await process_single_query_async(
                query, api_config, client=async_client, stream=False
            )
            if success:
                print(f"Success: {success}")
                print(f"Response: {result['response']}")
            else:
                print(f"Error: {result}")

            # Test 3b: Streaming async mode
            print("\n" + "=" * 60)
            print("Test 4: Gemini Streaming (Async)")
            print("=" * 60)
            success, result = await process_single_query_async(
                query, api_config, client=async_client, stream=True
            )
            if success:
                print(f"Success: {success}")
                print(f"Response: {result['response']}")
            else:
                print(f"Error: {result}")

    asyncio.run(async_test())

    print("\n" + "=" * 60)
    print("All Gemini tests completed!")
    print("=" * 60)


def main():
    # check the configuration
    print(myllmutils.about())

    # example_zeroshot()
    # example_disable_ssl()
    # example_fewshot()
    # example_logprobs()
    # example_sampling()
    # example_deepseek()
    # example_cache()
    # example_parallel() # TODO test parallel with completion
    # example_parallel_multiple()
    # example_ignore_cache_params()
    # example_offline()
    # example_vl()
    # example_batch_single()
    # example_count_tokens()
    example_stream_max_tokens()
    # example_gemini()
