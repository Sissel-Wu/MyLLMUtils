import myllmutils
from myllmutils import LLMService, ZeroShotMessages, FewShotMessages
from myllmutils import ResponseHelper


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
    import os
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

if __name__ == '__main__':
    # check the configuration
    print(myllmutils.about())

    # example_zeroshot()
    # example_disable_ssl()
    example_fewshot()
    # example_logprobs()
    example_sampling()
    # example_deepseek()
    example_cache()
