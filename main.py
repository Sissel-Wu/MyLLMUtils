import myllmutils
from myllmutils import LLMService, ZeroShotMessages, FewShotMessages
from myllmutils import ResponseHelper


if __name__ == '__main__':
    # check the configuration
    print(myllmutils.about())

    # A zero-shot chat with an LLM
    chat_llm = LLMService()
    chat_llm.set_output_dir("llm_output")
    messages = ZeroShotMessages(user_query="How are you doing?",
                                system_message="You are a cool girl and talk in that vibe.")
    print(chat_llm.chat_complete(messages,
                                 model="gpt-4o-mini",
                                 temperature=1.0,
                                 return_str=True,
                                 title="say_hi"))

    # A few-shot chat with an LLM
    messages = FewShotMessages(system_message="Answer math questions.",
                               shots=[("What is 1+1?", "1+1=2"), ("What is 2+2?", "2+2=4")],
                               user_query="What is 2+4?")
    print(chat_llm.chat_complete(messages,
                                 model="gpt-4o-mini",
                                 temperature=1.0,
                                 return_str=True,
                                 title="math"))

    messages = ZeroShotMessages(user_query="There are three balls (red, blue, blue) in a black box. Pick a random ball from the box, what is the color? Answer randomly.")
    print(chat_llm.chat_complete(messages,
                                 model="gpt-4o-mini",
                                 temperature=1.0,
                                 return_str=True,
                                 title="random_color",
                                 logprobs=True,
                                 top_logprobs=10))

    messages = ZeroShotMessages(user_query="Pick a random substring in \"woijroi23oijovjasoijweijrowjieorjowiejr\".")

    print(chat_llm.chat_complete(messages,
                                 model="gpt-4o-mini",
                                 temperature=1.0,
                                 return_str=True,
                                 title="random_substr",
                                 n=5))

    print(chat_llm.chat_complete_greedy(messages,
                                        model="gpt-4o-mini",
                                        return_str=True,
                                        title="random_substr_greedy",
                                        n=5))
