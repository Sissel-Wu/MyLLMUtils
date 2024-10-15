import myllmutils
from myllmutils import LLMService, ZeroShotMessages


if __name__ == '__main__':
    # check the configuration
    print(myllmutils.about())
    # The simplest chat with the LLM
    chat_llm = LLMService()
    chat_llm.set_output_dir("llm_output")
    messages = ZeroShotMessages(user_query="How are you doing?",
                                system_message="You are a cool girl and talk in that vibe.")
    print(chat_llm.chat_complete(messages,
                                 model="gpt-4o-mini",
                                 temperature=1.0,
                                 return_str=True,
                                 title="say_hi"))
