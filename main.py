import myllmutils
from myllmutils import LLMService


if __name__ == '__main__':
    # check the configuration
    print(myllmutils.about())
    # The simplest chat with the LLM
    chat_llm = LLMService()
    print(chat_llm.simple_chat("Good morning"))
