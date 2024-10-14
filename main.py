import myllmutils
from myllmutils import LLMService
import os


if __name__ == '__main__':
    print(myllmutils.about())

    api_key = os.getenv("MYLLM_API_KEY")  # e.g., sk-proj-xxxx...
    proxy_url = os.getenv("MYLLM_URL")  # e.g., http://localhost:31000/v1/
    model_name = "gpt-4o-mini"

    chat_llm = LLMService(proxy_url, api_key=api_key)
    print(chat_llm.simple_chat("Which is larger, 0.11 or 0.9?", model_name).choices[0].message.content)
