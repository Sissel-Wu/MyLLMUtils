from myllmutils.services import *
from myllmutils.output_utils import *


def about() -> str:
    """
    Return a string about the running configuration.
    """
    base_url = environ.get("MYLLM_URL")
    if not base_url:
        base_url = "N/A"
    oai_api_key = "set" if environ.get("OPENAI_API_KEY") else "not set"
    my_api_key = "set and replacing OPENAI_API_KEY" if environ.get("MYLLM_API_KEY") else "not set"
    res = "\n".join([
        "Configuration:",
        f"- env OPENAI_API_KEY is {oai_api_key}.",
        f"- env MYLLM_API_KEY is {my_api_key}.",
        f"- env MYLLM_URL is {base_url}."
    ])
    return res
