import json

Query = list[dict[str, str]]


class ResponseHelper:
    def __init__(self, response: dict):
        self.raw_response = response

    def get_logprobs_at(self, index: int) -> list[(str, float)]:
        """
        Get the top logprobs at a specific index.
        :param index: index of the token of the response content.
        :return:
        """
        top_logprobs = self.raw_response["choices"][0]["logprobs"]["content"][index]["top_logprobs"]
        return [(elem["token"], elem["logprob"]) for elem in top_logprobs]

    def content(self) -> str:
        """
        Return the content of the response.
        """
        return self.raw_response["choices"][0]["message"]["content"]


def load_from_json_file(file_path: str) -> (Query, ResponseHelper):
    """
    Load a query-response pair from a JSON file dumped by this library.
    :param file_path: path to the json file.
    :return: a tuple of query (list of {"role": "xx", "content": "xx"}) and response (dict following the openai.ChatCompletion structure).
    """
    with open(file_path, "r") as f:
        js_obj = json.load(f)
        q, r = js_obj["query"], js_obj["response"]
        return q, ResponseHelper(r)


if __name__ == '__main__':
    p0, rh = load_from_json_file("../llm_output/raw/random_color.json")
    print(p0[0]["role"])
    print(p0[0]["content"])
    print(rh.content())
    logprobs = rh.get_logprobs_at(-2)
    for k, v in logprobs:
        print(f"- '{k}': {v}")
