import json

Query = list[dict[str, str]]


class ResponseHelper:
    def __init__(self, response: dict):
        self.raw_response = response

    def get_logprobs_at(self, index: int, choice=0) -> list[(str, float)]:
        """
        Get the top logprobs at a specific index.
        :param index: index of the token of the response content.
        :param choice: index of the choice.
        :return:
        """
        top_logprobs = self.raw_response["choices"][choice]["logprobs"]["content"][index]["top_logprobs"]
        return [(elem["token"], elem["logprob"]) for elem in top_logprobs]

    def content(self, choice=0) -> str:
        """
        Return the content of the response.
        """
        return self.raw_response["choices"][choice]["message"]["content"]

    def reasoning_content(self, choice=0) -> str | None:
        """
        Return the reasoning content of the response, if any.
        """
        message = self.raw_response["choices"][choice]["message"]
        if "reasoning_content" in message:
            return message["reasoning_content"]
        return None

    def num_choices(self):
        """
        Return the number of choices in the response.
        """
        return len(self.raw_response["choices"])


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
    # test logprobs
    p0, rh = load_from_json_file("../llm_output/raw/random_color.json")
    print(p0[0]["role"])
    print(p0[0]["content"])
    print(rh.content())
    logprobs = rh.get_logprobs_at(-2)
    for k, v in logprobs:
        print(f"- '{k}': {v}")

    # test num_choices
    p0, rh = load_from_json_file("../llm_output/raw/random_substr.json")
    print(rh.num_choices())
    print(rh.content(0))
    print(rh.content(4))

    # test reasoning_content
    _, rh = load_from_json_file("../llm_output/raw/calc_reasoning.json")
    print(rh.reasoning_content(0))
