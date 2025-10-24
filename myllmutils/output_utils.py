import json
from pathlib import Path
from typing import Any

Query = list[dict[str, str]]
Params = dict[str, Any]


class ResponseHelper:
    def __init__(self, response: dict | list[dict]) -> None:
        self.raw_response = response
        if type(response) == dict:
            self.choices = response["choices"]
        else:
            self.choices = []
            for r in response:
                self.choices.extend(r["choices"])

    def get_logprobs_at(self, index: int, choice=0) -> list[tuple[str, float]]:
        """
        Get the top logprobs at a specific index.
        :param index: index of the token of the response content.
        :param choice: index of the choice.
        :return:
        """
        top_logprobs = self.choices[choice]["logprobs"]["content"][index]["top_logprobs"]
        return [(elem["token"], elem["logprob"]) for elem in top_logprobs]

    def content(self, choice: int | str | None = None) -> str | list[str]:
        """
        Return the content of the response. If choice="all", return a list of all choices. By default, return all if there is >1 choice, otherwise return the only choice.
        """
        all_content = [c["message"]["content"] for c in self.choices]
        if type(choice) is int:
            return all_content[choice]
        elif choice is None:
            return all_content[0] if len(all_content) == 1 else all_content
        elif choice == "all":
            return all_content
        else:
            raise ValueError("choice must be an integer or 'all'")

    def reasoning_content(self, choice=0) -> str | None:
        """
        Return the reasoning content of the response, if any.
        """
        message = self.choices[choice]["message"]
        if "reasoning_content" in message:
            return message["reasoning_content"]
        return None

    def num_choices(self):
        """
        Return the number of choices in the response.
        """
        return len(self.choices)

    def chat_id(self):
        """
        Return the chat id of the response. If more than one response is included, return the first one.
        """
        resp = self.raw_response if type(self.raw_response) == dict else self.raw_response[0]
        return resp["id"]


def load_from_json_file(file_path: str | Path) -> (Query, ResponseHelper, Params):
    """
    Load a query-response pair from a JSON file dumped by this library.
    :param file_path: path to the json file.
    :return: a tuple of query (list of {"role": "xx", "content": "xx"}) and response (dict following the openai.ChatCompletion structure).
    """
    with open(file_path, "r") as f:
        js_obj = json.load(f)
        q= js_obj["query"]
        ps = js_obj["params"] if "params" in js_obj else None
        r = ResponseHelper(js_obj["response"])
        return q, r, ps


def _to_key(query: Query, params: dict[str, Any] | None, ignore_params: list[str]):
    ignore_params = ignore_params or []
    key = (tuple(tuple(sorted([(_k, _v) for _k, _v in _dic.items()])) for _dic in query),
           tuple(sorted([(_k, _v) for _k, _v in params.items() if _k not in ignore_params])) if params else None)
    return key


class _QueryCache:
    def __init__(self, ignore_params: list[str]):
        self._map = {}
        self.ignore_params = ignore_params or []

    def add(self, query: Query, response: ResponseHelper, params: dict[str, Any] | None):
        self._map[_to_key(query, params, self.ignore_params)] = response

    def get(self, query: Query, params: dict[str, Any] | None) -> ResponseHelper | None:
        return self._map.get(_to_key(query, params, self.ignore_params), None)


class CacheHelper:
    def __init__(self, dir_path: str | Path, ignore_cache_params: list[str] | None = None):
        self.dir = Path(dir_path) if isinstance(dir_path, str) else dir_path
        self.map = None
        self.ignore_cache_params = ignore_cache_params or []

    def get(self, name: str) -> (Query, ResponseHelper, Params):
        """
        Get a query-response pair from the cache.
        :param name: name of the query-response pair.
        :return: the corresponding tuple.
        """
        file_path = self.dir / "raw" / f"{name}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Cache file {file_path} does not exist.")
        return load_from_json_file(file_path)

    def _get_map(self) -> _QueryCache:
        """
        Get (initialize if nonexistent) the map from queries to responses.
        :return:
        """
        if self.map is not None:
            return self.map
        self.map = _QueryCache(self.ignore_cache_params)
        raw_dir = self.dir / "raw"
        if not raw_dir.exists():  # empty cache
            return self.map
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Cache directory {raw_dir} is not a directory.")
        for file_path in raw_dir.glob("*.json"):
            q, r, ps = load_from_json_file(file_path)
            self.map.add(q, r, ps)
        return self.map

    def get_by_query(self, query: Query, params: Params | None) -> ResponseHelper | None:
        """
        Get a response from the cache by query.
        :param query: the query.
        :param params: the parameters, e.g., temperature, top_p. Use None for backward compatibility.
        :return: the corresponding response.
        """
        m = self._get_map()
        return m.get(query, params)

    def add(self, query: Query, response: ResponseHelper, params: Params):
        """
        Add a query-response pair to the cache.
        :param query: the query.
        :param response: the response.
        :param params: the parameters, e.g., temperature, top_p.
        :return:
        """
        m = self._get_map()
        m.add(query, response, params)


if __name__ == '__main__':
    # test logprobs
    p0, rh, _ = load_from_json_file("../llm_output/raw/random_color.json")
    print(p0[0]["role"])
    print(p0[0]["content"])
    print(rh.content())
    logprobs = rh.get_logprobs_at(-2)
    for k, v in logprobs:
        print(f"- '{k}': {v}")

    # test num_choices
    p0, rh, _ = load_from_json_file("../llm_output/raw/random_substr.json")
    print(rh.num_choices())
    print(rh.content(0))
    print(rh.content(4))

    # test reasoning_content
    _, rh, _ = load_from_json_file("../llm_output/raw/calc_reasoning.json")
    print(rh.reasoning_content(0))

    cache_helper = CacheHelper("../llm_output")
    _, rh2, _ = cache_helper.get("calc_reasoning")
    assert rh.reasoning_content(0) == rh2.reasoning_content(0)

    rh3 = cache_helper.get_by_query([{"role": "user", "content": "What is the sum of 124 and 789?"}], None) # calc_reasoning
    assert rh2.reasoning_content(0) == rh3.reasoning_content(0)
