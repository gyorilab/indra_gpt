import logging
import json
from indra_gpt.resources.constants import (INDRA_SCHEMA,
                                           INDRA_BENCHMARK_CORPUS_ALL_CORRECT)
import random

logger = logging.getLogger(__name__)


with open(INDRA_SCHEMA, "r") as file:
    indra_schema = json.load(file)
with open(INDRA_BENCHMARK_CORPUS_ALL_CORRECT, "r") as file:
    indra_benchmark_corpus_all_correct = json.load(file)


class EndToEndExtractor:
    def __init__(
        self,
        llm_client=None,
        num_history_examples=0
    ):
        self.llm = llm_client
        self.num_history_examples = num_history_examples

    def get_history_examples(self, num_examples):

        with open(INDRA_BENCHMARK_CORPUS_ALL_CORRECT, "r") as file:
            indra_benchmark_corpus_all_correct = json.load(file)

        history_samples = random.sample(
            indra_benchmark_corpus_all_correct,
            num_examples
        )

        user_prompt = (
            "Read the following JSON schema for a statement "
            "object:\n\n```json\n"
            + json.dumps(indra_schema, indent=2)
            + "\n```\n\nExtract the relation from the following sentence and put it in a JSON object matching the schema above. "
            "The JSON object needs to be able to pass validation against the schema. If the statement type is 'RegulateActivity', "
            "list it as either 'Activation' or 'Inhibition'. If the statement type is 'RegulateAmount', list it as either 'IncreaseAmount' "
            "or 'DecreaseAmount'. Only respond with a JSON array of objects. Do not include any explanation or text outside the array. "
            "Wrap even a single object in square brackets [ ].\n\nSentence: "
        )
        reduced_user_prompt = (
            "Extract the relation from the following sentence and put it in a JSON object "
            "matching the provided schema. The JSON object needs to be able to pass validation against the schema. "
            "If the statement type is 'RegulateActivity', list it as either 'Activation' or 'Inhibition'. If "
            "the statement type is 'RegulateAmount', list it as either 'IncreaseAmount' or 'DecreaseAmount'. "
            "Only respond with a JSON array of objects. Do not include any explanation or text outside the array. "
            "Wrap even a single object in square brackets [ ].\n\nSentence: "
        )

        # Construct history prompt
        history = []
        for i, sample_json_stmt in enumerate(history_samples):
            assistant_response = json.dumps([sample_json_stmt], indent=2)
            if i == 0:
                history.extend(
                    [{"role": "user", "content": user_prompt + sample_json_stmt["evidence"][0]["text"]},
                     {"role": "assistant", "content": assistant_response}]
                )
            else:
                history.extend(
                    [{"role": "user", "content": reduced_user_prompt + sample_json_stmt["evidence"][0]["text"]},
                     {"role": "assistant", "content": assistant_response}]
                )
        return history
    
    def raw_extract(self, text):
        if self.llm is None:
            raise ValueError("LLM client is not initialized.")

        # Prepare the prompt
        prompt = (
            "Read the following JSON schema for a statement object:\n\n```json\n"
            + json.dumps(indra_schema, indent=2)
            + "\n```\n\nExtract the relation from the following sentence and put it in a JSON object matching the schema above. "
            "The JSON object needs to be able to pass validation against the schema. If the statement type is 'RegulateActivity', "
            "list it as either 'Activation' or 'Inhibition'. If the statement type is 'RegulateAmount', list it as either 'IncreaseAmount' "
            "or 'DecreaseAmount'. Only respond with a JSON array of objects. Do not include any explanation or text outside the array. "
            "Wrap even a single object in square brackets [ ].\n\nSentence: "
            + text
        )

        # Get history examples
        history = self.get_history_examples(self.num_history_examples)

        # Call the LLM client
        response = self.llm.call(prompt, history=history, response_format={"type": "json_object"})
        
        return response

