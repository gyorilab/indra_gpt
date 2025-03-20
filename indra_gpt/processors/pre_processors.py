import logging
import json
import random
from typing import Any, Dict, List, Tuple

from indra_gpt.configs import PreProcessorConfig
from indra_gpt.resources.constants import (JSON_SCHEMA, 
                                           INPUT_DEFAULT,
                                           SCHEMA_STRUCTURED_OUTPUT_PATH)
from indra_gpt.util.util import sample_from_input_file

logger = logging.getLogger(__name__)


class PreProcessor:
    def __init__(self, config: PreProcessorConfig) -> None:
        self.config = config

    def process(self) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Processes user inputs by sampling input texts and constructing 
        n-shot prompt history if applicable.

        Returns:
            Tuple containing:
                - raw_input_data: List of dictionaries with "text" and "pmid"
                - preprocessed_data: Dictionary with input texts and n-shot history
        """
        logger.info("Processing user inputs...")

        # Sample input texts from the input file
        raw_input_data = sample_from_input_file(self.config, self.config.base_config.random_seed)
        input_texts = [entry["text"] for entry in raw_input_data]

        # Handle N-shot prompting
        n_shot_prompting = self.config.base_config.n_shot_prompting
        n_shot_history = []  # Default to an empty list if not used

        if n_shot_prompting > 0:
            n_shot_history = self.n_shot_prompt_history()

        # Store results
        preprocessed_data = {
            "input_texts": input_texts,
            "n_shot_history": n_shot_history
        }

        return raw_input_data, preprocessed_data

    def n_shot_prompt_history(self) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
        """
        Constructs n-shot examples as a list of (user prompt, assistant response) pairs.

        Returns:
            List of tuples where each tuple consists of:
                - A dictionary representing the user prompt
                - A dictionary representing the assistant response
        """
        n = self.config.base_config.n_shot_prompting
        if self.config.base_config.structured_output:
            with open(SCHEMA_STRUCTURED_OUTPUT_PATH) as file:
                json_schema = json.load(file)
            json_schema_string = json.dumps(json_schema, indent=2)
        else:
            json_schema_string = json.dumps(JSON_SCHEMA, indent=2)
        user_prompt = (
            "Read the following JSON schema for a statement "
            "object:\n\n```json\n"
            + json_schema_string
            + "\n```\n\nExtract the relation from the following sentence and put it in a JSON object matching the schema above. "
            "The JSON object needs to be able to pass validation against the schema. If the statement type is 'RegulateActivity', "
            "list it as either 'Activation' or 'Inhibition'. If the statement type is 'RegulateAmount', list it as either 'IncreaseAmount' "
            "or 'DecreaseAmount'. Only respond with the JSON object.\n\nSentence: "
        )

        reduced_user_prompt = (
            "Extract the relation from the following sentence and put it in a JSON object "
            "matching the provided schema. The JSON object needs to be able to pass validation against the schema. "
            "If the statement type is 'RegulateActivity', list it as either 'Activation' or 'Inhibition'. If "
            "the statement type is 'RegulateAmount', list it as either 'IncreaseAmount' or 'DecreaseAmount'. "
            "Only respond with the JSON object.\n\nSentence: "
        )

        with open(INPUT_DEFAULT, "r", encoding="utf-8") as f:
            statements_json_content = json.load(f)

        # Handle case where `n` exceeds available statements
        n = min(n, len(statements_json_content))
        if n < self.config.base_config.n_shot_prompting:
            logger.warning(f"Requested {n} samples, but only {len(statements_json_content)} are available. Using {n} available statements.")

        # Sample `n` statements ONCE to be used for ALL input samples
        history_sample_json_stmts = random.sample(statements_json_content, n)

        # Log how many statements are being used
        logger.info(f"Generating {n}-shot prompt history using {len(history_sample_json_stmts)} examples.")

        # Construct history prompt
        history_prompt_response_pairs = []
        for i, sample_json_stmt in enumerate(history_sample_json_stmts):
            assistant_response = json.dumps(sample_json_stmt)  # Convert to JSON format

            if i == 0:
                history_prompt_response_pairs.append(
                    ({"role": "user", "content": user_prompt + sample_json_stmt["evidence"][0]["text"]},
                     {"role": "assistant", "content": assistant_response})
                )
            else:
                history_prompt_response_pairs.append(
                    ({"role": "user", "content": reduced_user_prompt + sample_json_stmt["evidence"][0]["text"]},
                     {"role": "assistant", "content": assistant_response})
                )

        return history_prompt_response_pairs
