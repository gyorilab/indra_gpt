import logging
import json
import random
from indra_gpt.configs import PreProcessingConfig
from indra_gpt.resources.constants import JSON_SCHEMA, INPUT_DEFAULT
from indra_gpt.util.util import load_input_file

class PreProcessor:
    def __init__(self, config: PreProcessingConfig):
        """
        Initializes the preprocessor with the given configuration.

        Parameters:
        - config (PreProcessingConfig): Configuration containing user inputs, sampling settings, and preprocessing options.
        """
        self.config = config
        random.seed(42)
        self.logger = logging.getLogger(__name__)

    def process(self):
        self.logger.info("Processing user inputs...")

        # Sample input texts from the input file
        input_samples = self.sample_from_input_file()

        # Handle N-shot prompting
        n_shot_prompting = self.config.base_config.get("n_shot_prompting", 0)
        n_shot_history = []  # Default to an empty list if not used

        if n_shot_prompting > 0:
            n_shot_history = self.n_shot_prompt_history()

        # Store results
        preprocessed_data = {
            "input_samples": input_samples,
            "n_shot_history": n_shot_history
        }

        return preprocessed_data

    def sample_from_input_file(self): 
        user_inputs_file = load_input_file(self.config.base_config.get("user_inputs_file"))
        user_input_texts = [entry["text"] for entry in user_inputs_file]

        do_random_sample = self.config.base_config.get("random_sample", False)
        num_samples = self.config.base_config.get("num_samples", len(user_input_texts))  # Default to full dataset

        if num_samples > len(user_input_texts):
            self.logger.warning(f"Requested {num_samples} samples, but only {len(user_input_texts)} available. Using all available samples.")
            num_samples = len(user_input_texts)

        if do_random_sample:
            return random.sample(user_input_texts, num_samples)
        else:
            return user_input_texts[:num_samples]  # Select first N samples if not random

    def n_shot_prompt_history(self):
        """
        Constructs n-shot examples as a list of (user prompt, assistant response) pairs.
        """
        n = self.config.base_config.get("n_shot_prompting", 0)

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
        if n < self.config.base_config.get("n_shot_prompting", 0):
            self.logger.warning(f"Requested {n} samples, but only {len(statements_json_content)} are available. Using {n} available statements.")

        # Sample `n` statements ONCE to be used for ALL input samples
        history_sample_json_stmts = random.sample(statements_json_content, n)

        # Log how many statements are being used
        self.logger.info(f"Generating {n}-shot prompt history using {len(history_sample_json_stmts)} examples.")

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
