import logging
from time import sleep
import json
import time
from openai import OpenAI
from indra.config import IndraConfigError, get_config
from indra_gpt.resources.constants import OUTPUT_DIR, JSON_SCHEMA, SCHEMA_STRUCTURED_OUTPUT_PATH
import random
from tqdm import tqdm
from indra_gpt.util.util import merge_allOf
from indra_gpt.configs import GenerationConfig


logger = logging.getLogger(__name__)

try:
    api_key = get_config("OPENAI_API_KEY", failure_ok=False)
    organization = get_config("OPENAI_ORG")
except IndraConfigError as err:
    raise KeyError(
        "Please set OPENAI_API_KEY in the environment or in the indra config."
    ) from err

client = OpenAI(api_key=api_key, organization=organization)

class OpenAIClient:
    def __init__(self, config: GenerationConfig):
        self.config = config
    
    def get_chat_prompt(self, input_text):
        # reduced prompt not including schema
        PROMPT_reduced = (
            "Extract the relation from the following sentence and put it in a "
            "JSON object matching the provided schema. The JSON object needs to be "
            "able to pass a validation against the provided schema. If the "
            "statement type is 'RegulateActivity', list it instead as either "
            "'Activation' or 'Inhibition'. If the statement type is "
            "'RegulateAmount', list it instead as either 'IncreaseAmount' or 'DecreaseAmount'. Only "
            "respond with "
            "the JSON object.\n\nSentence: "
        )
        prompt = PROMPT_reduced + input_text
        return {"role": "user", "content": prompt}
    
    def make_api_call(self, messages, max_tokens=1, retry_count=3):
        retry_count = max(retry_count, 1)
        response = None

        for i in range(retry_count):
            try:
                if self.config.structured_output:
                    with open(SCHEMA_STRUCTURED_OUTPUT_PATH) as file:
                        schema = json.load(file)
                    post_processed_schema = merge_allOf(schema, schema)

                    response = client.beta.chat.completions.parse(
                        model=self.config.model,
                        messages=messages,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "indra_statement_json",
                                "strict": True,
                                "schema": post_processed_schema
                            }
                        }
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.config.model,
                        temperature=0,
                        max_tokens=max_tokens,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        messages=messages,
                    )
                return response  # If successful, return the response immediately

            except Exception as e:
                if i < retry_count - 1:
                    logger.warning(f"Request failed with error: {e}. Retrying after 5 seconds.")
                    sleep(5)
                else:
                    raise e

        return response  # Return the last response if retries exhausted

    def get_response(self, chat_prompt, chat_history=None, max_tokens=9000):
        feedback_iterations = self.config.feedback_refinement_iterations

        if chat_history is None:
            chat_history = []

        messages = chat_history + [chat_prompt]  # Initialize message history

        # Generate initial response
        response = self.make_api_call(messages, max_tokens)
        generated_response = response.choices[0].message.content

        # Perform iterative refinement if needed
        for _ in range(feedback_iterations):
            refinement_prompt = {
                "role": "user",
                "content": (
                    "Take a look at the response you just generated and the JSON schema provided. "
                    "If you detect any errors or missing elements, fix these errors and re-generate the response.\n\n"
                    "Previous Response:\n" + generated_response
                )
            }

            # Append previous response and refinement instruction
            messages.append({"role": "assistant", "content": generated_response})
            messages.append(refinement_prompt)

            # Generate refined response
            response = self.make_api_call(messages, max_tokens)
            generated_response = response.choices[0].message.content  # Update latest response

        return generated_response  # Return final refined response


    def generate_statement_json_objects(self, preprocessed_data):
        input_texts = preprocessed_data["input_samples"]
        n_shot_history = preprocessed_data["n_shot_history"]
        generated_statement_json_objects = []
        for input_text in tqdm(input_texts, desc="Extracting", unit="statement"):
            chat_prompt = self.get_chat_prompt(input_text)
            chat_history = n_shot_history
            response = self.get_response(chat_prompt, chat_history, max_tokens=9000)
            generated_json_object = response.choices[0].message.content
            generated_statement_json_objects.append(generated_json_object)
        return generated_statement_json_objects
