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
    
    def get_input_json_objects(self):
        with open(self.statements_file_json, "r") as f:
            statements_json_content = json.load(f)
        # assign first N json objects to json_object_list
        if len(statements_json_content) < self.iterations:
            logger.warning(f"Number of iterations is greater than the number of statements "
                        f"in the file. All {self.iterations} statements will be processed.")
        if self.random_sample:
            original_statement_json_objects = random.sample(statements_json_content, self.iterations)
        else:
            original_statement_json_objects = statements_json_content[:self.iterations]
        return original_statement_json_objects

    def get_input_texts(self, original_statement_json_objects):
        input_texts = []
        for original_statement_json_object in original_statement_json_objects:
            input_texts.append(original_statement_json_object["evidence"][0]["text"])
        return input_texts
    
    def get_json_object_map(self, original_statement_json_objects):
        json_object_map = {}
        for stmt_json in original_statement_json_objects:
            mh = stmt_json["matches_hash"]
            json_object_map[mh] = stmt_json
        return json_object_map

    def get_samples(self, 
                    original_statement_json_objects_map,
                    matches_hash,
                    n=2):
        sequence = list(set(original_statement_json_objects_map.keys()) - {matches_hash})
        samples_hashes = random.sample(sequence, n)
        samples = [original_statement_json_objects_map[h] for h in samples_hashes]
        return samples
    
    def get_chat_prompt(self, original_statement_json_object):
        # reduced prompt not including schema
        PROMPT_reduced = (
            "Extract the relation from the following sentence and put it in a "
            "JSON object matching the schema above. The JSON object needs to be "
            "able to pass a validation against the provided schema. If the "
            "statement type is 'RegulateActivity', list it instead as either "
            "'Activation' or 'Inhibition'. If the statement type is "
            "'RegulateAmount', list it instead as either 'IncreaseAmount' or 'DecreaseAmount'. Only "
            "respond with "
            "the JSON object.\n\nSentence: "
        )
        prompt = PROMPT_reduced + original_statement_json_object["evidence"][0]["text"]
        return {"role": "user", "content": prompt}
    
    def get_chat_history(self, samples):
        json_schema_string = json.dumps(JSON_SCHEMA)  # converting json schema to a string
        PROMPT = (
            "Read the following JSON schema for a statement "
            "object:\n\n```json\n"
            + json_schema_string
            + "\n```\n\nExtract the relation from the following sentence and put it in a JSON object matching the schema above. The JSON object needs to be able to pass a validation against the provided schema. If the statement type is 'RegulateActivity', list it instead as either 'Activation' or 'Inhibition'. If the statement type is 'RegulateAmount', list it instead as either 'IncreaseAmount' or 'DecreaseAmount'. Only respond with "
            "the JSON object.\n\nSentence: "
        )
        # reduced prompt not including schema
        PROMPT_reduced = (
            "Extract the relation from the following sentence and put it in a "
            "JSON object matching the schema above. The JSON object needs to be "
            "able to pass a validation against the provided schema. If the "
            "statement type is 'RegulateActivity', list it instead as either "
            "'Activation' or 'Inhibition'. If the statement type is "
            "'RegulateAmount', list it instead as either 'IncreaseAmount' or 'DecreaseAmount'. Only "
            "respond with "
            "the JSON object.\n\nSentence: "
        )
        history = []
        for i, sample in enumerate(samples):
            evidence_text = sample["evidence"][0]["text"]
            # If sample is not the last one, add the prompt with the schema
            if i == 0:
                user_message = {"role": "user", "content": PROMPT + evidence_text}
            else:
                user_message = {"role": "user", "content": PROMPT_reduced + evidence_text}
            assistant_message = {"role": "assistant", "content": json.dumps(sample)}
            history.extend([user_message, assistant_message])
        return history

    def get_response_single_inference(self, chat_prompt, chat_history=None, max_tokens=1, retry_count=3, strip=True, debug=False):
        messages = chat_history + [chat_prompt]
        retry_count = max(retry_count, 1)
        response = None
        for i in range(retry_count):
            try:
                if self.structured_output:
                    with open(SCHEMA_STRUCTURED_OUTPUT_PATH) as file:
                        schema = json.load(file)
                    post_processed_schema = merge_allOf(schema, schema)
                    response = client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[chat_prompt],
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
                        model=self.model,
                        temperature=0,
                        max_tokens=max_tokens,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        messages=messages,
                    )
            except Exception as e:
                # Retry the request if it fails
                if i < retry_count - 1:
                    logger.warning(
                        f"Request failed with error: {e}. Retrying after 5 seconds."
                    )
                    sleep(5)
                else:
                    raise e
        if debug:
            logger.info(
                f"messages:\n-------\n{messages}\n-------\n"
                f"Response:\n---------\n{response.dict()}\n---------\n\n"
            )
        if response is None:
            raise RuntimeError("No response from OpenAI")

        if response.choices[0].finish_reason == "length":
            logger.warning(
                "OpenAI response was truncated. Likely due to token "
                "constraints. Consider increasing the max_tokens parameter."
            )
        # Remove whitespace and trailing punctuations 
        reply = response.choices[0].message.content
        if strip:
            reply = reply.strip().rstrip(".,!")
        if reply == "":
            logger.warning(
                "OpenAI returned an empty reply. See full API response below for details."
            )
            print(f"Response:\n---------\n{response}\n---------\n\n")
        return response

    def generate_statement_json_objects(self, preprocessed_data: List(str)):
        generated_statement_json_objects = []
        for input_text in tqdm(preprocessed_data, desc="Extracting", unit="statement"):
            samples = self.get_samples()
            chat_prompt = self.get_chat_prompt(input_text)
            chat_history = self.get_chat_history(samples)
            response = self.get_response_single_inference(chat_prompt, chat_history, max_tokens=9000)
            generated_json_object = response.choices[0].message.content
            generated_statement_json_objects.append(generated_json_object)
        return generated_statement_json_objects
