from indra_gpt.clients.client_interface import ClientInterface
import openai

import logging
from time import sleep
import json
from pathlib import Path
import time
from openai import OpenAI
from indra.config import IndraConfigError, get_config
from indra_gpt.resources.constants import OUTPUT_DIR, JSON_SCHEMA, OUTPUT_DEFAULT, SCHEMA_STRUCTURED_OUTPUT_PATH
import random
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from indra_gpt.util import post_process_extracted_json
from indra_gpt.util.util import merge_allOf
from indra.statements.io import stmt_from_json
import pandas as pd
from json import JSONDecodeError


logger = logging.getLogger(__name__)

try:
    api_key = get_config("OPENAI_API_KEY", failure_ok=False)
    organization = get_config("OPENAI_ORG")
except IndraConfigError as err:
    raise KeyError(
        "Please set OPENAI_API_KEY in the environment or in the indra config."
    ) from err

client = OpenAI(api_key=api_key, organization=organization)

class OpenAIClient(ClientInterface):
    def __init__(self, **kwargs):
        self.client = client
        self.statements_file_json = kwargs.get("statements_file_json")
        self.model = kwargs.get("model")
        self.iterations = kwargs.get("iterations")
        self.output_file = kwargs.get("output_file")
        self.verbose = kwargs.get("verbose")
        self.batch_job = kwargs.get("batch_job")
        self.batch_id = kwargs.get("batch_id")
        self.structured_output = kwargs.get("structured_output")
    
    def get_input_json_objects(self):
        with open(self.statements_file_json, "r") as f:
            statements_json_content = json.load(f)
        # assign first N json objects to json_object_list
        if len(statements_json_content) < self.iterations:
            logger.warning(f"Number of iterations is greater than the number of statements "
                        f"in the file. All {self.iterations} statements will be processed.")
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

    def send_batch_inference(self, original_statement_json_objects, chat_prompts, chat_histories, max_tokens=1):
        batch_requests = []
        if self.structured_output:
            with open(SCHEMA_STRUCTURED_OUTPUT_PATH) as file:
                schema = json.load(file)
            post_processed_schema = merge_allOf(schema, schema)

        for i, (chat_prompt, chat_history) in enumerate(zip(chat_prompts, chat_histories)):
            messages = chat_history + [chat_prompt]
            if self.structured_output:
                batch_requests.append(
                    {
                        "custom_id": f"request-{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": [chat_prompt],
                            "max_tokens": max_tokens,
                            "response_format":{
                                "type": "json_schema", 
                                "json_schema": {
                                    "name": "indra_statement_json",
                                    "strict": True, 
                                    "schema": post_processed_schema
                                }
                            }
                        }
                    }
                )
            else:
                batch_requests.append(
                    {
                        "custom_id": f"request-{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": max_tokens
                        }
                    }
                )
        # Make a temporary directory in the directory one level above the directory of this file
        batches_dir_path = OUTPUT_DIR / "batches"
        batches_dir_path.mkdir(parents=True, exist_ok=True)  # Create the parent directory if it doesn't exist
        # Write the batch requests to a file
        tmp_batch_input_file_path = batches_dir_path / "batch_input.jsonl"
        with open(tmp_batch_input_file_path, "w") as f:
            for request in batch_requests:
                f.write(json.dumps(request) + "\n")
        # Upload the batch file to OpenAI
        batch_input_file = client.files.create(
            file=open(tmp_batch_input_file_path, "rb"),
            purpose="batch"
        )
        # delete the temporary file
        tmp_batch_input_file_path.unlink()
        # Create the batch job using the uploaded file
        batch_input_file_id = batch_input_file.id
        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "nightly eval job"
            }
        )
        # Retrieve batch ID
        batch_id = client.batches.list().data[0].to_dict()['id']
        # Make a batch input directory, where the batch input and output files are both stored
        batch_dir_path = batches_dir_path / batch_id
        batch_dir_path.mkdir(parents=True, exist_ok=True)
        # Write the batch requests to a file to keep track of the input
        batch_input_file_path = batch_dir_path / "batch_input.jsonl"
        with open(batch_input_file_path, "w") as f:
            for request in batch_requests:
                f.write(json.dumps(request) + "\n")
        # Write the original json statements to a file to keep track of the input
        original_json_statements_path = batch_dir_path / "original_statements.jsonl"
        with open(original_json_statements_path, "w") as f:
            for statement in original_statement_json_objects:
                f.write(json.dumps(statement) + "\n")
        # Also add a txt file with when this batch was created
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(batch_dir_path / "metadata.txt", "w") as f:
            f.write(f"Batch created at time: {current_time}\n")
            f.write(f"Batch ID: {batch_id}\n")

        return batch_id
    
    def get_response_batch_inference(self, batch_id):
        try:
            batch = client.batches.retrieve(batch_id)
            logger.info("Fetched data for batch id {batch_id})")
            status = batch.to_dict()['status']
            if status == "completed":
                batch_output_file_id = batch.to_dict()['output_file_id']
                # Assuming file_response_text contains the string with multiple JSON objects
                file_response_text = client.files.content(batch_output_file_id).text
                # Split the response into lines and load each line as a JSON object
                batch_response = []
                for line in file_response_text.splitlines():
                    if line.strip():  # Skip empty lines
                        batch_response.append(json.loads(line))                
            else:
                logger.info(f"Batch job info: {batch}")
                batch_response = None
        except Exception as e:
            logger.error(f"Error getting batch replies: {e}")
            batch_response = None
        return batch_response

    def generate_statement_json_objects(self):
        original_statement_json_objects = self.get_input_json_objects()
        get_json_object_map = self.get_json_object_map(original_statement_json_objects)
        generated_statement_json_objects = []

        if self.batch_id:
            batch_response = self.get_response_batch_inference(self.batch_id)
            if batch_response:
                for response in batch_response:
                    generated_json_object = response['response']['body']['choices'][0]['message']['content']
                    generated_statement_json_objects.append(generated_json_object)
                return generated_statement_json_objects
            else:
                logger.info("No results to save. Please check the status of the batch job.")
                return None
            
        elif self.batch_job:
            chat_prompts = []
            chat_histories = []
            for original_statement_json_object in original_statement_json_objects:
                chat_prompt = self.get_chat_prompt(original_statement_json_object)
                samples = self.get_samples(get_json_object_map, original_statement_json_object["matches_hash"])
                chat_history = self.get_chat_history(samples)
                chat_prompts.append(chat_prompt)
                chat_histories.append(chat_history)
            batch_id = self.send_batch_inference(original_statement_json_objects, chat_prompts, chat_histories, max_tokens=9000)
            logger.info(f"Batch job submitted with ID: {batch_id}")
            logger.info("Please check the status of the batch job later.")
            return None
        
        else:
            for matches_hash in tqdm(get_json_object_map.keys(), desc="Extracting", unit="statement"):
                original_statement_json_object = get_json_object_map[matches_hash]
                samples = self.get_samples(get_json_object_map, matches_hash)
                chat_prompt = self.get_chat_prompt(original_statement_json_object)
                chat_history = self.get_chat_history(samples)
                response = self.get_response_single_inference(chat_prompt, chat_history, max_tokens=9000)
                generated_json_object = response.choices[0].message.content
                generated_statement_json_objects.append(generated_json_object)
            return generated_statement_json_objects

    def get_results_df(self, generated_statement_json_objects):
        if self.batch_id:
            # Read the jsonl file containing the original statements
            original_json_statements_path = OUTPUT_DIR / "batches" / self.batch_id / "original_statements.jsonl"
            with open(original_json_statements_path, "r") as f:
                original_statement_json_objects = [json.loads(line) for line in f]
        else:
            original_statement_json_objects = self.get_input_json_objects()
        input_texts = self.get_input_texts(original_statement_json_objects)
        original_statements = [stmt_from_json(stmt_json) for stmt_json in original_statement_json_objects]

        extracted_statements = []
        for generated_statement_json_object in generated_statement_json_objects:
            try: 
                if self.structured_output: # output is a json object with property 'statements' which is a list of statements
                    stmts_json = json.loads(generated_statement_json_object)['statements']
                    stmts_json = [post_process_extracted_json(stmt_json) for stmt_json in stmts_json]
                    stmts_indra = [stmt_from_json(stmt_json) for stmt_json in stmts_json]
                    extracted_statements.append(str(stmts_indra))
                else:   # output is a single json object of a statement
                    stmt_json = json.loads(generated_statement_json_object)                    
                    stmt_json = post_process_extracted_json(stmt_json)
                    stmt_indra = stmt_from_json(stmt_json)
                    extracted_statements.append(str(stmt_indra))
            except (JSONDecodeError, IndexError, TypeError) as e:
                logger.error(f"Error extracting statement: {e}")
                extracted_statements.append(f"Error: {e}")
        result_df = pd.DataFrame(
            {
                "input_text": input_texts,
                "original_statement_json": original_statement_json_objects,
                "generated_statement_json": generated_statement_json_objects,
                "original_statement": original_statements,
                "generated_statement": extracted_statements
            }
        )
        return result_df

    def save_results_df(self, results_df):
        if self.batch_id and self.output_file == OUTPUT_DEFAULT:
            self.output_file = OUTPUT_DIR / "batches" / self.batch_id / "extraction_results.tsv"
        else:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.output_file, sep="\t", index=False)
        logger.info(f"Results saved to {self.output_file}")
