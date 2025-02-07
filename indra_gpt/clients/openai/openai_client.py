import logging
from time import sleep
import json
from pathlib import Path
import time
from openai import OpenAI
from indra.config import IndraConfigError, get_config
from indra_gpt.resources.constants import OUTPUT_DIR

logger = logging.getLogger(__name__)


try:
    api_key = get_config("OPENAI_API_KEY", failure_ok=False)
    organization = get_config("OPENAI_ORG")
except IndraConfigError as err:
    raise KeyError(
        "Please set OPENAI_API_KEY in the environment or in the indra config."
    ) from err
client = OpenAI(api_key=api_key, organization=organization)


def run_openai_chat(
    prompt: str,
    model="gpt-4o-mini",
    max_tokens=1,
    retry_count=3,
    strip=True,
    debug=False,
    chat_history=None,
):
    """Run OpenAI to check if the check sentence implies the check statement

    Parameters
    ----------
    prompt : str
        The prompt to send to the chat
    model : str
        The model to use. The default is the gpt-4-0613 model.
    max_tokens : int
        The maximum number of tokens to generate for chat completion. One
        token is roughly one word in plain text, however it can be more per
        word in some cases.
    retry_count : int
        The number of times to retry the request if it fails. The default is
        3. After the retry count is reached, an exception will be raised.
    strip : bool
        If True, the function will strip the response of whitespace and
        punctuations.
    debug : bool
        If True, the function will print the full response from
        openai.Completion/CharCompletion.create(). The default is False.
    chat_history : list
        A list of chat history to send to the chat. The default is None.
        This can be used to send example prompts and responses to the chat
        to improve the quality of the response.

    Returns
    -------
    :
        The response from OpenAI as a string
    """
    # chat mode documentation:
    # https://platform.openai.com/docs/api-reference/chat/create
    if chat_history is not None:
        messages = chat_history
    else:
        messages = []
    messages.append({"role": "user", "content": prompt})

    retry_count = max(retry_count, 1)
    response = None
    for i in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                messages=messages,
            )
            break
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

    reply = response.choices[0].message.content

    if response.choices[0].finish_reason == "length":
        logger.warning(
            "OpenAI response was truncated. Likely due to token "
            "constraints. Consider increasing the max_tokens parameter."
        )

    # Remove whitespace and trailing punctuations 
    if strip:
        reply = reply.strip().rstrip(".,!")
    if reply == "":
        logger.warning(
            "OpenAI returned an empty reply. See full API response below for details."
        )
        print(f"Response:\n---------\n{response}\n---------\n\n")

    return reply


def run_openai_chat_batch(prompts, chat_histories, model, max_tokens):

    batch_requests = []

    for i, (prompt, history) in enumerate(zip(prompts, chat_histories)):
        prompt_openai_format = [{"role": "user", "content": prompt}]
        messages = history + prompt_openai_format
        batch_requests.append(
            {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                     "model": model,
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
            
    # Upload the batch file
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

    # Write the batch requests to a file
    batch_input_file_path = batch_dir_path / "batch_input.jsonl"
    with open(batch_input_file_path, "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    # Also add a txt file with when this batch was created
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(batch_dir_path / "metadata.txt", "w") as f:
        f.write(f"Batch created at time: {current_time}\n")
        f.write(f"Batch ID: {batch_id}\n")
    return batch_id


def get_batch_replies(batch_id):
    replies = []
    try:
        batch = client.batches.retrieve(batch_id)
        logger.info("Fetched data for batch id {batch_id})")
        status = batch.to_dict()['status']
        if status == "completed":
            batch_output_file_id = batch.to_dict()['output_file_id']
            # Assuming file_response.text contains the string with multiple JSON objects
            file_response_text = client.files.content(batch_output_file_id).text
            # Split the response into lines and load each line as a JSON object
            response_data = []
            for line in file_response_text.splitlines():
                if line.strip():  # Skip empty lines
                    response_data.append(json.loads(line))
            # Now `response_data` is a list of dictionaries
            replies = []
            for entry in response_data:
                replies.append(entry['response']['body']['choices'][0]['message']['content'])
        else:
            logger.info(f"Batch job info: {batch}")
            replies = None
    except Exception as e:
        logger.error(f"Error getting batch replies: {e}")
        replies = None
    return replies


"""This module contains functions for extracting statements from text by
feeding the full json schema to ChatGPT."""

import json
import logging
import random
from json import JSONDecodeError
from pathlib import Path

import pandas as pd
from indra.statements.io import stmt_from_json
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from indra_gpt.api import run_openai_chat, run_openai_chat_batch, get_batch_replies
from indra_gpt.resources.constants import JSON_SCHEMA, OUTPUT_DEFAULT, INPUT_DEFAULT
import openai

from indra_gpt.util import trim_stmt_json, post_process_extracted_json

logger = logging.getLogger(__name__)


def chat_prompt_and_history(stmt_json_examples, evidence_text):
    json_schema_string = json.dumps(JSON_SCHEMA)  # converting json schema to a string

    # full prompt including schema
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

    # Add chat history
    # variables to feed the chat history:
    stmt_json_1 = stmt_json_examples[0]  #
    # first example json in the training
    # dataframe
    ev_text_1 = stmt_json_1["evidence"][0]["text"]  # first example sentence
    # extracted from the example json
    stmt_json_2 = stmt_json_examples[1]  #
    # second example json in the
    # training dataframe
    ev_text_2 = stmt_json_2["evidence"][0]["text"]  # second example sentence
    # extracted from the example json

    # create the chat history, including the prompt with the schema,
    # the first example json to feed chatGPT as a sample, the reduced prompt
    # without schema, the second example json to feed chatGPT as a sample
    history = [
        {"role": "user", "content": PROMPT + ev_text_1},  # prompt with schema
        {"role": "assistant", "content": json.dumps(stmt_json_1)},
        # first stmt json example
        {
            "role": "user",
            "content": PROMPT_reduced + ev_text_2,
        },  # prompt without schema
        {"role": "assistant", "content": json.dumps(stmt_json_2)},
        # second stmt json example
    ]

    # Prompt to feed chatGPT, including the reduced prompt without schema
    prompt = PROMPT_reduced + evidence_text["evidence"][0]["text"]
    # format the main prompt to ask chatGPT without being fed sample
    # data to only include the reduced prompt without the json schema +
    # the main sentence inputted into the gpt_stmt_json function

    return prompt, history


def gpt_stmt_json(stmt_json_examples, evidence_text, model: str, debug: bool = False):
    """Prompt chatGPT to generate a statement json given a sentence

    This function feeds chatGPT a prompt that includes the json schema of a statement
    json and question to generate a json object for a sentence using information in
    the schema and two previous extraction examples.

    Parameters
    ----------
    stmt_json_examples : list
        A list of two statement json objects to feed to chatGPT as examples
    evidence_text : dict
        A dictionary containing the evidence text to feed to chatGPT
    model : str
        The openai chat model to use.
    debug : bool
        If True, the function will print requests sent to and responses received
        from the API, respectively.

    Returns
    -------
    chat_gpt_json : str
        The response from OpenAI as a string
    """

    prompt, history = chat_prompt_and_history(stmt_json_examples, evidence_text)

    chat_gpt_json = run_openai_chat(
        prompt,
        model=model,
        chat_history=history,
        max_tokens=9000,
        strip=False,
        debug=debug,
    )
    return chat_gpt_json


def gpt_stmt_json_batch(sample_lists, json_objects, model: str, debug: bool = False):
    prompts = []
    histories = []
    for sample_list, json_object in zip(sample_lists, json_objects):
        prompt, history = chat_prompt_and_history(sample_list, json_object)
        prompts.append(prompt)
        histories.append(history)
    batch_id = run_openai_chat_batch(
        prompts,
        chat_histories=histories,
        model=model,
        max_tokens=9000
    )
    return batch_id
        

def create_json_object_map(json_object_list):
    json_object_map = {}
    for stmt_json in json_object_list:
        mh = stmt_json["matches_hash"]
        json_object_map[mh] = trim_stmt_json(stmt_json)
    return json_object_map


def main(
    json_file,
    model: str,
    n_iter: int,
    output_file: Path,
    verbose: bool = False,
    batch_jobs: bool = False,
    batch_id: str = None
):

    """Function to run above operations on inputted training dataframe of json objects

    Parameters
    ----------
    json_file : str
        path to the json file containing statement json objects
    model : str
        The openai chat model to use
    n_iter : int
        The number of statements to ask chatGPT to extract
    output_file : Path
        The path to save the output tsv file
    verbose : bool
        If True, the function will print requests sent to and responses received
        from the API, respectively.
    batch_jobs : bool
        If True, the script will run in batch job mode, processing groups of requests asynchronously to OpenAI API.
    batch_id : str
        Provide a string of the batch job ID to retrieve the results of a batch job.
    """
    
    if batch_id:
        replies = get_batch_replies(batch_id)
        
        if replies is None:
            # If replies is None, the batch job is not completed yet, or there was an error related to API, etc.
            # In this case, we do not save the output file and just return nothing. 
            logger.error(
                "Error retrieving batch replies. Please check the batch job ID or try again later."
            )
            return

        batches_dir_path = Path(__file__).resolve().parent.parent / "batches"
        batch_dir_path = batches_dir_path / batch_id

        # Check if the batch directory exists
        if not batch_dir_path.exists():
            raise FileNotFoundError(f"Batch directory for batch_id {batch_id} does not exist. Please check the batch ID and try again.")

        # Else save the output file
        batch_output_file_path = batch_dir_path / "batch_output.jsonl"

        with open(batch_output_file_path, "w") as f:
            for reply in replies:
                f.write(json.dumps(reply) + "\n")
        
        logger.info(f"Batch output saved to {batch_output_file_path}")
        return

    outputs = []  # list of every output by chatGPT
    sentences = []  # list of the sentences fed to the prompt
    statements = []  # list of json statements returned from outputted json
    # object by chatGPT (when applicable to a sentence and its outputted json
    # object)

    with open(json_file, "r") as f:
        json_content = json.load(f)

    # assign first N json objects to json_object_list
    if len(json_content) < n_iter:
        logger.warning(f"Number of iterations is greater than the number of statements "
                       f"in the file. All {n_iter} statements will be processed.")
    json_object_list = json_content[:n_iter]
    json_object_map = create_json_object_map(json_object_list)

    if batch_jobs:
        sentences = []
        sample_lists = []
        json_objects = []
        statements = []

        for matches_hash in json_object_map:
            json_object = json_object_map[matches_hash]
            json_objects.append(json_object)

            sentence = json_object["evidence"][0]["text"]
            sentences.append(sentence)

            sequence = list(set(json_object_map.keys()) - {matches_hash})
            sample_hashes = random.sample(sequence, 2)
            sample_list = [json_object_map[h] for h in sample_hashes]
            sample_lists.append(sample_list)

        batch_id = gpt_stmt_json_batch(
            sample_lists, json_objects, model=model, debug=verbose
        )
        logger.info(f"Batch job submitted with ID: {batch_id}")


    else:
        # Loop through each json object in json_object_list
        for matches_hash in tqdm(json_object_map, desc="Extracting", unit="statement"):
            json_object = json_object_map[matches_hash]

            # Uncomment to use when debugging
            # sample_list = [json_object_list[0],json_object_list[1]]

            # Take two statement jsons at random to feed to the chat history for each
            # iteration
            sequence = list(set(json_object_map.keys()) - {matches_hash})
            sample_hashes = random.sample(sequence, 2)
            sample_list = [json_object_map[h] for h in sample_hashes]

            sentence = json_object["evidence"][0]["text"]
            sentences.append(sentence) # get the sentence from the current json
            # object and append to the sentences list

            try:
                with logging_redirect_tqdm():
                    response = gpt_stmt_json(
                        sample_list, json_object, model=model, debug=verbose
                    )
                    # run gpt_stmt_json function on the randomly sampled list and
                    # current json object
            except openai.BadRequestError as e:
                error_text = f"OpenAI error: {e}" # if chatGPT fails to create a
                # response
                outputs.append(error_text) # append error message
                statements.append(error_text) # append error message
                continue

            outputs.append(response) # append chatGPT's response to outputs list

            """
            Some generated json objects are able to take in the
            json.loads and stmt_from_json functions to return a statement
            json. Some are not. Use the try/except method to extract statement
            jsons from the generated json objects for the ones that work.
            Append them to the list of all statements extracted. If it doesn't
            work on a generated json object, just append that there was an
            error loading the statement.
            """
            try:
                # Here run json.loads on the response separately from stmt_from_json
                # to clearly see if the response is a valid json object

                # JSON loads the response
                stmt_json = json.loads(response)

                # Run post-processing on the extracted json
                stmt_json = post_process_extracted_json(stmt_json)

                # Extract the INDRA statement object from the json
                stmt_n = stmt_from_json(stmt_json)

                # Append the str extracted statement to statements list (while not saved
                # as a valid statement in the output TSV file, this gives a quick
                # view of the statement extracted when loading the file later)
                statements.append(str(stmt_n))
            except (JSONDecodeError, IndexError, TypeError) as e:
                # TypeError can happen when the response contains more than one
                # statement json object and the post-processing function tries to
                # access the "type" key in the response from the list of statement
                # json objects.
                # Todo: handle multiple statement json objects in a response
                statements.append(f"Error: {e}")

        # Save sentences, json_object_list, outputs, and statements as a pandas dataframe
        df = pd.DataFrame(
            {
                "sentence": sentences,
                "input": [json.dumps(js) for js in json_object_list],
                "generated_json_object": outputs,
                "extracted_statement": statements,
            }
        )
        # Save dataframe as tsv file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, sep="\t", index=False)
        logger.info(f"Output written to {output_file}")
