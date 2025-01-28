import logging
from time import sleep
import json
from pathlib import Path
import time
from openai import OpenAI
from indra.config import IndraConfigError, get_config

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
        messages = history + prompt
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
    batches_dir_path = Path(__file__).resolve().parent.parent / "batches"  
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