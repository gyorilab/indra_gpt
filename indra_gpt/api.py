import logging
from time import sleep

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

def run_openai_chat_batch(prompts, histories, model, max_tokens):

    batch_requests = []

    for i, prompt, history in enumerate(zip(prompts, histories)):
        batch_requests.append(
            {"custom_id": f"request-{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": model, "messages": history.extend(prompt), "max_tokens": max_tokens}}
        )

    import json
    with open("./tmp/batch_input.jsonl", "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")
            
    # Upload the batch file
    batch_input_file = client.files.create(
        file=open("./tmp/batch_input.jsonl", "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    batch = client.batches.retrieve('batch_67926dab0a408190857a2a4aa833fa1e')
    batch_status = batch.to_dict()
    batch_output_file_id = batch_status['output_file_id']
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
    return replies