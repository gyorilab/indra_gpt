import logging
from time import sleep

import openai

logger = logging.getLogger(__name__)


def run_openai_chat(
    prompt: str,
    model="gpt-4-0613",
    max_tokens=1,
    retry_count=3,
    strip=True,
    debug=False,
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

    Returns
    -------
    :
        The response from OpenAI as a string
    """
    # For chat mode documentation:
    # https://platform.openai.com/docs/api-reference/chat/create
    messages = [
        {"role": "user", "content": prompt}
    ]

    retry_count = max(retry_count, 1)
    response = None
    for i in range(retry_count):
        try:
            response = openai.ChatCompletion.create(
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
                logger.warning(f"Request failed with error: {e}. Retrying "
                               f"after 5 seconds.")
                sleep(5)
            else:
                raise e

    if debug:
        logger.info(
            f"Prompt:\n-------\n{prompt}\n-------\n"
            f"Response:\n---------\n{response}\n---------\n\n"
        )
    if response is None:
        raise RuntimeError("No response from OpenAI")

    reply = response["choices"][0]["message"]["content"]

    if response["choices"][0]["finish_reason"] == "length":
        logger.warning(
            "OpenAI response was truncated. Likely due to token "
            "constraints. Consider increasing the max_tokens parameter."
        )

    # Remove whitespace and trailing punctuations
    if strip:
        reply = reply.strip().rstrip(".,!")
    if reply == "":
        logger.warning("OpenAI returned an empty response. See full response "
                       "below for details.")
        print(f"Response:\n---------\n{response}\n---------\n\n")

    return reply
