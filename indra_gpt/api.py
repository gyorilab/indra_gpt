import logging
from time import sleep

import openai

logger = logging.getLogger(__name__)


def run_openai_chat(
    prompt: str,
    model="gpt-3.5-turbo",
    max_tokens=1,
    retry_count=3,
    strip=True,
    debug=False,
):
    """Run OpenAI to check if the check sentence implies the check statement

    Parameters
    ----------
    prompt :
        The prompt to send to the chat
    model :
        The model to use. The default is the gpt-3.5-turbo model.
    max_tokens :
        The maximum number of tokens to generate for chat completion. One
        token is roughly one word in plain text, however it can be more per
        word in some cases.
    retry_count :
        The number of times to retry the request if it fails. The default is
        3. After the retry count is reached, the function will raise an
        exception.
    strip :
        If True, the function will strip the response of whitespace and
        punctuations.
    debug :
        If True, the function will print the full response from
        openai.Completion/CharCompletion.create(). The default is False.

    Returns
    -------
    :
        The response from OpenAI as a string
    """

    def _get_response(resp) -> str:
        if model == "gpt-3.5-turbo":
            choice = resp["choices"][0]["message"]["content"]
        else:  # text-davinci-003
            choice = resp["choices"][0]["text"]

        if resp["choices"][0]["finish_reason"] == "length" and \
                not choice.lower().startswith(("yes", "no")):
            logger.warning(
                "OpenAI response was truncated. Likely due to token "
                "constraints. Consider increasing the max_tokens parameter."
            )

        # Remove whitespace and trailing punctuations
        if strip:
            choice = choice.strip().rstrip(".,!")

        return choice

    options = {}
    # For gpt-3.5-turbo chat mode:
    # https://platform.openai.com/docs/api-reference/chat/create
    if model == "gpt-3.5-turbo":
        options["messages"] = [{"role": "user", "content": prompt}]
        api_class = openai.ChatCompletion
    else:  # text-davinci-003
        options["prompt"] = prompt
        api_class = openai.Completion

    # Retry the request if it fails
    retry_count = max(retry_count, 1)
    response = None
    for i in range(retry_count):
        try:
            response = api_class.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                **options,
            )
            break
        except Exception as e:
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

    resp_str = _get_response(response)
    if resp_str == "":
        logger.warning("OpenAI returned an empty response. See full response "
                       "below for details.")
        print(f"Response:\n---------\n{response}\n---------\n\n")

    return resp_str
