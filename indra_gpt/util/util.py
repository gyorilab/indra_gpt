from indra.statements import get_all_descendants, Statement

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


def trim_stmt_json(stmt):
    """Function to get rid of irrelevant parts of the indra statement json

    Parameter
    ---------
    stmt : dict
        The indra statement json object

    Returns
    -------
    dict
        The indra statement json object with irrelevant parts removed
    """

    stmt["evidence"] = [{"text": stmt["evidence"][0].get("text")}]

    del stmt["id"]
    del stmt["matches_hash"]

    if 'supports' in stmt:
        del stmt['supports']
    if 'supported_by' in stmt:
        del stmt['supported_by']

    return stmt


def post_process_extracted_json(gpt_stmt_json):
    """Function to post process the extracted json from chatGPT

    Parameters
    ----------
    gpt_stmt_json : dict
        The extracted json from chatGPT

    Returns
    -------
    dict
        The post processed json or if there is a KeyError, the original json
    """
    try:
        stmt_type = gpt_stmt_json["type"]
        mapped_type = stmt_mapping.get(stmt_type.lower(), stmt_type)
        gpt_stmt_json["type"] = mapped_type
    except KeyError:
        pass

    return gpt_stmt_json


def _get_statement_mapping():
    stmt_classes = get_all_descendants(Statement)
    mapping = {stmt_class.__name__.lower(): stmt_class.__name__ for stmt_class in stmt_classes}
    return mapping


stmt_mapping = _get_statement_mapping()
