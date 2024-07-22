"""This module contains functions for extracting statements from text by
feeding the full json schema to ChatGPT."""

# import libraries
import json
import random
import pandas as pd
from indra.statements.io import stmt_from_json
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from indra_gpt.api import run_openai_chat
from indra_gpt.constants import JSON_SCHEMA
import openai


def gpt_stmt_json(stmt_json_examples, evidence_text):
    """
    function to feed chatGPT a prompt including the full json schema and
    ask it to generate a json object for a sentence using information in
    the schema

    :param stmt_json_examples:
    :param evidence_text:
    :return response:
    """

    # PROMPT ENGINEERING

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

    chat_gpt_json = run_openai_chat(
        prompt,
        model="gpt-3.5-turbo-16k",
        chat_history=history,
        max_tokens=9000,
        strip=False,
        debug=False,
    )
    return chat_gpt_json


def trim_stmt_json(stmt):
    """
    function to get rid of irrelevant parts of the indra statement
    dictionary

    :param stmt:
    :return stmt:

    """

    stmt["evidence"] = [{"text": stmt["evidence"][0].get("text")}]

    del stmt["id"]
    del stmt["matches_hash"]

    if 'supports' in stmt:
        del stmt['supports']
    if 'supported_by' in stmt:
        del stmt['supported_by']

    return stmt


def main(json_file):

    """
    function to run above operations on inputted training dataframe of
    json objects

    :param json_file:
    :output tsv file:

    """

    outputs = []  # list of every output by chatGPT
    sentences = []  # list of the sentences fed to the prompt
    statements = []  # list of json statements returned from outputted json
    # object by chatGPT (when applicable to a sentence and its outputted json
    # object)

    with open(json_file, "r") as f:
        json_content = json.load(f)

    # assign first 50 json objects to json_object_list
    json_object_list = json_content[:50]

    # run processing function on each json object to trim it down
    json_object_list = [trim_stmt_json(stmt) for stmt in json_object_list]

    # Loop through each json object in json_object_list
    for json_object in tqdm(json_object_list, desc="Extracting", unit="statement"):

        # Uncomment to use when debugging
        # sample_list = [json_object_list[0],json_object_list[1]]

        # Take two statement jsons at random to feed to the chat history for each
        # iteration
        sample_list = random.sample(json_object_list, 2)

        sentence = json_object["evidence"][0]["text"]
        sentences.append(sentence) # get the sentence from the current json
        # object and append to the sentences list

        try:
            with logging_redirect_tqdm():
                response = gpt_stmt_json(sample_list, json_object) # run
                # gpt_stmt_json function on the randomly sampled list and
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

            # Extract the INDRA statement object from the json
            stmt_n = stmt_from_json(stmt_json)

            # Append the str extracted statement to statements list (while not saved
            # as a valid statement in the output TSV file, this gives a quick
            # view of the statement extracted when loading the file later)
            statements.append(str(stmt_n))
        except IndexError as e:
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
    df.to_csv("statement_json_extraction_results.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main("indra_benchmark_corpus_all_correct.json") # run main function here
