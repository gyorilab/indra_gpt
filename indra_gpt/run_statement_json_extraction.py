"""This module contains functions for extracting statements from text
feeding the full json schema using ChatGPT."""

import json
import random

import pandas as pd
from indra.statements.io import stmt_from_json
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from api import run_openai_chat
from constants import JSON_SCHEMA


def gpt_stmt_json(stmt_json_examples, evidence_text):
    """
    function to feed chatGPT a prompt including the full json schema and
    ask it to generate a json object for a sentence using information in
    the schema

    :param stmt_json_examples:
    :param evidence_text:
    :return:
    """
    ############################# PROMPT ENGINEERING ###########################

    json_schema_string = json.dumps(JSON_SCHEMA)  # converting json schema to a string
    # print(len(json))

    invalid_pieces = '"$ref": "#/definitions/Agent"'  # store any invalid
    # syntax in the json schema in the variable invalid_pieces to ask
    # chatGPT to remove them from its generated json object

    # full prompt including schema
    PROMPT = (
        "Read the following JSON schema for a statement "
        "object:\n\n```json\n"
        + json_schema_string.replace("{", "{{").replace("}", "}}")
        + "\n```\n\nExtract the relation from the following sentence and put it in a JSON object matching the schema above. The JSON object needs to be able to pass a validation against the provided schema. Remove "
        + invalid_pieces
        + ". Only respond with "
        "the JSON object.\n\nSentence: "
    )

    # reduced prompt not including schema
    PROMPT_reduced = (
        "Extract the relation from the following sentence and put it in a JSON object matching the schema above. The JSON object needs to be able to pass a validation against the provided schema. Remove "
        + invalid_pieces
        + ". Only respond with "
        "the JSON object.\n\nSentence: "
    )

    ############################## HISTORY ############################

    # variables to feed the chat history:
    stmt_json_1 = stmt_json_examples[0]  # first example json in the training
    # dataframe
    ev_text_1 = stmt_json_1["evidence"][0]["text"]  # first example sentence
    # extracted from the example json
    stmt_json_2 = stmt_json_examples[1]  # second example json in the
    # training dataframe
    ev_text_2 = stmt_json_2["evidence"][0]["text"]  # second example sentence
    # extracted from the example json

    # create the chat history, including the prompt with the schema,
    # the first example json to feed chatGPT as a sample, the reduced prompt
    # without schema, the second example json to feed chatGPT as a sample
    history = [
        {"role": "user", "content": PROMPT + ev_text_1},  # Prompt with schema
        {"role": "assistant", "content": str(stmt_json_1)},  # first stmt json example
        {
            "role": "user",
            "content": PROMPT_reduced + ev_text_2,
        },  # prompt without schema
        {"role": "assistant", "content": str(stmt_json_2)},  # second stmt json example
    ]

    ################### RUN PROMPT TO ASK CHATGPT #####################

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
    )  # use run_openai_chat
    # function on prompt, specifying model and max_tokens parameters as
    # needed
    return chat_gpt_json  # return chatGPT's response


def process_statement_json(statement):
    # TODO Bihan
    return statement


# main function to run on the inputted traing dataframe of json objects
def main(json_file):
    # main function on
    # path to training dataset of sample json objects (change
    # path as needed)
    outputs = []  # list of every output by chatGPT
    sentences = []  # list of the sentences fed to the prompt

    statements = []  # list of json statements returned from outputted json
    # object by chatGPT (when applicable to a sentence and its outputted json
    # object)

    with open(json_file, "r") as f:
        json_content = json.load(f)

    json_object_list = json_content[:50]  # append entire file of sample
    json_object_list = [process_statement_json(stmt) for stmt in json_object_list]
    # json
    # objects to the list json_object_list

    for json_object in tqdm(
        json_object_list, desc="Extracting", unit="statement", unit_scale=True
    ):
        sample_list = random.sample(json_object_list, 2)

        sentence = json_object["evidence"][0]["text"]
        sentences.append(sentence)

        try:
            with logging_redirect_tqdm():
                response = gpt_stmt_json(sample_list, json_object)
        except Exception as e:
            error_text = f"OpenAI error: {e}"
            outputs.append(error_text)
            statements.append(error_text)
            continue

        outputs.append(response)

        # Some generated json objects are able to take in the
        # json.loads and stmt_from_json functions to return a statement
        # json. Some are not. Use the try/except method to extract statement
        # jsons from the generated json objects for the ones that work.
        # Append them to the list of all statements extracted. If it doesn't
        # work on a generated json object, just append that there was an
        # error loading the statement.
        try:
            json_str = json.loads(response)
            stmt_n = stmt_from_json(json_str)  # INDRA statement object
            statements.append(stmt_n)
        except Exception as e:
            statements.append(f"Error: {e}")

    df = pd.DataFrame(
        {
            "sentence": sentences,
            "input": json_object_list,
            "generated_json_object": outputs,
            "extracted_statement": statements,
        }
    )
    df.to_csv("statement_json_extraction_results.tsv", sep="\t", index=False)

    print("First five json objects generated by chatGPT: \n\n" + str(outputs[:5]))  #
    # only print the first 5 results
    print("\n\nThe actual json objects: \n\n" + str(json_object_list[:5]))
    print("Done.")  # print done to know when main function has finished


if __name__ == "__main__":
    main("indra_benchmark_corpus_all_correct.json")
