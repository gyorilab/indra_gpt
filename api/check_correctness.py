import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tqdm
import random
import openai
from indra.assemblers.english import EnglishAssembler
from indra.statements.io import stmts_from_json_file


try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ["OPENAI_ORG"]
except KeyError as e:
    raise KeyError(
        "Please set the OPENAI_API_KEY and OPENAPI_ORG " "environment " "variable."
    ) from e


logger = logging.getLogger(__name__)


HERE = Path(__file__).parent
LOCAL_FILES = HERE.parent.joinpath("local_data")
LOCAL_FILES.mkdir(exist_ok=True)
curation_training_data = LOCAL_FILES.joinpath("training_data.tsv")

default_prompt_template = """The following sentences are paired with statements that are implied from their sentence:

{examples}

Does the following sentence "{check_sentence}" imply this statement "{check_eng_stmt}"?
Please answer with just Yes or No."""
old_prompt = (
    "You need to help me verify if a sentence I give you implies "
    "a statement I give you. Provide a correct answer with a "
    'simple yes or no. Sentence: "{check_sentence}" Statement: '
    '"{check_eng_stmt}" Answer:'
)


def get_create_training_set(
    curations_file: str = None, statement_json_file: str = None, refresh: bool = False
) -> pd.DataFrame:
    if curation_training_data.exists():
        return pd.read_csv(curation_training_data, sep="\t")

    # Create the training set
    assert curations_file is not None and statement_json_file is not None, (
        f"Please provide the curations and statement json file if "
        f"pre-generated training data is not available at "
        f"{curation_training_data}"
    )
    logger.info("Loading curations")
    curs = json.load(open(curations_file, "r"))
    logger.info("Loading statements")
    stmts = stmts_from_json_file(statement_json_file)
    stmts_by_hash = {s.get_hash(): s for s in stmts}

    # Loop the curations, get the corresponding statement with evidence and
    # extend the curation with the evidence text and english assembled
    # statement
    curation_data = []
    for cur in curs:
        stmt = stmts_by_hash[cur["pa_hash"]]
        ev = [e for e in stmt.evidence if e.get_source_hash() == cur["source_hash"]][0]
        cur["text"] = ev.text
        eng_stmt = EnglishAssembler([stmt]).make_model()
        cur["english_stmt"] = eng_stmt
        curation_data.append(cur)

    # Save the training data
    df = pd.DataFrame(curation_data)
    df.to_csv(curation_training_data, sep="\t", index=False)
    return df


def generate_examples_by_tag(curation_df: pd.DataFrame, tag: str, n_examples: int = 5):
    """Generate pairs of sentence-english statement for a given tag

    Parameters
    ----------
    curation_df :
        The curation dataframe
    tag :
        The tag to use e.g. "correct", "no_relation", "grounding"
    n_examples :
        The number of examples to generate for the tag. The default is 5. If
        -1 is given, all examples will be returned.

    Returns
    -------
    examples :
        A list of tuples with (sentence, english_stmt)
    """
    if n_examples == -1:
        kwargs = {"frac": 1.0}
    else:
        kwargs = {"n": max(abs(n_examples), 1)}

    examples = []
    for row in (
        curation_df[["text", "english"]][curation_df["tag"] == tag]
        .sample(**kwargs)
        .values
    ):
        examples.append(tuple(row))
    return examples


def run_openai_chat(
    examples,
    check,
    prompt_template=default_prompt_template,
    model="gpt-3.5-turbo",
    max_tokens=1,
):
    """Run OpenAI to check if the check sentence implies the check statement

    Parameters
    ----------
    examples :
        A list of tuples with (sentence, english_stmt) for the examples to
        give.
    check :
        A tuple with (sentence, english_stmt) for the check sentence and
        statement
    prompt_template :
        The prompt template to use. If the default is used, the examples
        will be used to fill in the template. If a custom template is used,
        only the check sentence and statement will be used to fill in the
        template.
    model :
        The model to use. The default is the gpt-3.5-turbo model.
    max_tokens :
        The maximum number of tokens to generate for chat completion. One
        token is roughly one word in plain text, however it can be more per
        word in some cases.

    """

    def _get_response(resp):
        if model == "gpt-3.5-turbo":
            return resp["choices"][0]["message"]["content"].strip()
        else:  # text-davinci-003
            return resp["choices"][0]["text"].strip()

    def _generate_prompt(ex_list, ch):
        examples_str = ""
        for i, (sentence, eng_stmt) in enumerate(ex_list):
            examples_str += (
                f'Sentence{i+1}: "{sentence}"\nStatement{i+1}: "{eng_stmt}"\n'
            )

        if prompt_template == default_prompt_template:
            prmt = default_prompt_template.format(
                examples=examples_str, check_sentence=ch[0], check_eng_stmt=ch[1]
            )
        else:
            prmt = prompt_template.format(check_sentence=ch[0], check_eng_stmt=ch[1])
        return prmt

    options = {}
    # todo: play around with the prompt. Add examples of correct and maybe
    #  also incorrect
    # For gpt-3.5-turbo chat mode:
    # https://platform.openai.com/docs/api-reference/chat/create
    prompt = _generate_prompt(examples, check)
    options["messages"] = [{"role": "user", "content": prompt}]
    api_class = openai.ChatCompletion

    # For text-davinci-003 chat mode:
    # options["prompt"] = prompt % (ev.text, eng_stmt)
    # api_class = openai.Completion
    response = api_class.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        **options,
    )

    logger.info("Got response: ", response)
    return _get_response(response)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        logger.error("Please provide the curations file and the statement json file")
        sys.exit(1)

    curations = args[0]
    statement_jsons_file = args[1]

    # 1. Get the dataframe of statements, evidence text and curation tags
    cur_df = get_create_training_set(
        curations_file=curations, statement_json_file=statement_jsons_file
    )

    # 2. Get two correct examples
    example_list = generate_examples_by_tag(cur_df, n_examples=2, tag="correct")

    # 3. Get one example to check at random
    checker_dict = cur_df.sample(1).to_dict(orient="records")[0]
    checker = (checker_dict["text"], checker_dict["english"])

    # 4. Run the chat completion
    choice = run_openai_chat(example_list, checker, max_tokens=2)

    # 5. Get the response and the tag and check if the response is correct
    print(f'Check text:\n"{checker[0]}"\n\nCheck statement:\n{checker[1]}\n\n')
    print(
        f"Output\n------\nChoice (correct? Yes/No): {choice or '(None)'}, "
        f"Originally "
        f"tagged as: {checker_dict['tag']}"
    )

# curs_sample = random.sample(curs, 100)
# responses = []
#
# for cur in tqdm.tqdm(curs_sample):
#     stmt = stmts_by_hash[cur["pa_hash"]]
#     ev = [e for e in stmt.evidence if e.get_source_hash() == cur["source_hash"]][0]
#     eng_stmt = EnglishAssembler([stmt]).make_model()
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt=prompt % (ev.text, eng_stmt),
#         temperature=0,
#         max_tokens=1,
#         top_p=1.0,
#         frequency_penalty=0.0,
#         presence_penalty=0.0,
#     )
#     responses.append(response)
#
# choices = [r["choices"][0]["text"].strip() for r in responses]
# tags = [c["tag"] for c in curs_sample]
# confusion = defaultdict(int)
# for choice, tag in zip(choices, tags):
#     if choice == "Yes" and tag == "correct":
#         confusion[("Yes", "correct")] += 1
#     elif choice == "Yes" and tag != "correct":
#         confusion[("Yes", "incorrect")] += 1
#     elif choice == "No" and tag == "correct":
#         confusion[("No", "correct")] += 1
#     elif choice == "No" and tag != "correct":
#         confusion[("No", "incorrect")] += 1
