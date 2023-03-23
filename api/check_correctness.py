import json
import logging
import os
import sys
from pathlib import Path

import openai
import pandas as pd
import requests

from indra.assemblers.english import EnglishAssembler
from indra.assemblers.indranet.assembler import NS_PRIORITY_LIST
from indra.statements.io import stmts_from_json_file

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ["OPENAI_ORG"]
except KeyError as err:
    raise KeyError(
        "Please set the OPENAI_API_KEY and OPENAPI_ORG " "environment " "variable."
    ) from err


logger = logging.getLogger(__name__)


HERE = Path(__file__).parent
LOCAL_FILES = HERE.parent.joinpath("local_data")
LOCAL_FILES.mkdir(exist_ok=True)
curation_training_data = LOCAL_FILES.joinpath("training_data.tsv")

default_prompt_template = """The following sentences are paired with statements that are implied from their sentence:

{examples}

Does the following sentence
"{check_sentence}"
imply the following statement
"{check_eng_stmt}"?{check_synonyms}
Please answer with just Yes or No."""
old_prompt = (
    "You need to help me verify if a sentence I give you implies "
    "a statement I give you. Provide a correct answer with a "
    'simple yes or no. Sentence: "{check_sentence}" Statement: '
    '"{check_eng_stmt}" Answer:'
)


def get_ag_ns_id(db_refs, default):
    """Return a tuple of name space, id from an Agent's db_refs."""
    for ns in NS_PRIORITY_LIST:
        if ns in db_refs:
            return ns, db_refs[ns]
    return "TEXT", db_refs.get("TEXT", default)


def get_names_gilda(db_refs, name):
    """Get the names for a given db_refs dict using the Gilda API."""
    db, _id = get_ag_ns_id(db_refs, name)
    res = requests.post("http://grounding.indra.bio/names", json={"db": db, "id": _id})
    res.raise_for_status()
    synonyms = res.json()
    if name not in synonyms:
        synonyms.append(name)
    if "TEXT" in db_refs and db_refs["TEXT"] not in synonyms:
        synonyms.append(db_refs["TEXT"])
    return synonyms


def find_synonyms(ev_text: str, eng_stmt: str, synonym_list, case_sensitive=False):
    """Find which synonym is in evidence text and which is in the English stmt."""
    # Remove possible punctuations and parentheses and the split the string
    # on space to match exact words instead of substrings.
    ev_text = ev_text.lower() if not case_sensitive else ev_text
    ev_text = (
        ev_text.replace("(", "").replace(")", "").replace(":", "").replace(
            ";", "").replace("?", "").replace("!", "").replace(
            ",", "").replace(".", "")
    )
    ev_text_list = ev_text.split()

    eng_stmt = eng_stmt.lower() if not case_sensitive else eng_stmt
    eng_stmt = (
        eng_stmt.replace("(", "").replace(")", "").replace(":", "").replace(
            ";", "").replace("?", "").replace("!", "").replace(
            ",", "").replace(".", "")
    )
    eng_stmt_list = eng_stmt.split()

    text_syn = None
    eng_syn = None
    for syn in synonym_list:
        syn_lower = syn.lower() if not case_sensitive else syn
        if text_syn is None and syn_lower in ev_text_list:
            text_syn = syn
        if eng_syn is None and syn_lower in eng_stmt_list:
            eng_syn = syn

        if text_syn and eng_syn:
            break
    return text_syn, eng_syn


def get_synonyms(examples):
    synonyms_per_example = []
    # Loop over examples
    for _, _, ag_json_list in examples:
        all_ag_synonyms = []
        # Loop over agents in the example
        for ag_json in ag_json_list:
            db_refs = ag_json.get("db_refs", {})
            name = ag_json.get("name") or db_refs.get("TEXT")
            all_ag_synonyms.append(get_names_gilda(db_refs=db_refs, name=name))
        synonyms_per_example.append(all_ag_synonyms)
    return synonyms_per_example


def generate_synonyms_string_example(syn_list,
                                     example_sentence,
                                     example_eng_stmt,
                                     index: int):
    """Generate a string with the synonyms for a given example.

    Parameters
    ----------
    syn_list : list
        A list of lists of synonyms for each agent in the statement.
    example_sentence : str
        The example sentence.
    example_eng_stmt : str
        The example statement in English.
    index : int
        The index of the example.

    Returns
    -------
    str
        A string with the synonyms for the given example. If no synonyms
        are found, an empty string is returned.
    """
    selected_synonyms = []
    equals = []
    for ag_synonyms in syn_list:
        s_in_text, s_in_stmt = find_synonyms(
            example_sentence, example_eng_stmt, ag_synonyms, False
        )
        # Both synonyms must be not-None and not be equal, if they are
        # equal, there is no need to add them to the list.
        if s_in_text and s_in_stmt and s_in_stmt != s_in_text:
            selected_synonyms.append((s_in_text, s_in_stmt))
        elif s_in_text and s_in_stmt and s_in_stmt == s_in_text:
            equals.append(s_in_text)

    if selected_synonyms:
        if len(selected_synonyms) == 1:
            s_in_text, s_in_stmt = selected_synonyms[0]
            synonym_str = (
                f'Assume "{s_in_text}" in Sentence{index} and "{s_in_stmt}" '
                f'Statement{index} are synonyms.\n'
            )
        else:
            synonym_str = ("Assume the following list of pairs are synonyms "
                           f"in Sentence{index} and Statement{index}, "
                           f"respectively, above:\n")
            for s_in_text, s_in_stmt in selected_synonyms:
                synonym_str += f'"{s_in_text}" and "{s_in_stmt}"\n'
        return synonym_str
    else:
        # True if there is a match in entity name between the sentence and
        # the statement, False otherwise.
        return len(equals) > 0


def generate_synonyms_string_check(syn_list, check_sentence, check_eng_stmt):
    selected_synonyms = []
    equals = []
    for ag_synonyms in syn_list:
        s_in_text, s_in_stmt = find_synonyms(
            check_sentence, check_eng_stmt, ag_synonyms, False
        )
        # Both synonyms must be not-None and not be equal, if they are
        # equal, there is no need to add them to the list.
        if s_in_text and s_in_stmt and s_in_stmt != s_in_text:
            selected_synonyms.append((s_in_text, s_in_stmt))
        elif s_in_text and s_in_stmt and s_in_stmt == s_in_text:
            equals.append(s_in_text)

    if selected_synonyms:
        if len(selected_synonyms) == 1:
            s_in_text, s_in_stmt = selected_synonyms[0]
            synonym_str = (
                f'Assume "{s_in_text}" and "{s_in_stmt}" are synonyms.\n'
            )
        else:
            synonym_str = "Assume the following list of pairs are synonyms:\n"
            for s_in_text, s_in_stmt in selected_synonyms:
                synonym_str += f'"{s_in_text}" and "{s_in_stmt}"\n'
        return synonym_str
    else:
        # True if there is a match in entity name between the sentence and
        # the statement, False otherwise.
        return len(equals) > 0



def get_create_training_set(
    curations_file: str = None, statement_json_file: str = None, refresh: bool = False
) -> pd.DataFrame:
    if curation_training_data.exists() and not refresh:
        df = pd.read_csv(curation_training_data, sep="\t")
        if isinstance(df["agent_json_list"][0], str):
            from collections import OrderedDict
            logger.info(
                "agent_json_list dtype is str, using eval to convert "
                "to list of OrderedDicts")
            # Apply 'eval' to the column and provide the OrderedDict class
            # as an arg variable to be used in the eval call
            df["agent_json_list"] = df["agent_json_list"].apply(
                eval, args=({"OrderedDict": OrderedDict},))
        return df

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
        cur["english"] = eng_stmt
        cur["agent_json_list"] = [dict(a.to_json()) for a in stmt.agent_list()]
        curation_data.append(cur)

    # Save the training data
    df = pd.DataFrame(curation_data)
    df.to_csv(curation_training_data, sep="\t", index=False)
    return df


def generate_examples_by_tag(curation_df: pd.DataFrame, tag: str, n_examples: int = 5):
    """Generate triples of sentence-english-agent json statement for a tag

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
        A list of tuples with (sentence, english_stmt, agent_json_list)
    """
    if n_examples == -1:
        kwargs = {"frac": 1.0}
    else:
        kwargs = {"n": max(abs(n_examples), 1)}

    cols = ["text", "english", "agent_json_list"]
    return list(map(tuple, curation_df[cols][curation_df["tag"] == tag]
                    .sample(**kwargs).values))


def generate_prompt(
    check,
    check_synonyms=None,
    ex_list=None,
    prompt_template=default_prompt_template,
    syn_list=None,
    min_examples=2,
):
    """Generate a prompt for the given examples.

    Parameters
    ----------
    check :
        The sentence - english statement pair to generate the prompt for.
    check_synonyms :
        A list of synonyms associated with the sentence - english statement
        pair to add to the prompt. Each item in the list is list of
        synonyms, one for each entity in the check statement. The default is
        None.
    ex_list :
        A list of tuples with (sentence, english_stmt) for the examples to
        use in the prompt. The default is None. This is only used if the
        prompt template is the default template.
    prompt_template :
        The prompt template to use.
    syn_list :
        A list of synonyms to use in the prompt. The default is None. It is
        assumed that the synonym lists are in the same order as the
        examples in ex_list. Each item in the list is a list of lists of
        synonyms, one for each entity appearing in the statement.
    min_examples :
        The minimum number of examples to use in the prompt. The default is 2.

    Returns
    -------
    prompt :
        The prompt string
    """
    min_examples = max(min_examples, 2)
    examples_used = 0
    # Generate a synonym string for the statement to check
    check_syn_str = generate_synonyms_string_check(
        check_synonyms, check[0], check[1]
    )
    if check_syn_str:
        check_syn_str = "\n" + check_syn_str if isinstance(check_syn_str, str) else ""
    else:
        logger.info("Although synonyms were needed, no synonyms were found "
                    "for the statement to check")
        return ""
    if prompt_template == default_prompt_template:
        # Generate example text
        example_template = (
            'Sentence{ix}: "{sentence}"\nStatement{ix}: {english}{synonyms}\n'
        )
        examples_str = ""
        syn_list_iter = (None,) * len(ex_list) if syn_list is None else syn_list
        for i, ((sentence, eng_stmt), sl) in enumerate(zip(ex_list, syn_list_iter)):
            # Get synonyms
            if sl is not None:
                synonym_str = generate_synonyms_string_example(
                    syn_list=sl, example_sentence=sentence,
                    example_eng_stmt=eng_stmt, index=i+1
                )
                if synonym_str:
                    if isinstance(synonym_str, str) and len(synonym_str) > 0:
                        sstr = "\n" + synonym_str
                    else:
                        sstr = "\n"
                    examples_str += example_template.format(
                        ix=i + 1, sentence=sentence,
                        english=eng_stmt, synonyms=sstr
                    )
                    examples_used += 1
                    if examples_used >= min_examples:
                        break
                else:
                    logger.warning(
                        "No synonyms were generated for the example "
                        f"{sentence} - {eng_stmt}"
                    )
                    continue

        if examples_used < min_examples:
            logger.warning(
                f"Only {examples_used} examples were used in the "
                f"prompt. The minimum number of examples is {min_examples}."
            )
            return ""
        prmt = default_prompt_template.format(examples=examples_str,
                                              check_synonyms=check_syn_str,
                                              check_sentence=check[0],
                                              check_eng_stmt=check[1])
    else:
        prmt = prompt_template.format(check_sentence=check[0],
                                      check_eng_stmt=check[1])
    return prmt


def run_openai_chat(
    prompt: str,
    model="gpt-3.5-turbo",
    max_tokens=1,
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

    """

    def _get_response(resp):
        if model == "gpt-3.5-turbo":
            return resp["choices"][0]["message"]["content"].strip()
        else:  # text-davinci-003
            return resp["choices"][0]["text"].strip()

    options = {}
    # todo: play around with the prompt. Add examples of correct and maybe
    #  also incorrect
    # For gpt-3.5-turbo chat mode:
    # https://platform.openai.com/docs/api-reference/chat/create
    if model == "gpt-3.5-turbo":
        options["messages"] = [{"role": "user", "content": prompt}]
        api_class = openai.ChatCompletion
    else:  # text-davinci-003
        # For text-davinci-003 chat mode:
        options["prompt"] = prompt
        api_class = openai.Completion

    response = api_class.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        **options,
    )

    logger.info("Got response: ", str(response))
    return _get_response(response)


def two_correct_sample(training_data_df: pd.DataFrame):
    """Test function to run the chat completion with two correct examples."""
    # Get two correct examples
    example_list = generate_examples_by_tag(training_data_df, n_examples=2, tag="correct")

    # Get one example to check at random
    checker_dict = training_data_df.sample(1).to_dict(orient="records")[0]
    checker = (checker_dict["text"], checker_dict["english"])
    checker_synonyms = get_synonyms([(*checker, checker_dict[
        "agent_json_list"])])[0]
    synonyms = get_synonyms(example_list)

    # Only keep the sentence and statement for the examples
    text_examples = [ex[:2] for ex in example_list]

    # Generate the prompt
    prompt = generate_prompt(check=checker,
                             check_synonyms=checker_synonyms,
                             ex_list=text_examples,
                             syn_list=synonyms)

    if not prompt:
        logger.warning("No prompt was generated. Will not run OpenAI.")
        return

    # Run the chat completion
    choice = run_openai_chat(prompt=prompt, max_tokens=2)

    # Get the response and the tag and check if the response is correct
    print(f'Text:\n"{checker[0]}"\n\nStatement:\n"{checker[1]}"\n\n')
    print(
        f"Output\n------\nChoice - Yes/No/(None):\n{choice or '(None)'}.\n"
        f"Originally tagged as: "
        f"{'correct' if checker_dict['tag'] == 'correct' else 'incorrect'}"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        logger.error("Please provide the curations file and the statement json file")
        sys.exit(1)

    curations = args[0]
    statement_jsons_file = args[1]
    main(curations_file=curations, statements_file=statement_jsons_file)

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
