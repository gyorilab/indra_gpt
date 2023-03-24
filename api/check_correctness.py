import json
import logging
import os
import sys
from collections import OrderedDict
from itertools import count
from pathlib import Path
from time import sleep

import openai
import pandas as pd
import requests
from tqdm import tqdm

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
positive_examples_path = LOCAL_FILES.joinpath("positive_examples.tsv")
negative_examples_path = LOCAL_FILES.joinpath("negative_examples.tsv")

default_prompt_template = """{examples}
{query}Please answer with just Yes or No."""
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


def generate_synonym_str(syn_list, index: int = None) -> str:
    """Generate a string with the list of synonyms

    Parameters
    ----------
    syn_list :
        A list of tuples with (synonym_in_sentence, synonym_in_statement)
    index :
        If provided, is the index of the sentence and statement. If None,
        the index will not be included in the string.
    """
    ix = str(index) if index is not None else ""
    sent_str = "Sentence" + ix if index is not None else "the sentence"
    stmt_str = "Statement" + ix if index is not None else "the statement"
    if len(syn_list) == 1:
        syn_sent, syn_stmt = syn_list[0]
        base_str = (
            f'Assume "{syn_sent}" in {sent_str} and "{syn_stmt}" in '
            f'{stmt_str} are synonyms.\n'

        )
    else:
        base_str = (
            f"Assume the following list of pairs are synonyms in {sent_str} "
            f"and {stmt_str}, respectively:\n"
        )

        for syn_sent, syn_stmt in syn_list:
            base_str += f'- "{syn_sent}" and "{syn_stmt}"\n'
    return base_str


def generate_example(
    sentence,
    statement,
    synonyms=None,
    index: int = None
) -> str:
    """Generate an example string

    Parameters
    ----------
    sentence :
        The example sentence.
    statement :
        The example english statement paired with the sentence.
    synonyms :
        A list of tuples with (synonym_in_sentence, synonym_in_statement).
    index :
        If provided, is the index of the sentence and statement. If None,
        the index will not be included in the string.

    Returns
    -------
    :
        A string with the example, including synonyms if provided.
    """
    ix = str(index) if index is not None else ""
    sent_str = "Sentence" + ix
    stmt_str = "Statement" + ix

    example_template = (
        '{sent_str}: "{sentence}"\n{stmt_str}: "{english}"{synonyms}\n'
    )
    if synonyms:
        syn_str = "\n" + generate_synonym_str(synonyms, index)
    else:
        syn_str = ""
    return example_template.format(sent_str=sent_str, sentence=sentence,
                                   stmt_str=stmt_str, english=statement,
                                   synonyms=syn_str)


def generate_example_list(examples, correct: bool, indexer) -> str:
    """Generate a list of examples

    Parameters
    ----------
    examples :
        A list of tuples with (sentence, english_stmt, list_of_synonyms).
        List of synonyms can be None if they are not needed. If provided,
        it should be a list of tuples with (synonym_in_sentence,
        synonym_in_statement).
    correct :
        If True, the statement in the examples are implied by their paired
        sentences. If False, the statement in the examples are **not** implied
        by their paired sentences.
    indexer :
        An iterator to get the index of the sentence and statement.
    """
    pos_str = "The following sentences are paired with statements that are " \
              "implied from their sentence:\n\n"
    neg_str = "The following sentences do not imply the statements they " \
              "are paired with:\n\n"
    template = pos_str if correct else neg_str
    for sentence, statement, syn_list in examples:
        # Synonyms is a list of tuples with (synonym_in_sentence,
        # synonym_in_statement)
        ix = next(indexer)
        ex_str = generate_example(sentence, statement, syn_list, ix)
        template += ex_str
    return template


def generate_query_str(query_sentence, query_stmt, query_synonyms=None) -> str:
    """Generate the query string for the prompt

    Parameters
    ----------
    query_sentence :
        The sentence to query.
    query_stmt :
        The english statement to query.
    query_synonyms :
        A list of tuples with (synonym_in_sentence, synonym_in_statement).
    """
    query_str = "Is the following statement implied by the sentence " \
                "assuming the sentence and the statement follow the same " \
                "pattern as in the examples above?\n\n"
    query_str += generate_example(query_sentence, query_stmt, query_synonyms)
    return query_str


def check_prompt_generation():
    """Quickly test the prompt generation by calling this function"""
    test_sentence1 = "a activates b in this text"
    test_stmt1 = "A activates B"
    test_synonyms1 = [("a", "A"), ("b", "B")]

    test_sentence2 = "c phosphorylates D in this text"
    test_stmt2 = "C phosphorylates D"
    test_synonyms2 = [("c", "C")]

    test_sentence3 = "E deactivates f in this text"
    test_stmt3 = "E activates F"
    test_synonyms3 = [("f", "F")]

    test_sentence4 = "X deactivates Y in this text"
    test_stmt4 = "WW deactivates ZZ"

    test_query_sentence = "a inhibits b in this text"
    test_query_stmt = "A inhibits B"
    test_query_synonyms = [("a", "A"), ("b", "B")]

    pos_examples = [
        (test_sentence1, test_stmt1, test_synonyms1),
        (test_sentence2, test_stmt2, test_synonyms2),
    ]
    neg_examples = [
        (test_sentence3, test_stmt3, test_synonyms3),
        (test_sentence4, test_stmt4, None),
    ]

    test_prompt = generate_prompt(
        query_sentence=test_query_sentence,
        query_stmt=test_query_stmt,
        pos_ex_list=pos_examples,
        neg_ex_list=neg_examples,
        query_synonyms=test_query_synonyms,
    )
    print(test_prompt)


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
    min_examples = max(min_examples, 1)
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
    retry_count=3,
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
    debug :
        If True, the function will print the full response from
        openai.Completion/CharCompletion.create(). The default is False.

    Returns
    -------
    :
        The response from OpenAI as a string
    """

    def _get_response(resp):
        if model == "gpt-3.5-turbo":
            return resp["choices"][0]["message"]["content"].strip()
        else:  # text-davinci-003
            return resp["choices"][0]["text"].strip()

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
        logger.info(f"Got response:\n'{str(response)}' to prompt:\n'{prompt}'")
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


def run_stats_positive_examples(
        training_data_df: pd.DataFrame, n=100, n_pos_examples=2,
        # n_neg_examples=NotImplemented
) -> pd.DataFrame:
    """Run the chat completion on n positive examples

    Parameters
    ----------
    training_data_df :
        The training data
    n :
        The number of chat completions to run
    n_pos_examples :
        The number of positive examples to use in the prompt per run

    Returns
    -------
    :
        A dataframe with the results of the chat completions
    """
    # Get n positive examples
    if positive_examples_path.exists() and positive_examples_path.stat().st_size > 0:
        pos_df = pd.read_csv(positive_examples_path, sep="\t")
        if pos_df.shape[0] < n_pos_examples:
            logger.info(f"Only {len(pos_df)} positive examples "
                        "found. Creating more...")
            save_examples(training_data_df, correct=True)
            pos_df = pd.read_csv(positive_examples_path, sep="\t")
    else:
        logger.info("No positive examples found. Creating...")
        save_examples(training_data_df, correct=True)
        pos_df = pd.read_csv(positive_examples_path, sep="\t")

    # Convert the agent json string to a list of agent jsons
    pos_df["agent_json_list"] = pos_df["agent_json_list"].apply(
        eval, args=({"OrderedDict": OrderedDict},))

    # Get n negative examples
    # if negative_examples_path.exists() and negative_examples_path.stat().st_size > 0:
    #     example_list_neg = pd.read_csv(negative_examples_path, sep="\t").to_dict(
    #         orient="records")
    #     if len(example_list_neg) < n:
    #         logger.info(f"Only {len(example_list_neg)} negative examples "
    #                     "found. Creating more...")
    #         save_examples(training_data_df, correct=False)
    # else:
    #     logger.info("No negative examples found. Creating...")
    #     save_examples(training_data_df, correct=False)
    #     example_list_neg = pd.read_csv(negative_examples_path, sep="\t").to_dict(
    #         orient="records")

    # todo:
    #  - get negative examples in the pipeline
    #  - allow to set the number of positive and negative examples to use in
    #    the prompt separately

    # Initialize 2x2 dataframe to store the results
    confusions = pd.DataFrame(
        data=[[0, 0], [0, 0]],
        columns=["cur_correct", "cur_incorrect"],
        index=["gpt_correct", "gpt_incorrect"]
    )
    previous_checks = set()
    for attempt in tqdm(range(n), desc="Running chat completions", total=n):
        # Get one example to check at random from the examples not matching
        # the positive examples' source_hash
        excluded_ids = set(pos_df["id"]) | previous_checks
        checker_dict = training_data_df[~training_data_df["id"].isin(
            excluded_ids)].sample(n=1).to_dict(orient="records")[0]

        previous_checks.add(checker_dict["id"])

        # Get the positive examples from the dataframe
        pos_examples = list(
            map(tuple,
                pos_df[['text', 'english', 'agent_json_list']].sample(
                    n=n_pos_examples).values)
        )

        checker = (checker_dict["text"], checker_dict["english"])
        checker_synonyms = get_synonyms([(*checker, checker_dict[
            "agent_json_list"])])[0]
        synonyms = get_synonyms(pos_examples)

        # Only keep the sentence and statement for the examples
        text_examples = [ex[:2] for ex in pos_examples]

        # Generate the prompt
        prompt = generate_prompt(check=checker,
                                 check_synonyms=checker_synonyms,
                                 ex_list=text_examples,
                                 syn_list=synonyms)

        if not prompt:
            logger.warning("No prompt was generated, skipping...")
            continue

        # Run the chat completion
        try:
            choice = run_openai_chat(prompt=prompt, max_tokens=2)
        except Exception as e:
            logger.warning(f"Error while running chat completion: {e}")
            continue

        # Update the confusion matrix
        if choice.lower() == "yes" and checker_dict["tag"] == "correct":
            confusions.loc["gpt_correct", "cur_correct"] += 1
        elif choice.lower() == "yes" and checker_dict["tag"] != "correct":
            confusions.loc["gpt_correct", "cur_incorrect"] += 1
        elif choice.lower() == "no" and checker_dict["tag"] == "correct":
            confusions.loc["gpt_incorrect", "cur_correct"] += 1
        elif choice.lower() == "no" and checker_dict["tag"] != "correct":
            confusions.loc["gpt_incorrect", "cur_incorrect"] += 1
        else:
            logger.warning(f"Choice {choice} not recognized.")

        sleep(0.1)

    # Add sum row and column
    confusions.loc["sum"] = confusions.sum()
    confusions["sum"] = confusions.sum(axis=1)

    return confusions


def save_examples(training_data_df, correct: bool = True):
    """Save the examples to a csv file"""
    saved = []
    if correct:
        eq = "=="
        out_file = positive_examples_path
    else:
        eq = "!="
        out_file = negative_examples_path
    for row in training_data_df.query('tag @eq "correct"').sample(
            frac=1.0).itertuples():
        eval(f'assert row.tag {eq} "correct"')
        synonyms = get_synonyms([(row.text, row.english,
                                  row.agent_json_list)])[0]
        syn_pairs = []
        for sl in synonyms:
            in_text, in_stmt = find_synonyms(row.text, row.english, sl, case_sensitive=False)
            if in_text and in_stmt:
                syn_pairs.append((in_text, in_stmt))
        if len(syn_pairs) != len(synonyms):
            skip = "Not all synonyms were found in the sentence and " \
                   "statement, recommend skipping this one\n"
        else:
            skip = ""
        syn_pairs_str = "\n".join([f"{s[0]} - {s[1]}" for s in syn_pairs])

        choice = input(
            f"Sentence:\n---------\n {row.text}\nStatement:\n----------"
            f"\n{row.english}\n\nSynonyms:\n---------\n{syn_pairs_str}\n\n"
            f"{skip}Got {len(saved)} examples so far\nSave? (y/n/b): "
        )
        if choice == "b":
            print("Breaking")
            break
        elif choice == "y":
            print("Saving")
            saved.append(row.Index)
        else:
            print("Not saving")

    if saved:
        dump_options = dict(index=False, header=True, sep="\t")
        if out_file.exists():
            # Get the file size
            file_size = out_file.stat().st_size
            choice = input(f"File {out_file} already exists (Size "
                           f"{file_size} B). Overwrite, Append or Cancel? "
                           f"(o/a/c): ")
            if choice == "o":
                print(f"Saving {len(saved)} examples to {out_file}")
                training_data_df.loc[saved].to_csv(out_file, **dump_options)
            elif choice == "a":
                print(f"Appending {len(saved)} examples to {out_file}")
                # Skip header and set mode to append
                dump_options.update(header=False, mode="a")
                training_data_df.loc[saved].to_csv(out_file, **dump_options)
        else:
            print(f"Saving {len(saved)} examples to {out_file}")
            training_data_df.loc[saved].to_csv(out_file, **dump_options)


# todo: create cli with click
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        logger.error("Please provide the curations file and the statement json file")
        sys.exit(1)

    curations = args[0]
    statement_jsons_file = args[1]
    #main(curations_file=curations, statements_file=statement_jsons_file)

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
