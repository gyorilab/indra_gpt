import json
import logging
import os
import sys
from collections import OrderedDict, Counter
from datetime import datetime
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
        "Please set the OPENAI_API_KEY and OPENAPI_ORG environment variables."
    ) from err


logger = logging.getLogger(__name__)


HERE = Path(__file__).parent
LOCAL_FILES = HERE.parent.joinpath("local_data")
LOCAL_FILES.mkdir(exist_ok=True)
curation_training_data = LOCAL_FILES.joinpath("training_data.tsv")
positive_examples_path = LOCAL_FILES.joinpath("positive_examples.tsv")
negative_examples_path = LOCAL_FILES.joinpath("negative_examples.tsv")

default_prompt_template = """{examples}
{query}Please answer with just 'Yes' or 'No'."""
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


def get_synonyms(ag_json_list, retries=3):
    """Get the synonyms for a given list of agent JSONs.

    Parameters
    ----------
    ag_json_list : list
        A list of agent JSONs corresponding to a single example.
    retries : int
        The number of times to retry getting the synonyms in case of an
        error with the Gilda Web API.

    Returns
    -------
    list
        A list of lists of synonyms for each agent in the example.
    """
    all_ag_synonyms = []
    # Loop over agents in the example
    for ag_json in ag_json_list:
        db_refs = ag_json.get("db_refs", {})
        name = ag_json.get("name") or db_refs.get("TEXT")

        # Get the synonyms for the agent
        for trie in range(retries):
            try:
                all_ag_synonyms.append(
                    get_names_gilda(db_refs=db_refs, name=name)
                )
                break
            except Exception as e:
                if trie == retries - 1:
                    logger.error(f"Failed to get synonyms for after "
                                 f"{retries} tries.")
                    raise e
                sleep(1)

    return all_ag_synonyms


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


def parse_synonyms(text, english, agent_json_list, agent_synonyms_list):
    """Get relevant synonyms given text, statement, agent jsons, synonyms

    Parameters
    ----------
    text : str
        The text to parse.
    english : str
        The English statement associated with the evidence text.
    agent_json_list : list
        A list of agent JSONs associated with the statement.
    agent_synonyms_list : list
        A list of lists of synonyms for each agent in the statement.

    Returns
    -------
    Optional[list]
        A list of tuples of (synonym in text, synonym in statement). If any
        list of synonyms can't be matched to both the statement and the
        evidence text, None is returned.
    """
    relevant_sl = []
    missing_synonyms = False
    for ag_json, sl in zip(agent_json_list, agent_synonyms_list):
        s_in_text, s_in_stmt = find_synonyms(
            text, english, sl, False
        )
        # Only keep the synonyms that are in the text and the
        # statement and also are not equal
        # todo: do this up front when examples are selected instead
        if s_in_text and s_in_stmt:
            if s_in_text != s_in_stmt:
                relevant_sl.append((s_in_text, s_in_stmt))
            else:
                # If the synonyms are equal, we don't need to
                # add them to the prompt
                pass
        # If no synonyms are found, check that the names still are
        # in the example
        else:
            name = ag_json["name"]
            text_name = ag_json["db_refs"].get("TEXT", None)
            names = {name, text_name} - {None}

            # If the same name isn't in the example, this is a missing synonym
            if not any(n.lower() in text.lower() and
                       n.lower() in english.lower()
                       for n in names):
                missing_synonyms = True
                break
    if missing_synonyms:
        return None

    return relevant_sl


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
        syn_str = "\n"
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

    test_sentence2 = "C phosphorylates D in this text"
    test_stmt2 = "C phosphorylates D"

    test_sentence3 = "E deactivates f in this text"
    test_stmt3 = "E activates F"
    test_synonyms3 = [("f", "F")]

    test_sentence4 = "X deactivates Y in this text"
    test_stmt4 = "x deactivates Y"
    test_synonyms4 = [("X", "x")]

    test_query_sentence = "a inhibits b in this text"
    test_query_stmt = "A inhibits B"
    test_query_synonyms = [("a", "A"), ("b", "B")]

    pos_examples = [
        (test_sentence1, test_stmt1, test_synonyms1),
        (test_sentence2, test_stmt2, None),
    ]
    neg_examples = [
        (test_sentence3, test_stmt3, test_synonyms3),
        (test_sentence4, test_stmt4, test_synonyms4),
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
    query_sentence,
    query_stmt,
    pos_ex_list=None,
    neg_ex_list=None,
    query_synonyms=None,
):
    """Generate a prompt for the given examples.

    Parameters
    ----------
    query_sentence :
        The sentence to query.
    query_stmt :
        The english statement to query.
    pos_ex_list :
        A list of tuples with (sentence, english_stmt, synonym_list)
        for the examples to use in the prompt. The synonym list is assumed
        to be a list of tuples with (synonym_in_sentence,
        synonym_in_statement). Default: None.
    neg_ex_list :
        A list of tuples with (sentence, english_stmt, synonym_list)
        for the examples to use in the prompt. The synonym list is assumed
        to be a list of tuples with (synonym_in_sentence,
        synonym_in_statement). Default: None.
    query_synonyms :
        A list of synonyms associated with the sentence - english statement
        pair that is queries. Each item in the list of tuples with
        (synonym in sentence, synonym in statement). The default is None.

    Returns
    -------
    prompt :
        The prompt string
    """
    if pos_ex_list is None and neg_ex_list is None:
        raise ValueError("Must provide at least one example list.")

    # Get positive and negative examples
    indexer = count(1)
    if pos_ex_list is not None:
        pos_ex_str = generate_example_list(pos_ex_list, True, indexer)
    else:
        pos_ex_str = ""

    if neg_ex_list is not None:
        neg_ex_str = generate_example_list(neg_ex_list, False, indexer)
    else:
        neg_ex_str = ""

    examples_str = pos_ex_str + neg_ex_str + "\n=======\n"

    # Generate query string
    query_str = generate_query_str(query_sentence, query_stmt, query_synonyms)

    # Generate positive and negative examples
    prmt = default_prompt_template.format(examples=examples_str,
                                          query=query_str)
    return prmt


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

    def _get_response(resp):
        if model == "gpt-3.5-turbo":
            choice = resp["choices"][0]["message"]["content"]
        else:  # text-davinci-003
            choice = resp["choices"][0]["text"]

        if resp["choices"][0]["finish_reason"] == "length":
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


def two_correct_sample(training_data_df: pd.DataFrame):
    """Test function to run the chat completion with two correct examples."""
    # Get two correct examples
    example_list = generate_examples_by_tag(training_data_df, n_examples=2, tag="correct")

    # Get one example to check at random
    checker_dict = training_data_df.sample(1).to_dict(orient="records")[0]
    checker = (checker_dict["text"], checker_dict["english"])
    checker_synonyms = get_synonyms(checker_dict["agent_json_list"])
    synonyms = [get_synonyms(ajl) for _, _, ajl in example_list]

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


def run_stats(
    training_data_df: pd.DataFrame,
    n_iter=100,
    n_pos_examples=2,
    n_neg_examples=2,
    neg_tag: str = None,
    debug_print: bool = False
):
    """Run the chat completion on n positive examples

    Parameters
    ----------
    training_data_df :
        The training data
    n_iter :
        The number of chat completions to run
    n_pos_examples :
        The number of positive examples to use in the prompt per run.
        Default: 2.
    n_neg_examples :
        The number of negative examples to use in the prompt per run.
        Default: 2.
    neg_tag :
        The tag to use for the negative examples. If None, all tags will be
        used. Default: None.
    debug_print :
        If True, the function will print the prompt and the response from the
        OpenAI API. The default is False.

    Returns
    -------
    :
        A dataframe containing the confusion matrix of the results of the
        chat completions
    """
    def _get_examples_df(examples_path: Path, n_examples: int) -> pd.DataFrame:
        if examples_path.exists() and examples_path.stat().st_size > 0:
            df = pd.read_csv(examples_path, sep="\t")
            if df.shape[0] < n_examples:
                logger.info(f"Only {len(df)} positive examples "
                            "found. Creating more...")
                save_examples(training_data_df, correct=True)
                df = pd.read_csv(examples_path, sep="\t")
        else:
            logger.info("No examples found. Creating...")
            save_examples(training_data_df, correct=True)
            df = pd.read_csv(examples_path, sep="\t")

        # Convert the agent json string to a list of agent jsons
        df["agent_json_list"] = df["agent_json_list"].apply(
            eval, args=({"OrderedDict": OrderedDict},))

        return df

    def _get_examples(df, n_examples):
        examples_base = list(map(tuple,
                                 df[
                                     ['text', 'english', 'agent_json_list']
                                 ].sample(frac=1.0).values))

        # Get the synonyms
        synonyms_base = [
            get_synonyms(ajl) for _, _, ajl in examples_base
        ]

        examples = []
        for (text, english, agent_json_list), agent_synonyms_list in zip(
                examples_base, synonyms_base
        ):
            relevant_sl = parse_synonyms(
                text, english, agent_json_list, agent_synonyms_list
            )
            # None means there are synonyms missing and the entity names are
            # not the same in the example and the english statement
            if relevant_sl is None:
                continue

            examples.append((text, english, relevant_sl or None))
            if len(examples) == n_examples:
                break
        if len(examples) < n_examples:
            logger.warning(f"Only {len(examples)} examples found. ")
            inp = input("break? (y/n)")
            if inp == "y":
                sys.exit(1)
        return examples

    # Get n positive examples
    if n_pos_examples > 0:
        pos_df = _get_examples_df(positive_examples_path, n_pos_examples)
    else:
        pos_df = None

    # Get n negative examples
    if n_neg_examples > 0:
        neg_df = _get_examples_df(negative_examples_path, n_neg_examples)
    else:
        neg_df = None

    if n_neg_examples and neg_tag:
        neg_df = neg_df[neg_df["tag"] == neg_tag]
        if neg_df.shape[0] < n_neg_examples:
            logger.warning(f"Only {len(neg_df)} negative examples "
                           f"with tag '{neg_tag}' found. Creating more...")
            save_examples(training_data_df, correct=False)
            neg_df = _get_examples_df(negative_examples_path, n_neg_examples)

    previous_checks = set()
    start_dt = datetime.utcnow()
    results_dict = {"start_time": start_dt.isoformat(),
                    "error_count": 0,
                    "prompts": [],
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "true_negative": 0}
    for _ in tqdm(range(n_iter), desc="Running chat completions", total=n_iter):
        # Get one example to check at random from the examples not matching
        # the examples' source_hash or the ones already checked
        excluded_ids = set(pos_df["id"]) | previous_checks
        checker_dict = training_data_df[~training_data_df["id"].isin(
            excluded_ids)].sample(n=1).to_dict(orient="records")[0]

        previous_checks.add(checker_dict["id"])

        # Get the positive examples from the dataframe
        if pos_df is not None:
            pos_examples = _get_examples(pos_df, n_pos_examples)
        else:
            pos_examples = None

        # Get the negative examples from the dataframe
        if neg_df is not None:
            neg_examples = _get_examples(neg_df, n_neg_examples)
        else:
            neg_examples = None

        query_synonyms_base = get_synonyms(checker_dict["agent_json_list"])
        query_synonyms = parse_synonyms(
            text=checker_dict["text"], english=checker_dict["english"],
            agent_json_list=checker_dict["agent_json_list"],
            agent_synonyms_list=query_synonyms_base
        )
        if query_synonyms is None:
            logger.warning(f"No matching synonyms between text and "
                           f"statement found in the {checker_dict['english']}")
            continue

        # Generate the prompt
        prompt = generate_prompt(query_sentence=checker_dict["text"],
                                 query_stmt=checker_dict["english"],
                                 query_synonyms=query_synonyms,
                                 pos_ex_list=pos_examples,
                                 neg_ex_list=neg_examples)

        if not prompt:
            logger.warning("No prompt was generated, skipping...")
            continue

        # Run the chat completion
        try:
            choice = run_openai_chat(prompt=prompt,
                                     max_tokens=2,
                                     debug=debug_print)
        except Exception as e:
            logger.warning(f"Error while running chat completion: {e}")
            results_dict["error_count"] += 1
            continue

        # Save the prompt
        results_dict["prompts"].append(prompt)

        # Update the confusion matrix
        # gpt correct - correct
        if choice.lower() == "yes" and checker_dict["tag"] == "correct":
            results_dict["true_positive"] += 1
        # gpt correct - incorrect
        elif choice.lower() == "yes" and checker_dict["tag"] != "correct":
            results_dict["false_positive"] += 1
        # gpt incorrect - correct
        elif choice.lower() == "no" and checker_dict["tag"] == "correct":
            results_dict["false_negative"] += 1
        # gpt incorrect - incorrect
        elif choice.lower() == "no" and checker_dict["tag"] != "correct":
            results_dict["true_negative"] += 1
        else:
            logger.warning(f"Choice {choice} not recognized.")
            continue

        sleep(0.1)

    results_dict["end_time"] = datetime.utcnow().isoformat()

    # Calculate the precision, recall and accuracy
    tp = results_dict["true_positive"]
    fp = results_dict["false_positive"]
    fn = results_dict["false_negative"]
    tn = results_dict["true_negative"]
    results_dict["total_examples"] = N = tp + fp + fn + tn
    results_dict["precision"] = tp / (tp + fp) if tp + fp > 0 else 0
    results_dict["recall"] = tp / (tp + fn) if tp + fn > 0 else 0
    results_dict["accuracy"] = (tp + tn) / N if N > 0 else 0

    # Print the results
    print("Confusion matrix:")
    print(pd.DataFrame(data=[[tp, fp], [fn, tn]],
                       index=["gpt_correct", "gpt_incorrect"],
                       columns=["correct", "incorrect"]))
    print(f"Precision: {results_dict['precision']}")
    print(f"Recall: {results_dict['recall']}")
    print(f"Accuracy: {results_dict['accuracy']}")
    print(f"Total examples: {N}")

    # Save the results
    logger.info("Saving results...")
    fname = start_dt.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
    LOCAL_FILES.joinpath("results").mkdir(exist_ok=True)
    with open(LOCAL_FILES.joinpath("results", fname), "w") as f:
        json.dump(results_dict, f, indent=4)

    return results_dict


def save_examples(training_data_df, correct: bool = True):
    """Save the examples to a tsv file"""
    saved = []
    saved_tags = []
    if correct:
        row_iter = training_data_df.query('tag == "correct"')
        out_file = positive_examples_path
    else:
        row_iter = training_data_df.query('tag != "correct"')
        out_file = negative_examples_path
        if out_file.exists():
            saved_df = pd.read_csv(out_file, sep="\t")
            saved_tags = list(saved_df["tag"])

    for row in row_iter.sample(frac=1.0).itertuples():
        if correct:
            assert row.tag == "correct"
        else:
            assert row.tag != "correct"

        synonyms = get_synonyms(row["agent_json_list"].values[0])
        syn_pairs = []
        for sl in synonyms:
            in_text, in_stmt = find_synonyms(row.text, row.english, sl, case_sensitive=False)
            if in_text and in_stmt:
                syn_pairs.append((in_text, in_stmt))
        if len(syn_pairs) != len(synonyms):
            skip = "> > Not all synonyms were found in the sentence and " \
                   "statement, recommend skipping this one\n"
        else:
            skip = ""
        syn_pairs_str = "\n".join([f"{s[0]} - {s[1]}" for s in syn_pairs])

        if not correct:
            tag_distr = ', '.join(f'{t}: {c}' for t, c in Counter(saved_tags).most_common())
            tag_str = (
                f"Current Tag: {row.tag}\nSaved tags: {tag_distr}\n"
            )
        else:
            tag_str = ""
        choice = input(
            f"\nSentence:\n---------\n{row.text}\n---------\n\n"
            f"Statement:\n----------\n{row.english}\n----------\n\nSynonyms:"
            f"\n---------\n{syn_pairs_str}\n\n{tag_str}"
            f"{skip}Got {len(saved)} examples so far\nSave? (y/n/b): "
        )
        if choice == "b":
            print("Breaking")
            break
        elif choice == "y":
            print("Saving")
            saved.append(row.Index)
            saved_tags.append(row.tag)
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
