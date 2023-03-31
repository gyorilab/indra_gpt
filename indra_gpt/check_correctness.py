import json
import logging
import sys
from collections import OrderedDict, Counter
from datetime import datetime
from itertools import count
from pathlib import Path
from time import sleep

import biolookup
import openai
import pandas as pd
from tqdm import tqdm

import gilda
from indra.config import get_config, IndraConfigError
from indra.assemblers.english import EnglishAssembler
from indra.assemblers.indranet.assembler import NS_PRIORITY_LIST
from indra.statements.io import stmts_from_json_file

try:
    openai.api_key = get_config("OPENAI_API_KEY", failure_ok=False)
    organization = get_config("OPENAI_ORG")
    if organization:
        openai.organization = organization
except IndraConfigError as err:
    raise KeyError(
        "Please set OPENAI_API_KEY in the environment or in the indra config."
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
    """Get the names for a given db_refs dict using Gilda

    Parameters
    ----------
    db_refs : dict
        A dictionary of db_refs.
    name : str
        The name of the agent.

    Returns
    -------
    list
        A list of names for the agent.
    """
    db, _id = get_ag_ns_id(db_refs, name)
    synonyms = gilda.get_names(db, _id)
    if name not in synonyms:
        synonyms.append(name)
    if "TEXT" in db_refs and db_refs["TEXT"] not in synonyms:
        synonyms.append(db_refs["TEXT"])
    return synonyms


def find_synonyms(
    ev_text: str,
    eng_stmt: str,
    synonym_list,
    case_sensitive=False,
    substring_match=False,
):
    """Find which synonym is in the evidence text and in the English stmt

    Parameters
    ----------
    ev_text : str
        The evidence text.
    eng_stmt : str
        The English statement.
    synonym_list : list
        A list of synonyms for the agent.
    case_sensitive : bool
        Whether to match case when looking for synonyms.
    substring_match : bool
        Whether to allow substring match or not. If allowed, the synonym
        must match in both the text and the statement at the same time, i.e.
        the names are the same but could be embedded in text e.g.
        "RIG-I" matches "... RIG-I-induced activation...".

    Returns
    -------
    tuple
        A tuple of the synonym in the evidence text and the synonym in the
        English statement.
    """
    def _clean(s):
        return s.replace("(", " ").replace(")", " ").replace(
            ":", " ").replace(";", " ").replace("?", " ").replace(
            "!", " ").replace(",", " ").replace(".", " ").replace(
            "/", " ").replace("  ", " ")

    # Remove possible punctuations and parentheses and then split the string
    # on space to match exact words instead of substrings.
    ev_text = ev_text.lower() if not case_sensitive else ev_text
    org_ev_text = ev_text
    ev_text = _clean(ev_text)
    ev_text_list = ev_text.split()

    eng_stmt = eng_stmt.lower() if not case_sensitive else eng_stmt
    org_eng_stmt = eng_stmt
    eng_stmt = _clean(eng_stmt)
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

    # If by now one or both of the synonyms are not found, try substring
    # match, if requested.
    if substring_match and (text_syn is None or eng_syn is None):
        for syn in synonym_list:
            syn_lower = syn.lower() if not case_sensitive else syn
            if text_syn is None and syn_lower in org_ev_text:
                text_syn = syn
            if eng_syn is None and syn_lower in org_eng_stmt:
                eng_syn = syn

            if text_syn and eng_syn:
                break

    return text_syn, eng_syn


def get_agent_info(ev_text, english, ag_list, retry_count=3):
    """Get the synonyms and definitions for each agent

    Parameters
    ----------
    ev_text : str
        The evidence text.
    english : str
        The English statement.
    ag_list : list
        A list of agents corresponding to a single example statement.
    retry_count : int
        The number of times to retry getting the definition for an agent.
        Default: 3.

    Returns
    -------
    dict
        A dict of dicts of synonyms and definitions for each agent keyed by
        curie.
    """
    ag_info = {}
    retry_count = max(retry_count, 1)

    # Loop over agents in the example
    for ag in ag_list:
        db_refs = ag.db_refs
        name = ag.name or ag.db_refs.get("TEXT")
        curie = ":".join(ag.get_grounding())

        # Get the synonyms for the agent
        synonyms = get_names_gilda(db_refs=db_refs, name=name)

        # Get the definition for the agent
        # todo: Run local setup of biolookup when the errors are fixed
        # for trie in range(retry_count):
        #     try:
        #         bl_info = biolookup.lookup(curie)
        #         definition = bl_info.get("definition", "")
        #         break
        #     except Exception as e:
        #         logger.warning(
        #             f"Error getting definition for {curie} "
        #             f"on try {trie+1}: {e}"
        #         )
        #         if trie < retry_count - 1:
        #             sleep(1)
        # else:
        #     definition = ""

        # Use None to indicate that the definition is missing and should be
        # tried to be filled out at runtime
        definition = None

        in_text, in_stmt = find_synonyms(ev_text,
                                         english,
                                         synonyms,
                                         case_sensitive=False,
                                         substring_match=True)

        ag_info[curie] = (
            {"name": name, "synonyms": synonyms, "definition": definition,
             "syn_in_text": in_text, "syn_in_stmt": in_stmt}
        )

    return ag_info


def parse_synonyms(
    text,
    english,
    agent_json_list,
    agent_synonyms_list,
    case_sensitive=False,
    substring_match=False
):
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
    case_sensitive : bool
        Whether to match case when looking for synonyms.
    substring_match : bool
        Whether to allow substring match or not. If allowed, the synonym
        must match in both the text and the statement at the same time, i.e.
        the names are the same but could be embedded in text e.g.
        "RIG-I" matches "... RIG-I-induced activation..." in the text and
        "RIG-I activates X" in the statement.

    Returns
    -------
    list
        A list of tuples of (synonym in text, synonym in statement). If any
        list of synonyms can't be matched to both the statement and the
        evidence text, the entry is None for both entities. If the synonym in
        the text and the statement are the same, the entry is True for both
        entities.
    """
    relevant_sl = []
    for ag_json, sl in zip(agent_json_list, agent_synonyms_list):
        s_in_text, s_in_stmt = find_synonyms(
            text, english, sl,
            case_sensitive=case_sensitive,
            substring_match=substring_match
        )
        # Only keep the synonyms that are in the text and the
        # statement and also are not equal
        if s_in_text and s_in_stmt:
            if s_in_text != s_in_stmt:
                relevant_sl.append((s_in_text, s_in_stmt))
            else:
                # If the synonyms are equal, add True to the list
                relevant_sl.append((True, True))
        # If no synonyms are found, do a last check if the names are equal to
        # begin with
        else:
            name = ag_json["name"]
            text_name = ag_json["db_refs"].get("TEXT", None)
            names = {name, text_name} - {None}

            # Unless the same name is in the example and in the statement,
            # this is a missing synonym -> Add None to the list
            for n in names:
                if n.lower() in text.lower() and n.lower() in english.lower():
                    relevant_sl.append((True, True))
                    break
            else:
                relevant_sl.append((None, None))

    return relevant_sl


def get_create_training_set(
    curations_file: str = None,
    statement_json_file: str = None,
    refresh: bool = False,
    test: bool = False,
) -> pd.DataFrame:
    """Get the training set for curation.

    Parameters
    ----------
    curations_file : str
        The path to the curations file.
    statement_json_file : str
        The path to the statement json file.
    refresh : bool
        If True, the training set will be regenerated even if it already
        exists. Default: False.
    test : bool
        If True, run a quick test of the training set generation by only
        looping 10 random examples and don't save the result. Default: False.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the training set.
    """
    if curation_training_data.exists() and not refresh and not test:
        df = pd.read_csv(curation_training_data, sep="\t")
        if isinstance(df["agent_json_list"][0], str):
            logger.info(
                "agent_json_list dtype is str, using eval to convert "
                "to list of OrderedDicts")
            # Apply 'eval' to the column and provide the OrderedDict class
            # as an arg variable to be used in the eval call
            df["agent_json_list"] = df["agent_json_list"].apply(
                eval, args=({"OrderedDict": OrderedDict},))
        if isinstance(df["agent_info"][0], str):
            logger.info(
                "agent_info dtype is str, using eval to convert "
                "to dict"
            )
            df["agent_info"] = df["agent_info"].apply(eval)
        return df

    # Create the training set
    if curations_file is None or statement_json_file is None:
        raise FileNotFoundError(
            f"Please provide the curations and statement json file if "
            f"pre-generated training data is not available at "
            f"{curation_training_data}"
        )
    # Initialize Gilda so that tqdm is more accurate
    _ = gilda.get_grounder()
    logger.info("Loading curations")
    curs = json.load(open(curations_file, "r"))
    logger.info("Loading statements")
    stmts = stmts_from_json_file(statement_json_file)
    stmts_by_hash = {s.get_hash(): s
                     for s in tqdm(stmts, desc="Creating statement lookup")}

    # Loop the curations, get the corresponding statement with evidence and
    # extend the curation with the evidence text and english assembled
    # statement
    curation_data = []
    skipped = 0
    for cur in tqdm(curs, desc="Matching curations to statements"):
        stmt = stmts_by_hash[cur["pa_hash"]]
        ev = [e for e in stmt.evidence if e.get_source_hash() == cur["source_hash"]][0]
        if ev.text is None:
            skipped += 1
            continue

        cur["text"] = ev.text
        eng_stmt = EnglishAssembler([stmt]).make_model()
        cur["english"] = eng_stmt
        ag_list = stmt.agent_list()
        cur["agent_json_list"] = [a.to_json() for a in ag_list]
        cur["agent_info"] = get_agent_info(ev_text=ev.text,
                                           english=eng_stmt,
                                           ag_list=ag_list)

        curation_data.append(cur)
        if test and len(curation_data) == 10:
            logger.info(
                "Test mode: Breaking after 10 examples - not saving data"
            )
            break

    if skipped:
        logger.info(f"Skipped {skipped} examples due to missing synonyms")

    # Save the training data
    df = pd.DataFrame(curation_data)
    if not test:
        logger.info(f"Saving training data to {curation_training_data}")
        df.to_csv(curation_training_data, sep="\t", index=False)

    return df


def generate_synonym_str(agents_info, index: int = None) -> str:
    """Generate a string with the list of synonyms

    Parameters
    ----------
    agents_info :
        A dictionary with agent information for each agent in the
        statement keyed by curie. Each agent dictionary has the name,
        synonyms, and definition of the agent.
    index :
        If provided, is the index of the sentence and statement. If None,
        the index will not be included in the string.
    """
    index_str = f" in example {index}" if index is not None else ""
    def_fmt = "The definition of {name} is: {definition}.\n"
    base_str_fmt_plural = 'These are synonyms for "{name}"{ex_str}: {synonyms}.\n'
    base_str_fmt_singular = 'This is a synonym for "{name}"{ex_str}: {synonyms}.\n'
    base_str = ""
    for curie, agent_info in agents_info.items():
        in_text = agent_info["syn_in_text"]
        in_stmt = agent_info["syn_in_stmt"]
        name = in_stmt or in_text
        synonyms = set(agent_info["synonyms"]) - {name}
        if len(synonyms) == 0:
            continue
        if agent_info["definition"]:
            definition = agent_info["definition"]
        else:
            res = biolookup.lookup(curie)
            definition = res.get("definition", "")

        base_str += def_fmt.format(
            name=name, definition=definition) if definition else ""

        if len(synonyms) == 1:
            base_str_fmt = base_str_fmt_singular
            synonyms = list(synonyms)[0]
        else:
            base_str_fmt = base_str_fmt_plural
            synonyms = list(synonyms)
            if len(synonyms) == 2:
                synonyms = '"' + '" and "'.join(synonyms) + '"'
            else:
                synonyms = \
                    '"' + '", "'.join(synonyms[:-1]) + \
                    '", and "' + synonyms[-1] + '"'
        base_str += base_str_fmt.format(
            name=name, synonyms=synonyms, ex_str=index_str
        )

    return base_str


def generate_example(
    sentence,
    statement,
    agents_info=None,
    index: int = None
) -> str:
    """Generate an example string

    Parameters
    ----------
    sentence :
        The example sentence.
    statement :
        The example english statement paired with the sentence.
    agents_info :
        A dict with agent info keyed by curie for each agent. Contains the
        name, synonyms, definition (if available), synonym used in text and
        synonym used in statement for each agent.
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
    if agents_info:
        syn_str = "\n" + generate_synonym_str(agents_info, index)
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
        A list of tuples with (sentence, english_stmt, agents_info_dict).
        Agents info dict is keyed by curie and contains the name, synonyms,
        definition (if available), synonym used in text and synonym used in
        statement for each agent.
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
    for sentence, statement, agents_info in examples:

        ix = next(indexer)
        ex_str = generate_example(sentence, statement, agents_info, ix)
        template += ex_str
    return template


def generate_query_str(query_sentence, query_stmt, agents_info=None) -> str:
    """Generate the query string for the prompt

    Parameters
    ----------
    query_sentence :
        The sentence to query.
    query_stmt :
        The english statement to query.
    agents_info :
        A dictionary keyed by curie with agent information for each agent
        in the statement. Each agent dictionary has the name, synonyms,
        definition (if available), synonym used in the sentence, and synonym
        used in the statement.
    """
    query_str = "Is the following statement implied by the sentence " \
                "assuming the sentence and the statement follow the same " \
                "pattern as in the examples above?\n\n"
    query_str += generate_example(query_sentence, query_stmt, agents_info)
    return query_str


def check_prompt_generation():
    """Quickly test the prompt generation by calling this function"""
    test_sentence1 = "a activates b in this text"
    test_stmt1 = "A activates B"
    test_synonyms1 = {"A": {"name": "a",
                            "definition": "a is a protein",
                            "synonyms": ["a", "A", "aa", "A-A"],
                            "syn_in_text": "a",
                            "syn_in_stmt": "A"},
                      "B": {"name": "b",
                            "definition": "b is a protein",
                            "synonyms": ["b", "B"],
                            "syn_in_text": "b",
                            "syn_in_stmt": "B"}}

    test_sentence2 = "C phosphorylates D in this text"
    test_stmt2 = "C phosphorylates D"

    test_sentence3 = "E deactivates f in this text"
    test_stmt3 = "E activates F"
    test_synonyms3 = {"F": {"name": "F",
                            "definition": "F is a small molecule",
                            "synonyms": ["f", "F", "ff", "F3"],
                            "syn_in_text": "f",
                            "syn_in_stmt": "F"},}

    test_sentence4 = "X deactivates Y in this text"
    test_stmt4 = "x deactivates Y"
    test_synonyms4 = {"x": {"name": "X",
                            "definition": "X is a protein",
                            "synonyms": ["X", "x", "XX", "X1"],
                            "syn_in_text": "X",
                            "syn_in_stmt": "x"},}

    test_query_sentence = "a inhibits b in this text"
    test_query_stmt = "A inhibits B"
    test_query_synonyms = {"A": {"name": "A",
                                 "definition": "A is a protein",
                                 "synonyms": ["a", "A", "a1"],
                                 "syn_in_text": "a",
                                 "syn_in_stmt": "A"},
                           "B": {"name": "B",
                                 "definition": "B is a protein",
                                 "synonyms": ["b", "B", "b1", "B1", "bb"],
                                 "syn_in_text": "b",
                                 "syn_in_stmt": "B"}}

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
        query_agent_info=test_query_synonyms,
    )
    print(test_prompt)


def generate_prompt(
    query_sentence,
    query_stmt,
    pos_ex_list=None,
    neg_ex_list=None,
    query_agent_info=None,
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
    query_agent_info :
        A dict of agent info associated with the sentence - english statement
        pair that is queried. Each entry is keyed by its curie and contains
        name, synonym list and optionally a definition. Default: None.

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
        pos_ex_str = generate_example_list(pos_ex_list,
                                           correct=True,
                                           indexer=indexer)
    else:
        pos_ex_str = ""

    if neg_ex_list is not None:
        neg_ex_str = generate_example_list(neg_ex_list,
                                           correct=False,
                                           indexer=indexer)
    else:
        neg_ex_str = ""

    examples_str = pos_ex_str + neg_ex_str + "\n=======\n"

    # Generate query string
    query_str = generate_query_str(query_sentence, query_stmt, query_agent_info)

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


def explain_negative_examples(
    training_data_df: pd.DataFrame,
    tag: str = None,
    n_iter: int = 10,
    max_tokens: int = 150,
):
    """Submit statements curated as incorrect that asks why they are incorrect

    Parameters
    ----------
    training_data_df :
        The training data DataFrame
    tag :
        The tag to filter the training data by. If None, all examples will be
        used. The default is None.
    n_iter :
        The number of iterations to run. The default is 10.
    max_tokens :
        The maximum number of tokens to generate for chat completion. One
        token is roughly one word in plain text, however it can be more per
        word in some cases. The default is 150.

    Returns
    -------
    :
        The results as a dictionary
    """
    prompt_template = (
        "Here is a {text_type} and a statement:\n\n"
        '{text_type}: "{evidence_text}"\n\n'
        'statement: "{statement}"\n\n'
        "{synonyms}"
        "Is the statement implied by the {text_type}?\n"
        "If it isn't, please explain why.\n"
    )
    synonym_template_multiple = (
        "The following list of pairs are synonyms, with the first being the "
        "synonym used in the {text_type} and the second being the synonym used "
        "in the statement:\n{synonyms}\n\n"
    )
    synonym_template_single = (
        '"{in_text}" in the {text_type} is a synonym for "{in_stmt}" in the '
        "statement\n\n"
    )
    start_dt = datetime.utcnow()
    results_dict = {"start_time": start_dt.isoformat(),
                    "error_count": 0,
                    "empty_response_count": 0,
                    "chat_qa": []}
    df_query_str = "tag != 'correct'" if tag is None else f"tag == '{tag}'"
    example_iter = map(tuple, training_data_df.query(df_query_str)[
        ['text', 'english', 'agent_json_list', 'tag']
    ].sample(frac=1.0).values)

    for query_text, query_english, ag_json_list, row_tag in tqdm(
            example_iter, desc="Running explanation queries", total=n_iter
    ):
        text_type = "paragraph" if query_text.count(".") > 1 else "sentence"
        query_synonyms_base = get_synonyms(ag_json_list)
        query_synonyms = parse_synonyms(
            text=query_text, english=query_english,
            agent_json_list=ag_json_list,
            agent_synonyms_list=query_synonyms_base
        )

        # Fill out the synonyms template
        if query_synonyms is None:
            logger.warning(f"Could not get synonyms for {query_text} even "
                           f"though they were needed for the prompt.")
            continue

        if query_synonyms:
            if len(query_synonyms) > 1:
                synonyms_str = synonym_template_multiple.format(
                    text_type=text_type,
                    synonyms="\n".join(
                        f'"{in_text}" and "{in_stmt}"'
                        for in_text, in_stmt in query_synonyms
                    )
                )
            else:
                in_text, in_stmt = query_synonyms[0]
                synonyms_str = synonym_template_single.format(
                    text_type=text_type,
                    in_text=in_text,
                    in_stmt=in_stmt,
                )
        else:
            synonyms_str = ""

        # Fill out the prompt template
        prompt = prompt_template.format(
            text_type=text_type,
            evidence_text=query_text,
            statement=query_english,
            synonyms=synonyms_str,
        )

        # Run the chat completion
        try:
            response = run_openai_chat(prompt=prompt,
                                       max_tokens=max_tokens,
                                       strip=False)
        except Exception as e:
            logger.error(f"Error running OpenAI chat: {e}")
            results_dict["error_count"] += 1
            continue

        # Empty string
        if not response:
            logger.warning("No response from OpenAI")
            results_dict["empty_response_count"] += 1
            continue

        resp_dict = {
            "prompt": prompt,
            "response": response,
            "curation_tag": row_tag,
        }
        results_dict["chat_qa"].append(resp_dict)

        if len(results_dict["chat_qa"]) >= n_iter:
            break

        sleep(0.1)

    end_dt = datetime.utcnow()
    logger.info(f"Finished running {n_iter} queries in "
                f"{(end_dt - start_dt).total_seconds():.2f} seconds.")
    results_dict["end_time"] = end_dt.isoformat()

    # Save the results
    fname = f"explain_incorrect_{start_dt.strftime('%Y%m%d_%H%M%S')}.json"
    (LOCAL_FILES / "results").mkdir(exist_ok=True)
    fpath = LOCAL_FILES / "results" / fname
    logger.info(f"Saving results to {fpath}")
    with open(fpath, "w") as f:
        json.dump(results_dict, f, indent=4)

    return results_dict


def run_stats(
    training_data_df: pd.DataFrame,
    n_iter=100,
    n_pos_examples=2,
    n_neg_examples=2,
    max_tokens=2,
    neg_tag: str = None,
    debug_print: bool = False
):
    """Run chat completion with show-and-tell prompts.

    Parameters
    ----------
    training_data_df :
        The training data.
    n_iter :
        The number of chat completions to run.
    n_pos_examples :
        The number of positive examples to use in the prompt per run.
        Default: 2.
    n_neg_examples :
        The number of negative examples to use in the prompt per run.
        Default: 2.
    max_tokens :
        The maximum number of tokens openai can use for the response.
        Default: 2.
    neg_tag :
        The tag to use for the negative examples. If None, all tags will be
        used. Default: None.
    debug_print :
        If True, the function will print the prompt and the full response
        from the OpenAI API. Default: False.

    Returns
    -------
    :
        A dictionary containing the data about the results of the chat
        completions.
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
        # Convert the agent_info column to a dict
        df["agent_info"] = df["agent_info"].apply(eval)

        return df

    examples_ids = set()

    # Get n positive examples
    if n_pos_examples > 0:
        pos_df = _get_examples_df(positive_examples_path, n_pos_examples)
        examples_ids.update(pos_df["id"].values)
    else:
        pos_df = None

    # Get n negative examples
    if n_neg_examples > 0:
        neg_df = _get_examples_df(negative_examples_path, n_neg_examples)
        examples_ids.update(neg_df["id"].values)
    else:
        neg_df = None

    if n_neg_examples and neg_tag:
        neg_df = neg_df[neg_df["tag"] == neg_tag]
        if neg_df.shape[0] < n_neg_examples:
            logger.warning(f"Only {len(neg_df)} negative examples "
                           f"with tag '{neg_tag}' found. Creating more...")
            save_examples(training_data_df, correct=False)
            neg_df = _get_examples_df(negative_examples_path, n_neg_examples)

    n_iter = min(n_iter, training_data_df.shape[0] - len(examples_ids))
    previous_checks = set()
    start_dt = datetime.utcnow()
    results_dict = {"start_time": start_dt.isoformat(),
                    "error_count": 0,
                    "chat_qa": [],
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "true_negative": 0}

    t = tqdm(desc="Running chat completions", total=n_iter)
    while True:
        # Get one example to check at random from the examples not matching
        # the examples' source_hash or the ones already checked
        excluded_ids = examples_ids | previous_checks
        checker_dict = training_data_df[~training_data_df["id"].isin(
            excluded_ids)].sample(n=1).to_dict(orient="records")[0]

        previous_checks.add(checker_dict["id"])

        # Get the positive examples from the dataframe
        if pos_df is not None:
            pos_examples = list(
                map(
                    tuple,
                    pos_df[
                        ['text', 'english', 'agent_info']
                    ].sample(n=n_pos_examples).values
                )
            )
        else:
            pos_examples = None

        # Get the negative examples from the dataframe
        if neg_df is not None:
            neg_examples = list(
                map(
                    tuple,
                    neg_df[
                        ['text', 'english', 'agent_info']
                    ].sample(n=n_neg_examples).values
                )
            )
        else:
            neg_examples = None

        # Generate the prompt
        prompt = generate_prompt(query_sentence=checker_dict["text"],
                                 query_stmt=checker_dict["english"],
                                 query_agent_info=checker_dict["agent_info"],
                                 pos_ex_list=pos_examples,
                                 neg_ex_list=neg_examples)

        if not prompt:
            logger.warning("No prompt was generated, skipping...")
            continue

        # Run the chat completion
        chat_qa = {"prompt": prompt, "tag": checker_dict["tag"]}
        try:
            choice = run_openai_chat(prompt=prompt,
                                     max_tokens=max_tokens,
                                     debug=debug_print)
        except Exception as e:
            logger.warning(f"Error while running chat completion: {e}")
            chat_qa["response"] = None
            results_dict["chat_qa"].append(chat_qa)
            results_dict["error_count"] += 1
            if results_dict["error_count"] >= n_iter:
                logger.error("Too many errors, stopping...")
                break
            else:
                continue

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

        # Save the prompt, response and the tag
        chat_qa["response"] = choice.lower()
        results_dict["chat_qa"].append(chat_qa)
        t.update(1)
        if t.n >= n_iter:
            break

        sleep(0.1)

    t.close()

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
    fname = start_dt.strftime("correct_vs_incorrect_%Y%m%d_%H%M%S") + ".json"
    LOCAL_FILES.joinpath("results").mkdir(exist_ok=True)
    out_path = LOCAL_FILES.joinpath("results", fname)
    logger.info(f"Saving results to {out_path}")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    return results_dict


def save_examples(training_data_df, correct: bool = True):
    """Save examples of correct or incorrect statements to a file.

    Parameters
    ----------
    training_data_df : pd.DataFrame
        The training data dataframe.
    correct : bool, optional
        Whether to save the correct or incorrect examples, by default True
    """
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
