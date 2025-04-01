import json
import logging
from collections import Counter, OrderedDict
from datetime import datetime
from itertools import count
from pathlib import Path
from textwrap import dedent
import time
from time import sleep

import biolookup
import gilda
import pandas as pd
from indra.assemblers.english import EnglishAssembler
from indra.statements import default_ns_order
from indra.statements.io import stmts_from_json_file
from tqdm import tqdm

from indra_gpt.clients.openai.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


HERE = Path(__file__).parent
LOCAL_FILES = HERE.parent.parent.joinpath("local_data")
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


def get_git_revision_hash() -> str:
    """Return the git revision hash."""
    import subprocess

    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE)
        .decode("ascii")
        .strip()
    )


def get_ag_ns_id(db_refs, default):
    """Return a tuple of name space, id from an Agent's db_refs."""
    # Note that only the first namespace in the default_ns_order 
    # that is found in the db_refs will be returned. This might
    # not be the best way to handle this, but it is the current 
    # behavior. If this db is not a good one this is a problem. 
    for ns in default_ns_order:
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
    # db, _id = get_ag_ns_id(db_refs, name)

    # The following code block finds the first namespace in the default_ns_order
    # that has non-empty synonym list when using gilda.
    synonyms = []
    for ns in default_ns_order:
        if ns in db_refs:
            _id = db_refs[ns]
            synonyms = gilda.get_names(ns, _id)
            if len(synonyms) > 0:
                break

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
        English statement. A synonym is None if it is not found in the text
        or the statement.
    """

    def _clean(s):
        return (
            s.replace("(", " ")
            .replace(")", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("?", " ")
            .replace("!", " ")
            .replace(",", " ")
            .replace(".", " ")
            .replace("/", " ")
            .replace("  ", " ")
        )

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

        in_text, in_stmt = find_synonyms(
            ev_text, english, synonyms, case_sensitive=False, substring_match=True
        )

        ag_info[curie] = {
            "name": name,
            "synonyms": synonyms,
            "definition": definition,
            "syn_in_text": in_text,
            "syn_in_stmt": in_stmt,
        }

    return ag_info


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
    # Return the pre-existing default training data if user did not specify any files. 
    if  curation_training_data.exists() and not curations_file and not statement_json_file and not refresh and not test:
        df = pd.read_csv(curation_training_data, sep="\t")
        if isinstance(df["agent_json_list"][0], str):
            logger.info(
                "agent_json_list dtype is str, using eval to convert "
                "to list of OrderedDicts"
            )
            # Apply 'eval' to the column and provide the OrderedDict class
            # as an arg variable to be used in the eval call
            df["agent_json_list"] = df["agent_json_list"].apply(
                eval, args=({"OrderedDict": OrderedDict},)
            )
        if isinstance(df["agent_info"][0], str):
            logger.info("agent_info dtype is str, using eval to convert " "to dict")
            df["agent_info"] = df["agent_info"].apply(eval)
        return df

    # Else generate the training data
    # It both files are not provided raise error. 
    if  curations_file is None or statement_json_file is None:
        raise FileNotFoundError(
            f"Please provide the curations and statement json file if "
            f"pre-generated training data is not available at "
            f"{curation_training_data}"
        )
    # Else if both files are provided, generate the training data
    # Initialize Gilda so that tqdm is more accurate
    _ = gilda.get_grounder()
    logger.info("Loading curations")
    curs = json.load(open(curations_file, "r"))
    logger.info("Loading statements")
    stmts = stmts_from_json_file(statement_json_file)
    stmts_by_hash = {
        s.get_hash(): s for s in tqdm(stmts, desc="Creating statement lookup")
    }

    # Loop the curations, get the corresponding statement with evidence and
    # extend the curation with the evidence text and english assembled
    # statement
    # Basically a join operation between the curations and the statements by
    # the hashes. 
    curation_data = []
    skipped = 0
    for cur in tqdm(curs, desc="Matching curations to statements"):
        # If the pre-assembled has of curatation is not in the statements, skip this curation data.
        try:
            stmt = stmts_by_hash[cur["pa_hash"]]
        except KeyError:
            logger.warning(f"Skipping {cur['pa_hash']}, not found in statements")
            skipped += 1
            continue
        
        # Get the evidence text (get the first evidence text if there are multiple)
        # If there aren't any evidence texts, skip this curation data.
        ev = [e for e in stmt.evidence if e.get_source_hash() == cur["source_hash"]][0]
        if ev.text is None:
            skipped += 1
            continue
        
        # Extend the curation data with: Evidence text, English statement, Agent list, and Agent info
        cur["text"] = ev.text
        eng_stmt = EnglishAssembler([stmt]).make_model()
        cur["english"] = eng_stmt
        ag_list = stmt.agent_list()
        logger.debug(f"Agents: {ag_list}")
        cur["statement_json"] = stmt.to_json()
        cur["statement_indra"] = stmt
        cur["agent_json_list"] = [a.to_json() for a in ag_list]
        cur["agent_info"] = get_agent_info(
            ev_text=ev.text, english=eng_stmt, ag_list=ag_list
        )

        curation_data.append(cur)
        if test and len(curation_data) == 10:
            logger.info("Test mode: Breaking after 10 examples - not saving data")
            break

    if skipped:
        logger.info(f"Skipped {skipped} examples due to missing synonyms")

    # Save the training data
    df = pd.DataFrame(curation_data)
    if not test:
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        output_dir = LOCAL_FILES / "training_data"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"training_data_{current_time}.tsv"
        logger.info(f"Saving training data to {output_file}")
        df.to_csv(output_file, sep="\t", index=False)

    return df


def generate_synonym_str(
    agents_info, include_def: bool = True, index: int = None
) -> str:
    """Generate a string with the list of synonyms

    Parameters
    ----------
    agents_info :
        A dictionary with agent information for each agent in the
        statement keyed by curie. Each agent dictionary has the name,
        synonyms, and definition of the agent.
    include_def :
        If True, include the definition of the agent in the string.
        Default: True.
    index :
        If provided, is the index of the sentence and statement. If None,
        the index will not be included in the string.
    """
    # Format the synonyms to something like:
    # """The definition of {name1}{ex_str} is: "{definition1}".
    # The definition of {name2}{ex_str} is: "{definition2}".
    # The statement{ex_str} assumes that "{name1}" is the same as "{synonym1}"
    # and "{name2}" is the same as "{synonym2}"."""
    index_str = f"in example {index}" if index is not None else ""
    def_fmt = 'The definition of {name}%s is: "{definition}".\n' % index_str
    syn_str_intro = (
        'The statement%s assumes that "{name}" is the same '
        'as "{synonym}"' % index_str
    )
    syn_str_contd = '{comma}{and_} "{name}" is the same as "{synonym}"'
    def_str = ""
    syn_strs = []
    for curie, agent_info in agents_info.items():
        in_text = agent_info["syn_in_text"]
        in_stmt = agent_info["syn_in_stmt"]
        synonyms = set(agent_info["synonyms"]) - {in_stmt}
        name = in_stmt or in_text
        if len(synonyms) == 0 or name is None:
            # No string to generate
            continue

        if include_def:
            if agent_info["definition"]:
                definition = agent_info["definition"]
            else:
                try:
                    res = biolookup.lookup(curie)
                    definition = res.get("definition", "<Definition not found from biolookup>")
                except Exception as e:
                    logger.error(f"Error looking up {curie}: {e}")
                    definition = "<Definition not found from biolookup>"

            def_str += (
                def_fmt.format(name=in_stmt, definition=definition)
                if definition
                else ""
            )

        if in_text and in_stmt:
            # 1. 'real' synonyms
            if in_text != in_stmt:
                syn_strs.append((in_stmt, in_text))
            # 2. They are the same, no need to add a synonym
            else:
                continue
        else:
            # 3. In statement but not in text, continue
            if in_text is None and in_stmt or in_text and in_stmt is None:
                #  Use continue to test the assumption that chat-gpt gets
                # more confused by listing synonyms that don't appear in the
                # text
                # continue

                # List the synonyms to test the assumption that some of the
                # synonyms are descriptive enough to clarify the meaning of
                # the statement and text pair rather than confuse chat-gpt.
                synonyms = list(synonyms)
                if len(synonyms) == 1:
                    synonyms = synonyms[0]
                elif len(synonyms) == 2:
                    synonyms = '"' + '" and "'.join(synonyms) + '"'
                elif len(synonyms) > 2:
                    synonyms = (
                        '"'
                        + '", "'.join(synonyms[:-1])
                        + '", and "'
                        + synonyms[-1]
                        + '"'
                    )
                else:
                    #
                    pass
                syn_strs.append((name, synonyms))
            else:
                # 4. neither in text nor in statement (should already be
                #    handled above)
                continue

    if len(syn_strs) == 1:
        _and = ""
        comma = ""
    elif len(syn_strs) == 2:
        _and = " and"
        comma = ""
    else:
        _and = " and"
        comma = ","
    syn_str = ""
    for ix, (name, synonym) in enumerate(syn_strs):
        syn_str_fmt = syn_str_intro if ix == 0 else syn_str_contd
        and_ = "" if ix < len(syn_strs) - 1 else _and
        fmt_dict = {"name": name, "synonym": synonym}
        if ix > 0:
            fmt_dict["and_"] = and_
            fmt_dict["comma"] = comma
        syn_str += syn_str_fmt.format(**fmt_dict)

    return def_str + syn_str


def generate_example(sentence, statement, agents_info=None, index: int = None) -> str:
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

    example_template = '{sent_str}: "{sentence}"\n{stmt_str}: "{english}"{synonyms}\n'
    if agents_info:
        syn_str = "\n" + generate_synonym_str(agents_info, index)
    else:
        syn_str = "\n"
    return (
        example_template.format(
            sent_str=sent_str,
            sentence=sentence,
            stmt_str=stmt_str,
            english=statement,
            synonyms=syn_str,
        )
        + "\n"
    )


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
    pos_str = (
        "The following sentences are paired with statements that are "
        "implied from their sentence:\n\n"
    )
    neg_str = (
        "The following sentences do not imply the statements they "
        "are paired with:\n\n"
    )
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
    query_str = (
        "Is the following statement implied by the sentence, "
        "assuming the sentence and the statement follow the same "
        "pattern as in the examples above?\n\n"
    )
    query_str += generate_example(query_sentence, query_stmt, agents_info)
    return query_str


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
    # Get positive and negative examples
    indexer = count(1)
    if pos_ex_list is not None:
        pos_ex_str = generate_example_list(pos_ex_list, correct=True, indexer=indexer)
    else:
        pos_ex_str = ""

    if neg_ex_list is not None:
        neg_ex_str = generate_example_list(neg_ex_list, correct=False, indexer=indexer)
    else:
        neg_ex_str = ""

    examples_str = (
        pos_ex_str + neg_ex_str + "\n=======\n" if pos_ex_str or neg_ex_str else ""
    )

    # Generate query string
    query_str = generate_query_str(query_sentence, query_stmt, query_agent_info)

    # Generate positive and negative examples
    prmt = default_prompt_template.format(examples=examples_str, query=query_str)
    return prmt


def generate_negative_expl_prompt(
    query_text: str, query_stmt: str, query_agent_info
) -> str:
    """Generate a prompt for negative examples"""
    text_type = "paragraph" if query_text.count(".") > 1 else "text"
    prompt_template = (
        "Here is a {text_type} and a statement:\n\n"
        '{text_type}: "{evidence_text}"\n\n'
        'statement: "{statement}"\n\n'
        "{synonyms}\n\n"
        "Is the statement implied by the {text_type}?\n"
        "If it isn't, please explain why.\n"
    )
    syn_str = generate_synonym_str(query_agent_info)
    return prompt_template.format(
        text_type=text_type,
        evidence_text=query_text,
        statement=query_stmt,
        synonyms=syn_str,
    )


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
    start_dt = datetime.utcnow()
    results_dict = {
        "start_time": start_dt.isoformat(),
        "git_revision": get_git_revision_hash(),
        "error_count": 0,
        "empty_response_count": 0,
        "chat_qa": [],
    }
    df_query_str = "tag != 'correct'" if tag is None else f"tag == '{tag}'"
    example_iter = map(
        tuple,
        training_data_df.query(df_query_str)[["text", "english", "agent_info", "tag"]]
        .sample(frac=1.0)
        .values,
    )

    for query_text, query_english, ag_info, row_tag in tqdm(
        example_iter, desc="Running explanation queries", total=n_iter
    ):
        # Get the prompt
        prompt = generate_negative_expl_prompt(
            query_text=query_text, query_stmt=query_english, query_agent_info=ag_info
        )
        chat_prompt = {"role": "user", "content": prompt}
        # Run the chat completion
        try:
            client = OpenAIClient()
            response = client.get_response_single_inference(chat_prompt, 
                                                            max_tokens=max_tokens, 
                                                            strip=False).choices[0].message.content
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
    logger.info(
        f"Finished running {n_iter} queries in "
        f"{(end_dt - start_dt).total_seconds():.2f} seconds."
    )
    results_dict["end_time"] = end_dt.isoformat()

    # Save the results
    fname = f"explain_incorrect_{start_dt.strftime('%Y%m%d_%H%M%S')}.json"
    (LOCAL_FILES / "results").mkdir(exist_ok=True)
    fpath = LOCAL_FILES / "results" / fname
    logger.info(f"Saving results to {fpath}")
    with open(fpath, "w") as f:
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
        out_file = positive_examples_path
        query_str = 'tag == "correct"'

    else:
        out_file = negative_examples_path
        query_str = 'tag != "correct"'

    if out_file.exists():
        saved_df = pd.read_csv(out_file, sep="\t")
        saved_tags = list(saved_df["tag"])
        saved_ids = set(saved_df["id"])
        row_iter = training_data_df[~training_data_df["id"].isin(saved_ids)].query(
            query_str
        )
    else:
        row_iter = training_data_df.query(query_str)

    for row in row_iter.sample(frac=1.0).itertuples():
        if correct:
            assert row.tag == "correct"
        else:
            assert row.tag != "correct"

        ags_info_dict = row.agent_info
        syn_pairs = []
        for curie, info in ags_info_dict.items():
            in_text, in_stmt = find_synonyms(
                row.text,
                row.english,
                info["synonyms"],
                case_sensitive=False,
                substring_match=True,
            )
            if in_text or in_stmt:
                syn_pairs.append((in_text, in_stmt))
        if len(syn_pairs) != len(ags_info_dict):
            skip = (
                "> > Not all synonyms were found in the sentence and "
                "statement, recommend skipping this one\n"
            )
        else:
            skip = ""
        syn_pairs_str = "\n".join([f"{s[0]} - {s[1]}" for s in syn_pairs])

        if not correct:
            tag_distr = ", ".join(
                f"{t}: {c}" for t, c in Counter(saved_tags).most_common()
            )
            tag_str = f"Current Tag: {row.tag}\nSaved tags: {tag_distr}\n"
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
            choice = input(
                f"File {out_file} already exists (Size "
                f"{file_size} B). Overwrite, Append or Cancel? "
                f"(o/a/c): "
            )
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


def run_stats(
    training_data_df: pd.DataFrame,
    n_iter=100,
    n_pos_examples=2,
    n_neg_examples=2,
    max_tokens=2,
    neg_tag: str = None,
    debug_print: bool = False,
    file_title: str = None,
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
    file_title :
        The title to use for the file name. If None, the current date and
        time will be used. Default: None.

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
                logger.info(
                    f"Only {len(df)} positive examples " "found. Creating more..."
                )
                save_examples(training_data_df, correct=True)
                df = pd.read_csv(examples_path, sep="\t")
        else:
            logger.info("No examples found. Creating...")
            save_examples(training_data_df, correct=True)
            df = pd.read_csv(examples_path, sep="\t")

        # Convert the agent json string to a list of agent jsons
        df["agent_json_list"] = df["agent_json_list"].apply(
            eval, args=({"OrderedDict": OrderedDict},)
        )
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
            logger.warning(
                f"Only {len(neg_df)} negative examples "
                f"with tag '{neg_tag}' found. Creating more..."
            )
            save_examples(training_data_df, correct=False)
            neg_df = _get_examples_df(negative_examples_path, n_neg_examples)

    n_iter = min(n_iter, training_data_df.shape[0] - len(examples_ids))
    previous_checks = set()
    start_dt = datetime.utcnow()
    results_dict = {
        "start_time": start_dt.isoformat(),
        "git_revision": get_git_revision_hash(),
        "error_count": 0,
        "chat_qa": [],
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
    }

    t = tqdm(desc="Running chat completions", total=n_iter)
    while True:
        # Get one example to check at random from the examples not matching
        # the examples' source_hash or the ones already checked
        excluded_ids = examples_ids | previous_checks
        checker_dict = (
            training_data_df[~training_data_df["id"].isin(excluded_ids)]
            .sample(n=1)
            .to_dict(orient="records")[0]
        )

        previous_checks.add(checker_dict["id"])

        # Get the positive examples from the dataframe
        if pos_df is not None:
            pos_examples = list(
                map(
                    tuple,
                    pos_df[["text", "english", "agent_info"]]
                    .sample(n=n_pos_examples)
                    .values,
                )
            )
        else:
            pos_examples = None

        # Get the negative examples from the dataframe
        if neg_df is not None:
            neg_examples = list(
                map(
                    tuple,
                    neg_df[["text", "english", "agent_info"]]
                    .sample(n=n_neg_examples)
                    .values,
                )
            )
        else:
            neg_examples = None

        # Generate the prompt
        prompt = generate_prompt(
            query_sentence=checker_dict["text"],
            query_stmt=checker_dict["english"],
            query_agent_info=checker_dict["agent_info"],
            pos_ex_list=pos_examples,
            neg_ex_list=neg_examples,
        )

        if not prompt:
            logger.warning("No prompt was generated, skipping...")
            continue

        # Run the chat completion
        chat_qa = {"prompt": prompt, "tag": checker_dict["tag"]}
        chat_prompt = {"role": "user", "content": prompt}
        # Run the chat completion
        try:
            client = OpenAIClient()
            choice = client.get_response_single_inference(chat_prompt, 
                                                            max_tokens=max_tokens, 
                                                            debug=debug_print,
                                                            strip=False).choices[0].message.content
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
    print(
        pd.DataFrame(
            data=[[tp, fp], [fn, tn]],
            index=["gpt_correct", "gpt_incorrect"],
            columns=["correct", "incorrect"],
        )
    )
    print(f"Precision: {results_dict['precision']}")
    print(f"Recall: {results_dict['recall']}")
    print(f"Accuracy: {results_dict['accuracy']}")
    print(f"Total examples: {N}")

    # Save the results
    if file_title and not file_title.endswith("_"):
        file_title += "_"
    ftitle = (file_title or "") + "correct_vs_incorrect_%Y%m%d_%H%M%S.json"
    logger.info(f"Saving results to {ftitle}")
    fname = start_dt.strftime(ftitle)
    LOCAL_FILES.joinpath("results").mkdir(exist_ok=True)
    out_path = LOCAL_FILES.joinpath("results", fname)
    logger.info(f"Saving results to {out_path}")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    return results_dict


def generate_classifier_prompt(
    ev_text: str,
    eng_stmt: str,
    agent_info,
    ignore_tags=None,
) -> str:
    """Generate a prompt for the classifier.

    Parameters
    ----------
    ev_text :
        The evidence text.
    eng_stmt :
        The English statement.
    agent_info :
        The agent info.
    ignore_tags :
        The tags to ignore. Default: None.

    Returns
    -------
    :
        The prompt as a string.

    """
    # Follows the tags available in the training data
    curation_tags = {
        "other": "When no other tag is applicable, use this tag.",
        "correct": "The statement is correct and is implied by the sentence.",
        "no_relation": "This tag is applicable if the sentence does not "
        "imply a relationship between the agents appearing in "
        "the Statement.",
        "wrong_relation": "This tag is applicable if the sentence implies a "
        "relationship between the entities appearing in "
        "the statement but the type of statement is "
        "inconsistent with the sentence.",
        "mod_site": "This tag is applicable if an amino-acid site is missing "
        "or is incorrect in a statement implying a modification, "
        "but the statement is otherwise correct. Example: "
        'sentence: "MAP2K1 phosphorylates MAPK1 at T185."; '
        'statement: Statement: "Phosphorylation(MAP2K1(), '
        'MAPK1())"',
        "hypothesis": "This tag is applicable if the sentence describes a "
        "hypothesis or an experiment or is otherwise "
        "speculative, rather than a result or mechanism.",
        "negative_result": "This tag is applicable if the sentence implies "
        "the lack of or opposite of a relationship.",
        "grounding": "This tag is applicable when one of the named entities "
        "in the statement is assigned an incorrect database "
        "identifier and therefore refers to the wrong entity.",
        "entity_boundaries": "This tag is applicable when one of the named "
        "entities in the statement is misidentified "
        'from a too greedy match, e.g. "gap" vs "gap '
        'junction", an ambiguous acronym, e.g. "IR" for '
        'infrared radiation vs insulin receptor", or '
        "similar.",
        "act_vs_amt": "This tag is applicable when the sentence implies a "
        "regulation of amount but the corresponding statement "
        "implies regulation of activity or vice versa.",
        "polarity": "This tag is applicable if a statement was correctly "
        "extracted but for polarity of the statement, "
        "e.g. Activation instead of Inhibition, "
        "or Phosphorylation instead of Dephosphorylation.",
        "agent_conditions": "This tag is applicable if one of the "
        "named entities (i.e. agents) in the statement is "
        "missing relevant conditions that are mentioned "
        "in the sentence, or has incorrect conditions "
        "attached to it, but the statement is otherwise "
        'correct. Example: sentence "Mutant BRAF '
        'activates MEK"; statement: "BRAF activates MEK".',
    }
    prompt_templ = dedent(
        """
    Here is a list of tags and descriptions describing how a sentence - statement pair can be classified:
    
    {tag_descriptions}
    
    Please help me put the right tag to the following sentence - statement pair:
    
    Sentence: {sentence}
    Statement: {statement}
    {synonyms}"""
    )
    ignore_tags = ignore_tags or []
    tag_desc = "\n".join(
        [
            f"{tag}: {description}"
            for tag, description in curation_tags.items()
            if tag not in ignore_tags
        ]
    )
    synonyms = generate_synonym_str(agents_info=agent_info)
    prompt = prompt_templ.format(
        tag_descriptions=tag_desc,
        sentence=ev_text,
        statement=eng_stmt,
        synonyms=synonyms,
    )
    return prompt


def classify_statements(
    training_data_df: pd.DataFrame,
    n_iter: int = 10,
    debug_print: bool = False,
    file_title: str = None,
    max_tokens: int = 100,
    ignore_tags=None,
):
    """Classify statements according to the valid curation tags"""
    start_dt = datetime.utcnow()
    results_dict = {
        "start_time": start_dt.isoformat(),
        "git_revision": get_git_revision_hash(),
        "error_count": 0,
        "empty_response_count": 0,
        "chat_qa": [],
    }

    row_iter = map(
        tuple,
        training_data_df[["text", "english", "agent_info", "tag"]]
        .sample(frac=1.0)
        .values,
    )

    for text, english, agent_info, curation_tag in tqdm(
        row_iter, desc="Classifying", total=n_iter
    ):
        if curation_tag in ignore_tags:
            continue

        # Generate the prompt
        prompt = generate_classifier_prompt(
            ev_text=text,
            eng_stmt=english,
            agent_info=agent_info,
            ignore_tags=ignore_tags,
        )

        # Run the chat completion
        chat_prompt = {"role": "user", "content": prompt}
        try:
            client = OpenAIClient()
            response = client.get_response_single_inference(chat_prompt, 
                                                            max_tokens=max_tokens, 
                                                            debug=debug_print,
                                                            strip=False).choices[0].message.content
        except Exception as e:
            logger.warning(f"Error while running chat completion: {e}")
            results_dict["error_count"] += 1
            continue

        # Empty string in response
        if not response:
            results_dict["empty_response_count"] += 1
            continue

        resp_dict = {
            "prompt": prompt,
            "response": response,
            "curation_tag": curation_tag,
        }
        results_dict["chat_qa"].append(resp_dict)

        if len(results_dict["chat_qa"]) >= n_iter:
            break

        sleep(0.1)

    end_dt = datetime.utcnow()
    logger.info(
        f"Finished running {n_iter} classification queries in "
        f"{(end_dt - start_dt).total_seconds():.2f} seconds."
    )
    results_dict["end_time"] = end_dt.isoformat()

    # Save the results
    if file_title and not file_title.endswith("_"):
        file_title += "_"

    fname = (
        file_title or ""
    ) + f"classification_{start_dt.strftime('%Y%m%d_%H%M%S')}.json"
    (LOCAL_FILES / "results").mkdir(exist_ok=True)
    fpath = LOCAL_FILES / "results" / fname
    logger.info(f"Saving results to {fpath}")
    with open(fpath, "w") as f:
        json.dump(results_dict, f, indent=4)

    return results_dict
