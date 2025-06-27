import json
import logging
from collections import OrderedDict
from itertools import count
from pathlib import Path
from textwrap import dedent
import random
import os

import biolookup
import gilda
import pandas as pd
from indra.assemblers.english import EnglishAssembler
from indra.statements import default_ns_order
from indra.statements.io import stmts_from_json_file
from tqdm import tqdm

from indra_gpt.clients.llm_client import LitellmClient
from indra_gpt.utils.prompt import get_schema_wrapped_prompt

logger = logging.getLogger(__name__)


HERE = Path(__file__).parent
RESOURCES = HERE.parent.joinpath("resources")
curation_training_data = RESOURCES.joinpath("training_data.tsv")
positive_examples_path = RESOURCES.joinpath("positive_examples.tsv")
negative_examples_path = RESOURCES.joinpath("negative_examples.tsv")

default_prompt_template = """{examples}
{query}Please answer with just 'correct' or 'incorrect'."""

CURATION_TAGS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tag": {
            "type": "string",
            "description": "The curation tag indicating the classification of the sentence-statement pair."
        },
        "explanation": {"type": "string", "description": "An explanation for the curation tag."}
    },
    "required": ["tag", "explanation"],
    "additionalProperties": False,
    "description": "A JSON schema for the curation tags for the sentence-statement pair.",
}


def get_ag_ns_id(db_refs, default):
    """Return a tuple of name space, id from an Agent's db_refs."""
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
        # Skip agents that are None or have no db_refs
        if not ag or not ag.db_refs:
            logger.info(
                f"Skipping agent {ag} with no db_refs or None value in "
                "the agent list."
            )
            continue

        db_refs = ag.db_refs
        name = ag.name or ag.db_refs.get("TEXT")

        try:
            curie = ":".join(ag.get_grounding())
        except Exception as e:
            logger.info(
                f"Skipping agent {ag} with no grounding: {e}"
            )
            continue

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
    if curation_training_data.exists() and not refresh and not test:
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
    stmts_by_hash = {
        s.get_hash(): s for s in tqdm(stmts, desc="Creating statement lookup")
    }

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
        logger.info(f"Saving training data to {curation_training_data}")
        df.to_csv(curation_training_data, sep="\t", index=False)

    return df

def get_examples(examples_path: Path, n_examples: int):
    df = pd.read_csv(examples_path, sep="\t").sample(n=n_examples, random_state=42)
    examples = []
    for ix, row in df.iterrows():
        ev_text = row["text"]
        english = row["english"]
        agent_info = eval(row["agent_info"]) if row["agent_info"] else None
        tag = row["tag"] if "tag" in row else None
        example = (ev_text, english, agent_info, tag)

        examples.append(example)
    return examples

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
    index_str = f" in example {index}" if index is not None else ""
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
            continue

        if include_def:
            if agent_info["definition"]:
                definition = agent_info["definition"]
            else:
                res = biolookup.lookup(curie)
                definition = res.get("definition", "")

            def_str += (
                def_fmt.format(name=in_stmt, definition=definition)
                if definition
                else ""
            )

        if in_text and in_stmt and in_text == in_stmt:
            continue
        else:
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
                pass
            syn_strs.append((name, synonyms))

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

def generate_example(sentence, statement, agents_info=None, tag=None, index: int = None) -> str:
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
    tag :
        The curation tag for the example.
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

    example_template = """
{sent_str}: "{sentence}"
{stmt_str}: "{english}"
{synonym_str}
Tag: "{tag}"
    """
    if agents_info:
        synonym_str = "\n" + generate_synonym_str(agents_info, index)
    else:
        synonym_str = "\n"
    return (
        example_template.format(
            sent_str=sent_str,
            sentence=sentence,
            stmt_str=stmt_str,
            english=statement,
            synonym_str=synonym_str,
            tag=tag,
        )
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
        "implied from their sentence:\n"
    )
    neg_str = (
        "The following sentences do not imply the statements they "
        "are paired with:\n"
    )
    template = pos_str if correct else neg_str
    for sentence, statement, agents_info, tag in examples:
        ix = next(indexer)
        ex_str = generate_example(sentence, statement, agents_info, tag, ix)
        template += ex_str
    return template

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


def generate_tag_classifier_prompt(
    ev_text: str,
    eng_stmt: str,
    agent_info,
    binary_classification: bool = False,
    ignore_tags=None,
    pos_examples=None,
    neg_examples=None,

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
    binary :
        If True, we only consider the "correct" and "incorrect" tags.
    ignore_tags :
        The tags to ignore. Default: None.

    Returns
    -------
    :
        The prompt as a string.

    """
    # Follows the tags available in the training data
    curation_tags = {
        "correct": "The statement is implied by the sentence.",
        "incorrect": "The statement is not implied by the sentence.",
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
        "other": "When no other tag is applicable, use this tag.",
    }

    # Get positive and negative examples
    indexer = count(1)
    if pos_examples is not None:
        pos_ex_str = generate_example_list(pos_examples, correct=True, indexer=indexer)
    else:
        pos_ex_str = ""

    if neg_examples is not None:
        neg_ex_str = generate_example_list(neg_examples, correct=False, indexer=indexer)
    else:
        neg_ex_str = ""

    prompt_templ = dedent(
        """
        You are given a sentence and a candidate statement. Your task is to assign a curation tag indicating whether the statement is implied by the sentence.

        Here is a list of available tags and their meanings:

        {tag_descriptions}

        Here are some examples of sentence-statement pairs with their corresponding tags:

        Positive examples (the statement is implied by the sentence):
        {pos_ex_str}

        Negative examples (the statement is not implied by the sentence):
        {neg_ex_str}

        Carefully read the following pair and choose the appropriate tag:

        Sentence:
        {sentence}

        Statement:
        {statement}

        Entity definitions and synonyms (if provided):
        {synonyms}

        Guidelines for assigning the tag:

        - Only consider what is directly stated in the sentence, along with any provided definitions and synonyms.
        - If the relationship expressed in the statement is synonymous with one explicitly stated in the sentence, the statement is considered implied.
        - If the sentence discusses the **mechanism**, **specificity**, or **context** of a relationship, you may assume that the relationship itself is implied.
        - Do not rely on external background knowledge unless the implication logically follows from the sentence.
        - If the sentence expresses uncertainty about *how* something happens, but not *whether* it happens, the underlying relationship is still implied.

        Respond only with the appropriate tag and a brief explanation in the required schema format.
        """
    )
    if binary_classification:
        tag_desc = "\n".join(
            [
                f"{tag}: {description}"
                for tag, description in curation_tags.items()
                if tag in ["correct", "incorrect"]
            ]
        )
    else:
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
        pos_ex_str=pos_ex_str,
        neg_ex_str=neg_ex_str
    )
    return prompt

def chat_curate_stmt(stmt,
                     n_evidence_to_curate=None,
                     decision_threshold=None,
                     binary_classification=False,
                     ignore_tags=['incorrect'],
                     pos_examples=None,
                     neg_examples=None,
                     client=None,
                     ):
    eng_stmt = EnglishAssembler([stmt]).make_model()
    indra_stmt_str = str(stmt)

    ag_list = stmt.agent_list()
    pa_hash = stmt.get_hash()
    all_evidences = stmt.evidence

    # Determine which evidences to curate
    if n_evidence_to_curate is not None and n_evidence_to_curate < len(all_evidences):
        curated_evs = set(random.sample(all_evidences, n_evidence_to_curate))
    else:
        curated_evs = set(all_evidences)

    stmt_curation = {
        "eng_stmt": eng_stmt,
        "indra_stmt_str": indra_stmt_str,
        "pa_hash": pa_hash,
        "evidences_curations": [],
        "predicted_tags": [],
    }

    curated_tags = []

    for ev in curated_evs:
        ev_text = ev.text
        if ev_text is None:
            logger.info(f"Skipping evidence with no text for statement {pa_hash}: {ev}")
            continue
        agent_info = get_agent_info(ev_text, eng_stmt, ag_list)

        # Generate prompt
        prompt = generate_tag_classifier_prompt(
            ev_text=ev_text,
            eng_stmt=f"English statement: {eng_stmt}\nINDRA statement: {indra_stmt_str}",
            agent_info=agent_info,
            binary_classification=binary_classification,
            ignore_tags=ignore_tags,
            pos_examples=pos_examples,
            neg_examples=neg_examples
        )

        # Wrap and send prompt
        schema_wrapped_prompt = get_schema_wrapped_prompt(prompt, CURATION_TAGS_JSON_SCHEMA)
        raw_response = client.call(schema_wrapped_prompt)
        try:
            json_response = json.loads(raw_response)
        except json.JSONDecodeError:
            print(f"Error decoding JSON response: {raw_response}")
            json_response = None

        tag = json_response.get("tag") if json_response else None
        curated_tags.append(tag)
        stmt_curation['evidences_curations'].append({
            "sentence": ev_text,
            "source_hash": ev.source_hash,
            "prompt": schema_wrapped_prompt,
            "raw_response": raw_response,
            "json_response": json_response,
        })
        stmt_curation['predicted_tags'].append(tag)

    # Optional overall prediction
    if decision_threshold is not None and curated_tags:
        num_correct = sum(1 for tag in curated_tags if tag == 'correct')
        prop_correct = num_correct / len(curated_tags)
        stmt_curation['proportion_correct'] = round(prop_correct, 2)
        stmt_curation['overall_prediction'] = (
            'correct' if prop_correct >= decision_threshold else 'incorrect'
        )
    return stmt_curation

def chat_curate_stmts(indra_stmts,
                      n_evidence_to_curate=5,
                      decision_threshold=0.5,
                      binary_classification=False,
                      ignore_tags=['incorrect'],
                      n_fewshot_examples=0,
                      pos_examples_path=positive_examples_path,
                      neg_examples_path=negative_examples_path,
                      model="gpt-4o-mini",
                      ):
    
    llm_client = LitellmClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model,
        response_format="json"
    )
    
    pos_examples = get_examples(pos_examples_path, n_examples=n_fewshot_examples)
    neg_examples = get_examples(neg_examples_path, n_examples=n_fewshot_examples)

    return [
        chat_curate_stmt(stmt,
                         n_evidence_to_curate=n_evidence_to_curate,
                         decision_threshold=decision_threshold,
                         binary_classification=binary_classification,
                         ignore_tags=ignore_tags,
                         pos_examples=pos_examples,
                         neg_examples=neg_examples,
                         client=llm_client,
        )               
        for stmt in tqdm(indra_stmts, desc="Curating statements with chat curation")
    ]
