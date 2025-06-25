import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from time import sleep
from collections import OrderedDict
import logging

from indra.sources.indra_db_rest import get_curations, get_statements_by_hash

logger = logging.getLogger(__name__)


from indra_gpt.chat_curate.check_correctness import (
    positive_examples_path,
    negative_examples_path,
    llm_client,
    generate_negative_expl_prompt,
    generate_correctness_prompt,
    generate_tag_classifier_prompt,
    find_synonyms
)

HERE = Path(__file__).parent


def get_git_revision_hash() -> str:
    """Return the git revision hash."""
    import subprocess

    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE)
        .decode("ascii")
        .strip()
    )

def explain_negative_examples(
    training_data_df: pd.DataFrame,
    tag: str = None,
    n_iter: int = 10,
    max_tokens: int = 150,
    output_dir: Path = None
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

        # Run the chat completion
        try:
            response = llm_client.call(
                prompt=prompt, max_tokens=max_tokens, strip=False
            )
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
    (output_dir / "results").mkdir(exist_ok=True)
    fpath = output_dir / "results" / fname
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
    debug_print: bool = False,
    file_title: str = None,
    output_dir: Path = None,
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
                    f"Only {len(df)} positive examples found. Creating more..."
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
        prompt = generate_correctness_prompt(
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
        try:
            choice = llm_client.call(
                prompt=prompt, max_tokens=max_tokens, debug=debug_print
            )
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
    fname = start_dt.strftime(ftitle)
    output_dir.joinpath("results").mkdir(exist_ok=True)
    out_path = output_dir.joinpath("results", fname)
    logger.info(f"Saving results to {out_path}")
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    return results_dict

def classify_statements(
    training_data_df: pd.DataFrame,
    n_iter: int = 10,
    debug_print: bool = False,
    file_title: str = None,
    max_tokens: int = 100,
    binary: bool = False,
    ignore_tags=None,
    output_dir: Path = None
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
        prompt = generate_tag_classifier_prompt(
            ev_text=text,
            eng_stmt=english,
            agent_info=agent_info,
            binary=binary,
            ignore_tags=ignore_tags,
        )

        # Run the chat completion
        try:
            response = llm_client.call(
                prompt=prompt, max_tokens=max_tokens, debug=debug_print
            )
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
    (output_dir / "results").mkdir(exist_ok=True)
    fpath = output_dir / "results" / fname
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

#################################

def curation_comparison_json(llm_curations: list):
    curation_comparison_json = []
    for llm_curation in llm_curations:
        eng_stmt = llm_curation['eng_stmt']
        pa_hash = llm_curation['pa_hash']
        evidences_curations = llm_curation['evidences_curations']
        for ev_curation in evidences_curations:
            ev_text = ev_curation['sentence']
            source_hash = ev_curation['source_hash']
            prompt = ev_curation['prompt']
            ev_predicted_tag = ev_curation['json_response']['tag']
            ev_predicted_tag_explanation = ev_curation['json_response']['explanation']

            indra_curation = get_curations(pa_hash, source_hash)[0]
            indra_curation['english'] = eng_stmt
            indra_curation['source_hash'] = source_hash
            indra_curation['text'] = ev_text #if not indra_curation['text'] else indra_curation['text']
            indra_curation['prompt'] = prompt
            indra_curation['raw_response'] = ev_curation['raw_response']
            indra_curation['predicted_tag'] = ev_predicted_tag
            indra_curation['predicted_tag_explanation'] = ev_predicted_tag_explanation

            indra_curation = {
                **{k: v for k, v in indra_curation.items() if k not in
                ['english', 'text', 'tag', 'predicted_tag', 'predicted_tag_explanation']},
                'english': indra_curation['english'],
                'text': indra_curation['text'],
                'tag': indra_curation['tag'],
                'predicted_tag': indra_curation['predicted_tag'],
                'predicted_tag_explanation': indra_curation['predicted_tag_explanation'],
            }

            curation_comparison_json.append(indra_curation)
    return curation_comparison_json
