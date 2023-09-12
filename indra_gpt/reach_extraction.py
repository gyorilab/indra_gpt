"""This module contains functions for extracting statements from text using
first ChatGPT and then REACH."""
import argparse

import pandas as pd

from indra_gpt.api import run_openai_chat
from indra.sources import reach

from indra.statements.io import stmts_to_json_file

import json


def run_chat_gpt_on_ev_text(ev_text: str, examples, debug=False) -> str:
    """ takes in ev_text and returns english statement

    Parameters
    ----------
    ev_text :
        the text to generate an english statement from
    examples :
        a list of lists where the first item is the english statement and the
        second item is the evidence text that supports the english statement

    Returns
    -------
    :
        english statement from ChatGPT
    """


    prompt_templ = 'Extract the relation from this sentence:  \n"{' \
                   'prompt}"'
    history = [
        {"role": "user",
         "content": prompt_templ.format(prompt=examples[0][1])},
        {"role": "assistant",
         "content": examples[0][0]},
        {"role": "user",
         "content": prompt_templ.format(prompt=examples[1][1])},
        {"role": "assistant",
         "content": examples[1][0]}
    ]

    prompt = prompt_templ.format(prompt=ev_text)

    chat_gpt_english = run_openai_chat(prompt, chat_history=history,
                                       max_tokens=25, strip=False, debug=debug)
    return chat_gpt_english


def main(training_df, n_statements=10, debug=False):
    # Ensure we only use statements curated as correct
    training_df = training_df[training_df['tag'] == 'correct']

    # Loop over the training data and extract english statements from the

    # chatGPT output and save to a list
    gpt_english_statements = []
    statistics = []

    for item in training_df[['pa_hash','source_hash','text','english']].values:
        pa_hash, source_hash,text, english = item
        examples = training_df[['english','text']].sample(2).values
        gpt_english = run_chat_gpt_on_ev_text(text,examples,debug=debug)
        gpt_english_statements.append(gpt_english)
        statistics.append((
            pa_hash, source_hash, text, english, gpt_english
        ))
        if len(gpt_english_statements)==n_statements:
            break

    # Concatenate the chatGPT output to a single string with one english
    # statement per line
    gpt_englsh_statements_str = '\n'.join(gpt_english_statements)

    # Run REACH on the chatGPT output
    reach_processor = reach.process_text(gpt_englsh_statements_str,
                                         url=reach.local_text_url)

    # Compare with original statements and check if ChatGPT+REACH
    # extracted the same statements as the original statements

    return reach_processor, statistics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_df', type=str, required=True)
    parser.add_argument('--n_statements', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    training_data_df = pd.read_csv(args.training_df, sep='\t')

    reach_processor, statistics = main(training_data_df, args.n_statements,
                                       debug=args.debug)
    # save statements in reach_processor in json or pickle into a file
    # save the statistics
    # change the prompt so ChatGPT writes something like the statement we want

# 'reach_processor'+'_('+prompt+')_'+'.json'

    #with open('reach_processor.json', 'w') as f:
       #json.dump(reach_processor, f)

    stmts_to_json_file(stmts=reach_processor.statements,
                       fname='reach_statements_9.json')
    #stmts_to_json_file(stmts=statistics,
                       #fname='statistics.json')

    with open('statistics_9.json', 'w') as f:
        json.dump(statistics, f)

    print("Done.")