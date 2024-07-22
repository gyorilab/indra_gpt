# INDRA GPT

## Overview

This repository contains code for interacting with Chat GPT's chat API. The code is 
divided into several files:

- `api.py`: Contains the function `run_openai_chat` which sends a prompt to the Chat GPT
  chat API and returns the response. It takes optional arguments to customize some of 
  the settings of the API call.
- `check_correctnes.py`: Contains functions to check for statement correctness, 
  including checking for type of error in incorrect statements.
- `cli.py` A CLI for `check_correctness.py`.
- `constants.py` Contains constants used in `run_statement_json_extraction.py`.
- `reach_extraction.py` Contains functions for extracting English statements given 
  evidence text from a correct statement.
- `run_statement_json_extraction.py` Contains functions for extracting sparse statement 
  json objects given evidence text from a correct statement.
- `utils.py` Contains utility functions used in `run_statement_json_extraction.py`.

## Installation
Clone this repository and install the requirements with:
```shell
pip install -r requirements.txt
```

## Running the statement extraction pipeline

To run the statement extraction pipeline:
```shell
python -m indra_gpt.run_statement_json_extraction
```

View the results:
```shell
less statement_json_extraction_results.tsv
```

`run_statement_json_extraction` takes a couple of optional arguments:

- `--stmts-file` Path to a json file containing statement json objects to check. They
  are assumed to be correct, i.e. explicitly curated as correct. This option defaults to
  `indra_gpt/indra_benchmark_corpus_all_correct.json`
- `--openai-version` A string corresponding to one of the OpenAI model names. See
  https://platform.openai.com/docs/models for available models. Default is
  `'gpt-4o-mini'`.
- `--iterations | -n` Number of statements to guess. Minimum is 5. Default is 50.
- `--output-file` Path to save the output tsv file. Defaults to
  `indra_gpt/statement_json_extraction_results.tsv`.
