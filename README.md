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

## Details of Statement Extraction

The statement extraction pipeline uses OpenAI's Chat-GPT chat API to generate 
statements by the 'show-and-tell' method. The pipeline iterates over a set of correct 
statements. For each iteration, two pieces of evidence text from other statements in the 
same set are picked to use as examples. The example text is paired with trimmed down 
versions of their corresponding correct statement json objects and put into the chat 
history. The evidence text of the statement for the current iteration is appended as the 
next question in the chat history. The full chat with history is sent to the chat API to 
generate a response. From the response, a statement json object is extracted.

An illustrative example of the messages sent (the actual prompt is larger than this and 
includes a simplified JSON schema of the statement object):
```json
[
  {
    "role": "user",
    "content": "Read the following JSON schema for a statement object: [...].\n\nExtract the relation from the following sentence and put it in a JSON object matching the schema above.\n\nSentence: Furthermore, acetylation of p53 K120 by the MOF and MSL1v1 complex greatly enhanced the transcription activity of p53 (XREF_FIG).'"
  },
  {
    "role": "assistant",
    "content": "{\"type\": \"Acetylation\", \"enz\": {\"name\": \"KANSL1\", \"db_refs\": {\"UP\": \"Q7Z3B3\", \"HGNC\": \"24565\", \"TEXT\": \"MSL1v1\"}}, \"sub\": {\"name\": \"TP53\", \"db_refs\": {\"UP\": \"P04637\", \"HGNC\": \"11998\", \"TEXT\": \"p53\"}}, \"residue\": \"K\", \"position\": \"120\", \"belief\": 0.9927351664162256, \"evidence\": [{\"text\": \"Furthermore, acetylation of p53 K120 by the MOF and MSL1v1 complex greatly enhanced the transcription activity of p53 (XREF_FIG).\"}]}"
  },
  {
    "role": "user",
    "content": "Extract the relation from the following sentence and put it in a JSON object matching the schema above. The JSON object needs to be able to pass a validation against the provided schema.[...],\n\nSentence: Indeed, we show that upon treatment with chemotherapeutic drugs c-Abl enhances the phosphorylation-dependent interaction between Pin1 and p73, and this in turn promotes p73 acetylation by p300."
  },
  {
    "role": "assistant", "content": "{\"type\": \"Acetylation\", \"enz\": {\"name\": \"EP300\", \"db_refs\": {\"UP\": \"Q09472\", \"HGNC\": \"3373\", \"TEXT\": \"p300\"}}, \"sub\": {\"name\": \"TP73\", \"db_refs\": {\"UP\": \"O15350\", \"HGNC\": \"12003\", \"TEXT\": \"p73\"}}, \"belief\": 0.9999999998071971, \"evidence\": [{\"text\": \"Indeed, we show that upon treatment with chemotherapeutic drugs c-Abl enhances the phosphorylation-dependent interaction between Pin1 and p73, and this in turn promotes p73 acetylation by p300.\"}]}"
  },
  {
    "role": "user",
    "content": "Extract the relation from the following sentence and put it in a JSON object matching the schema above. The JSON object needs to be able to pass a validation against the provided schema.[...]\n\nSentence: C5a promotes the proliferation of human nasopharyngeal carcinoma cells through PCAF-mediated STAT3 acetylation."
  }
]
```

## Statement Extraction Results

The results of the statement extraction pipeline are saved in a tsv file. The notebook 
`notebooks/Check statement json extraction.ipynb` contains code to analyze check the 
correctness of the extracted statements and also attempts to salvage statements with 
agents that were not properly regonized by the Chat GPT.
