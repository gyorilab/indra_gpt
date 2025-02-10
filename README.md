# INDRA GPT

## Overview

This repository contains code for interacting with Chat GPT's chat API. The project has
some key scripts that can be run on the command line. 

- `indra_gpt/scripts/run_statement_json_extraction.py`: Extracts sparse statement 
  json objects given evidence text.

- `indra_gpt/scripts/cli.py`: A CLI to check for English statement correctness, including checking for type of error in
  incorrect English statements. 

- `indra_gpt/scripts/reach_extraction.py`: Extracts implied English statements given 
  evidence text.

## Installation
Clone this repository and install the requirements with:
```shell
pip install -r requirements.txt
```

## Running the statement extraction pipeline

To run the statement extraction pipeline:
```shell
python -m indra_gpt.scripts.run_statement_json_extraction
```

View the results:
```shell
less ./output/statement_json_extraction_results.tsv
```

`run_statement_json_extraction` takes a couple of optional arguments:

- `--stmts-file` Path to a json file containing statement json objects to check. They
  are assumed to be correct, i.e. explicitly curated as correct. This option defaults to
  `indra_gpt/resources/indra_benchmark_corpus_all_correct.json`
- `--openai-version` A string corresponding to one of the OpenAI model names. See
  https://platform.openai.com/docs/models for available models. Default is
  `'gpt-4o-mini'`.
- `--iterations | -n` Number of statements to guess. Minimum is 5. Default is 50.
- `--output-file` Path to save the output tsv file. Defaults to
  `indra_gpt/statement_json_extraction_results.tsv`.
- `--batch_jobs` Use to run script will run as a batch job.
- `--batch_id` Batch job id to see the current status of the job, if the job is completed, output will be downloaded. 

### Details of Statement Extraction

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

### Statement Extraction Results

The results of the statement extraction pipeline are saved in a tsv file. The notebook 
`notebooks/Check statement json extraction.ipynb` contains code to analyze check the 
correctness of the extracted statements and also attempts to salvage statements with 
agents that were not properly regonized by the Chat GPT.

## Evaluating Statement Correctness

### Creating training data
To run the evaluation we first need training data. Training data is constructed by joining a curation file and a statements file
on their hash keys.

Here is an example of creating training data:
```shell
python -m indra_gpt.scripts.cli create-training-data --curations-file "./indra_gpt/resources/sample_curation.json" --statements-file "./indra_gpt/resources/sample_statements.json" 
```

### Evaluating training data
A training data can be evaluated by running this script:
```shell
python -m indra_gpt.scripts.cli run-stats
```
View the evaluation statistics in command line (example):
```
Confusion matrix:
               correct  incorrect
gpt_correct         44         12
gpt_incorrect       22         22
Precision: 0.7857142857142857
Recall: 0.6666666666666666
Accuracy: 0.66
Total examples: 100
```

The evalation result file is saved here:
```
./local_data/results/correct_vs_incorrect_<creation date when you run above script>.json
```
