#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate indra_gpt

python test.py \
    --input indra_gpt/resources/indra_benchmark_corpus_all_correct.json \
    --client ollama \
    --model llama3.2:latest \
    --max_examples 100 \
    --logfile benchmark.log
