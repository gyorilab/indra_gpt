#!/bin/bash

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate indra_gpt

python test.py \
    --input indra_gpt/resources/indra_benchmark_corpus_all_correct.json \
    --client openai \
    --model gpt-4o-mini \
    --max_examples 100 \
    --logfile benchmark.log
