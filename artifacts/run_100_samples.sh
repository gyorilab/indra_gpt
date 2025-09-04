model="deepseek-chat"

python -m indra_gpt.scripts.run_statement_extraction \
  --output_folder "./output/statement_extractions/${model}/" \
  --num_samples 100 \
  --random_sample \
  --random_seed 42 \
  --model "${model}" \
  --structured_output \
  --n_shot_prompting 2 \
  --self_correction_iterations 1 \
  --grounding \
  --grounding_strategy "gilda"
