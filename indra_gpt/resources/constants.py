import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()

INDRA_BENCHMARK_CORPUS_ALL_CORRECT = HERE / "indra_benchmark_corpus_all_correct.json"
ROOT_DIR = HERE.parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DEFAULT = OUTPUT_DIR / "statement_json_extraction_results.pkl"
INDRA_SCHEMA = HERE / "indra_schema.json"

USER_INIT_PROMPT_REDUCED = HERE / "prompts" / "user_init_prompt_reduced.txt"
GENERIC_REFINEMENT_PROMPT = HERE / "prompts" / "refinement_prompts" / "generic_refinement.txt"
STATEMENT_TYPE_REFINEMENT_PROMPT = HERE / "prompts" / "refinement_prompts" / "statement_type_refinement.txt"
ERROR_CONTEXT_PROMPT = HERE / "prompts" / "refinement_prompts" / "error_context.txt"
