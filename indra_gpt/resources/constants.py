import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()

INPUT_DEFAULT = HERE / "indra_benchmark_corpus_all_correct.json"
ROOT_DIR = HERE.parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DEFAULT = OUTPUT_DIR / "statement_json_extraction_results.pkl"
SCHEMA_PATH = HERE / "indra_schema.json"
SCHEMA_STRUCTURED_OUTPUT_PATH = HERE / "indra_schema_openai_structured_output.json"

GENERIC_REFINEMENT_PROMPT = HERE / "prompts" / "refinement_prompts" / "generic_refinement.txt"
STATEMENT_TYPE_REFINEMENT_PROMPT = HERE / "prompts" / "refinement_prompts" / "statement_type_refinement.txt"

with open(SCHEMA_PATH) as file:
    JSON_SCHEMA = json.load(file)
