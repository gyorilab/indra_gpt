import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()

INPUT_DEFAULT = HERE / "indra_benchmark_corpus_all_correct.json"
ROOT_DIR = HERE.parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DEFAULT = OUTPUT_DIR / "statement_json_extraction_results.tsv"
SCHEMA_PATH = HERE / "indra_schema.json"
SCHEMA_STRUCTURED_OUTPUT_PATH = HERE / "indra_schema_openai_structured_output.json"

with open(SCHEMA_PATH) as file:
    JSON_SCHEMA = json.load(file)
