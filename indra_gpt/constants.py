import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()

INPUT_DEFAULT = HERE / "indra_benchmark_corpus_all_correct.json"
OUTPUT_DEFAULT = HERE / "statement_json_extraction_results.tsv"
SCHEMA_PATH = HERE / "indra_schema.json"

with open(SCHEMA_PATH) as file:
    JSON_SCHEMA = json.load(file)
