import json
import copy
import random
import logging
import csv
from typing import Any, Union, List, Dict

from indra_gpt.configs import ProcessorConfig
from indra_gpt.resources.constants import INPUT_DEFAULT

logger = logging.getLogger(__name__)

def sample_from_input_file(
    config: ProcessorConfig, random_seed: int = 42
) -> List[Dict[str, str]]: 
    """Samples input data from the given input file."""
    random.seed(random_seed)
    user_inputs = load_input_file(config.base_config.user_inputs_file)

    do_random_sample = config.base_config.random_sample
    num_samples = config.base_config.num_samples

    if num_samples > len(user_inputs):
        logger.warning(
            f"Requested {num_samples} samples, but only {len(user_inputs)} "
            f"available. Using all available samples."
        )
        num_samples = len(user_inputs)

    return (
        random.sample(user_inputs, num_samples) 
        if do_random_sample 
        else user_inputs[:num_samples]
    )

def load_input_file(input_file_path: str) -> List[Dict[str, str]]:
    """Loads input data from a TSV or JSON file and returns a list of dictionaries."""
    if input_file_path is None:
        with open(INPUT_DEFAULT, encoding="utf-8") as f:
            benchmark_corpus = json.load(f)
            return [
                {
                    "text": get_input_text_from_original_statement_json(stmt),
                    "pmid": get_pmid_from_original_statement_json(stmt),
                }
                for stmt in benchmark_corpus
            ]

    elif input_file_path.endswith(".tsv"):
        with open(input_file_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            formatted_input_file = []
            first_line = True

            for parts in reader:
                if len(parts) != 2:
                    raise ValueError(f"Invalid TSV format in line: {parts}")

                text, pmid = parts
                text = text.strip('"')  # Remove quotes
                pmid = pmid.strip()  # Remove extra spaces

                if first_line and text.lower() == "text" and pmid.lower() == "pmid":
                    first_line = False
                    continue

                formatted_input_file.append({"text": text, "pmid": pmid})

        return formatted_input_file

    elif input_file_path.endswith(".json"):
        with open(input_file_path, encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Invalid input file format: {input_file_path}")

def get_input_text_from_original_statement_json(
    original_statement_json_object: Union[str, Dict[str, Any]]
) -> str:
    """Extracts the 'text' field from the given JSON object."""
    if isinstance(original_statement_json_object, str):
        try:
            original_statement_json_object = json.loads(
                original_statement_json_object
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON string provided: {original_statement_json_object}"
            ) from e
    return original_statement_json_object["evidence"][0]["text"]

def get_pmid_from_original_statement_json(
    original_statement_json_object: Union[str, Dict[str, Any]]
) -> str:
    """Extracts the 'pmid' field from the given JSON object."""
    if isinstance(original_statement_json_object, str):
        try: 
            original_statement_json_object = json.loads(
                original_statement_json_object
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON string provided: {original_statement_json_object}"
            ) from e
    return original_statement_json_object["evidence"][0]["pmid"]

def resolve_ref(
    schema: Dict[str, Any], ref_path: str
) -> Dict[str, Any]:
    """Helper function to resolve a $ref path in a JSON schema."""
    keys = ref_path.lstrip("#/").split("/")
    ref_obj = schema
    for key in keys:
        ref_obj = ref_obj.get(key, {})
    return copy.deepcopy(ref_obj)

def merge_allOf(
    schema: Dict[str, Any], root_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merges allOf definitions into their parent objects."""
    if isinstance(schema, dict):
        if "allOf" in schema:
            merged_schema = {}
            required_fields = set()

            for sub_schema in schema["allOf"]:
                if "$ref" in sub_schema:
                    ref_obj = resolve_ref(root_schema, sub_schema["$ref"])
                    sub_schema = ref_obj.copy()

                for key, value in sub_schema.items():
                    if key == "required":
                        required_fields.update(value)
                    elif (
                        key in merged_schema
                        and isinstance(merged_schema[key], dict)
                        and isinstance(value, dict)
                    ):
                        merged_schema[key].update(value)
                    else:
                        merged_schema[key] = value

            merged_schema.pop("allOf", None)
            if required_fields:
                merged_schema["required"] = list(required_fields)

            schema.clear()
            schema.update(merged_schema)

        for key in ["properties", "items", "definitions"]:
            if key in schema and isinstance(schema[key], dict):
                schema[key] = {
                    k: merge_allOf(v, root_schema) for k, v in schema[key].items()
                }

        return schema

    elif isinstance(schema, list):
        return [merge_allOf(item, root_schema) for item in schema]

    return schema
