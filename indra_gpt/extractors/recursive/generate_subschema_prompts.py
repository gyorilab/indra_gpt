from typing import get_args, get_origin, List, Type, Union
from pydantic import BaseModel, Field
import json


def unwrap_optional(annotation):
    """Unwrap Optional[T] (i.e., Union[T, NoneType]) to get T"""
    if get_origin(annotation) is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return args[0] if args else annotation
    return annotation


def is_list_type(annotation) -> bool:
    """Return True if the annotation is or wraps a List[...]"""
    actual = unwrap_optional(annotation)
    return get_origin(actual) in (list, List)

def prompt_boolean_normalization(field_name: str, value: str) -> str:
    return f"""
You are a boolean classifier.

Given the value for the field `{field_name}`: "{value}", decide whether it implies True or False.

Respond strictly as a JSON object that conforms to the following schema:

{{
  "type": "object",
  "properties": {{
    "value": {{
      "type": "boolean"
    }}
  }},
  "required": ["value"],
  "additionalProperties": false
}}

Do not include any explanation, markdown, or additional text.
Only return the valid JSON object.
"""

def generate_span_extraction_prompt(model_cls: Type[BaseModel], text: str) -> str:
    """
    Generate a prompt for extracting text spans for any INDRA statement Pydantic class.

    - List fields → "array" of strings, with description: "Extract relevant spans..."
    - Non-list fields → "string", with description: "Extract relevant span..."
    """
    schema = {
        "type": "object",
        "properties": {}
    }

    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        base_desc = (field.description or "").strip()

        try:
            pattern = field.metadata[0].pattern
        except Exception as e:
            pattern = None

        if pattern:
            choices = "|".join(pattern.split("|")).replace("(", "").replace(")", "").replace("^", "").replace("$", "")
            base_desc += (
                f"\nIf the input text directly mentions one of the following types, select exactly one of: {choices.strip()}."
                " If the text does not directly mention any of them, return an empty string ('')."
            )
        if is_list_type(annotation):
            description = f"{base_desc}\nExtract relevant spans of text if directly mentioned in the input text, else return an empty array."
            schema["properties"][name] = {
                "type": "array",
                "items": {"type": "string"},
                "description": description
            }
        else:
            description = f"{base_desc}\nExtract relevant span of text if directly mentioned in the input text, else return empty string."
            schema["properties"][name] = {
                "type": "string",
                "description": description
            }

    # Remove "title" to prevent it being echoed in LLM output
    # Add extra constraints for better LLM behavior
    prompt = f"""
You are a biomedical information extractor.

Your task is to extract **relevant span(s) of text** from the input that correspond to the fields of the following INDRA statement type: **{model_cls.__name__}**

Return a single JSON object that conforms to the schema below.
Each field should contain a direct phrase or clause from the input.

SCHEMA:
{json.dumps(schema, indent=2)}

TEXT:
{text}

Again your output must be a single JSON object that conforms to the schema above.
Do not include any additional text, comment, or explanation in your output.
Do not include any field that is not present in the input text.
Do not include any formatting syntax like markdown or HTML tags in your output.

OUTPUT:
"""
    return prompt
