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

def normalize_span_to_enum(span: str, field_name: str, choices: List[str]) -> str:
    enum_options = ', '.join([f'"{choice}"' for choice in choices])
    return f"""
You are a classification assistant.

Given the extracted span: "{span}" from the field `{field_name}`, choose the most appropriate label from the list of valid options.

Respond strictly as a JSON object that conforms to the following schema:

{{
  "type": "object",
  "properties": {{
    "value": {{
      "type": "string",
      "enum": [{enum_options}, ""]
    }}
  }},
  "required": ["value"],
  "additionalProperties": false
}}

Instructions:
- If the span clearly matches one of the valid labels, return that label as the `"value"`.
- If the span does not clearly match any label, return an empty string (`""`) as the value.
- Do not include any explanation, extra text, or formatting.

Only return a valid JSON object that conforms to the schema above.
"""

def generate_span_extraction_prompt(model_cls: Type[BaseModel], text: str) -> str:
    """
    Generate a prompt for extracting text spans for any INDRA statement Pydantic class.

    - List fields → "array" of strings, with description: "Extract relevant spans..."
    - Non-list fields → "string", with description: "Extract relevant span..."
    """
    schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False
    }

    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        base_desc = (field.description or "").strip()

        if is_list_type(annotation):
            description = base_desc
            schema["properties"][name] = {
                "type": "array",
                "items": {"type": "string"},
                "description": description
            }
        else:
            description = base_desc
            schema["properties"][name] = {
                "type": "string",
                "description": description
            }

    # Add contextual wrapper using model config description
    model_desc = model_cls.model_config.get("description", "").strip()

    context_block = f"""
Context:
The following span of text has already been identified as relevant to an instance of **{model_cls.__name__}**:
\"{text.strip()}\"

Where the description for {model_cls.__name__} is: {model_desc}
"""

    prompt = f"""
You are a biomedical information extractor.

{context_block}

Your task is to extract **exact span(s) of text** from the input that correspond to the fields of the following INDRA statement type: **{model_cls.__name__}**.

Return a single JSON object that conforms to the schema below.
Each field must be a **contiguous substring directly copied** from the input — no rewording, inference, or paraphrasing.

SCHEMA:
{json.dumps(schema, indent=2)}

Instructions:
- For each field, use the description to guide which span to extract.
- If the field is a string, return a single exact substring. If no such span is found, return an empty string: "".
- If the field is a list, return an array of exact substrings. If no spans are found, return an empty array: [].
- Only include spans that are **explicitly present** in the input text.

TEXT:
{text}

Return only the JSON object described by the schema above.
Do not include any explanations, extra fields, comments, or formatting (like markdown or HTML tags).

OUTPUT:
"""
    return prompt
