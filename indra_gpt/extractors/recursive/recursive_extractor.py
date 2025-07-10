import json
from typing import Type, Union, get_origin, get_args, List, Optional
from pydantic import BaseModel, ValidationError
from indra_gpt.extractors.recursive.generate_subschema_prompts import (generate_span_extraction_prompt, 
                                                                       prompt_boolean_normalization,
                                                                       normalize_span_to_enum)

from indra_gpt.extractors.recursive.indra_stmts_model import IndraStatements

def unwrap_optional(annotation):
    if get_origin(annotation) is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return args[0] if args else annotation
    return annotation

def is_list_of_pydantic(annotation):
    actual = unwrap_optional(annotation)
    return get_origin(actual) in (list, List) and issubclass(get_args(actual)[0], BaseModel)

def is_pydantic_model(annotation):
    actual = unwrap_optional(annotation)
    return isinstance(actual, type) and issubclass(actual, BaseModel)

def get_field_pattern_choices(field) -> Optional[List[str]]:
    try:
        pattern = field.metadata[0].pattern
        raw = pattern.replace("^", "").replace("$", "").replace("(", "").replace(")", "")
        choices = [s for s in raw.split("|") if s]
        return choices
    except Exception:
        return None

def recursive_extract(
    text: str,
    llm_client,
    model_cls: Type[BaseModel] = IndraStatements,
    path: str = ""
) -> Optional[BaseModel]:
    
    if not text.strip():
        return None

    prompt = generate_span_extraction_prompt(model_cls, text)
    raw_output = llm_client.call(prompt)

    try:
        span_dict = json.loads(raw_output)
    except Exception as e:
        print(f"[JSONError] at {path or model_cls.__name__}:\n{e}")
        return None

    final_fields = {}
    for field_name, field in model_cls.model_fields.items():
        annotation = field.annotation
        span_values = span_dict.get(field_name)
        if span_values is None:
            continue

        full_path = f"{path}.{field_name}" if path else field_name

        # Single nested model (expects one string span)
        if is_pydantic_model(annotation):
            if isinstance(span_values, str):
                sub_instance = recursive_extract(span_values, llm_client, unwrap_optional(annotation), full_path)
                if sub_instance:
                    final_fields[field_name] = sub_instance

        # List of nested models (expects list of string spans)
        elif is_list_of_pydantic(annotation):
            item_cls = get_args(unwrap_optional(annotation))[0]
            sub_items = []
            for i, span_text in enumerate(span_values):
                if not isinstance(span_text, str):
                    continue
                rec_item = recursive_extract(span_text, llm_client, item_cls, f"{full_path}[{i}]")
                if rec_item:
                    sub_items.append(rec_item)
            final_fields[field_name] = sub_items

        elif unwrap_optional(annotation) is bool:
            if isinstance(span_values, bool):
                final_fields[field_name] = span_values
            elif isinstance(span_values, str):
                if span_values.strip() == "":
                    final_fields[field_name] = None
                else:
                    bool_prompt = prompt_boolean_normalization(field_name, span_values)
                    bool_output_raw = llm_client.call(bool_prompt).strip()
                    try:
                        bool_output = json.loads(bool_output_raw)
                        if isinstance(bool_output, dict) and "value" in bool_output:
                            final_fields[field_name] = bool_output["value"]
                        else:
                            print(f"[BoolParsingError] No 'value' key found in boolean response for {full_path}: {bool_output}")
                    except json.JSONDecodeError:
                        print(f"[BoolParsingError] Failed to parse boolean response for {full_path}:\n{bool_output_raw}")
            else:
                print(f"[UnexpectedType] Bool field {full_path} has type {type(span_values)}")

        else:
            actual = unwrap_optional(annotation)

            # Empty string → None
            if actual is str and isinstance(span_values, str) and span_values.strip() == "":
                final_fields[field_name] = None

            # Pattern normalization for Optional[str]
            elif actual is str and isinstance(span_values, str):
                pattern_choices = get_field_pattern_choices(field)
                if pattern_choices:
                    enum_prompt = normalize_span_to_enum(span_values, field_name, pattern_choices)
                    enum_output_raw = llm_client.call(enum_prompt).strip()
                    try:
                        enum_output = json.loads(enum_output_raw)
                        value = enum_output.get("value", "")
                        final_fields[field_name] = value if value != "" else None
                    except json.JSONDecodeError:
                        print(f"[EnumParsingError] Failed to parse enum value for {full_path}:\n{enum_output_raw}")
                else:
                    final_fields[field_name] = span_values

            else:
                final_fields[field_name] = span_values

    try:
        instance = model_cls(**final_fields)
        return instance
    except ValidationError as e:
        print(f"[Final ValidationError] while constructing {model_cls.__name__} at {path}:\n{e}")
        return None

def get_indra_stmts_json_list(model_instance):
    stmts_json = []
    for val in model_instance.model_dump().values():
        if isinstance(val, list):
            stmts_json.extend(val)
    return stmts_json   
