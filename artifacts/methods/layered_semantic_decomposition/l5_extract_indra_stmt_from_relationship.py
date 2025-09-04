import os
from openai import OpenAI
import json
import logging


logger = logging.getLogger(__name__)


api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

import json
import logging

from typing import Union
import json
import logging

import sys
from pathlib import Path
sys.path.append(Path.cwd().parent.as_posix())
from indra_gpt.resources.constants import SCHEMA_PATH
indra_schema = json.load(open(SCHEMA_PATH, 'r'))

import jsonref
import copy



logger = logging.getLogger(__name__)


indra_statement_types_schema = {
  "type": "object",
  "description": "Classification of an INDRA statement into one of the supported statement types.",
  "anyOf": [
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "RegulateActivity",
          "description": "Regulation of the activity (e.g., activation or inhibition) of one protein by another."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "Modification",
          "description": "Modification (e.g., phosphorylation) of one protein by another."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "SelfModification",
          "description": "A protein modifying itself, such as autophosphorylation."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "ActiveForm",
          "description": "A protein is active or inactive under certain conditions."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "Gef",
          "description": "A GEF activates a GTPase protein by promoting GDP-GTP exchange."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "Gap",
          "description": "A GAP accelerates GTP hydrolysis by a GTPase protein."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "Complex",
          "description": "Two or more proteins physically form a complex."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "Association",
          "description": "A looser conceptual association between entities or proteins."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "Translocation",
          "description": "Movement of a protein or agent from one cellular location to another."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "RegulateAmount",
          "description": "Increase or decrease in the amount of a protein due to another protein."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "object",
      "properties": {
        "StatementType": {
          "type": "string",
          "const": "Conversion",
          "description": "A biochemical conversion where one or more molecules are transformed into others, mediated by a protein."
        }
      },
      "required": ["StatementType"]
    },
    {
      "type": "null",
      "description": "Indicates that no INDRA statement type is relevant to the input."
    }
  ]
}

def identify_indra_statement_type_from_relationship(relationship: str, schema: dict=indra_statement_types_schema):
    if not isinstance(relationship, str):
        raise ValueError("Input relation must be a string.")

    prompt = f"""
    You are an assistant that maps a **single relational verb or verb phrase** (e.g., 'phosphorylates', 'forms complex with') 
    to the most appropriate INDRA statement type.

    Your task is to examine the input relation and classify it using the provided INDRA statement type schema.

    ❗Only choose a type if there is a clear conceptual match to a known biochemical mechanism.
    ❗If no INDRA type matches well, return: {{ "StatementType": null }}.

    Examples:
    - "activates" → "RegulateActivity"
    - "phosphorylates" → "Modification"
    - "autophosphorylates" → "SelfModification"
    - "forms complex with" → "Complex"
    - "translocates" → "Translocation"

    Return your output as a valid JSON object that strictly conforms to the schema.

    SCHEMA:
    {json.dumps(schema, indent=2)}

    RELATION:
    {relationship}

    OUTPUT:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Failed to identify INDRA statement type: {e}")
        return {}

def collect_referenced_defs(schema_fragment, full_definitions, collected=None):
    if collected is None:
        collected = {}

    if isinstance(schema_fragment, dict):
        if "$ref" in schema_fragment:
            ref_path = schema_fragment["$ref"]
            if ref_path.startswith("#/definitions/"):
                ref_key = ref_path.split("/")[-1]
                if ref_key not in collected:
                    ref_def = full_definitions[ref_key]
                    collected[ref_key] = ref_def
                    collect_referenced_defs(ref_def, full_definitions, collected)
        else:
            for value in schema_fragment.values():
                collect_referenced_defs(value, full_definitions, collected)

    elif isinstance(schema_fragment, list):
        for item in schema_fragment:
            collect_referenced_defs(item, full_definitions, collected)

    return collected


def get_subschema_with_refs(statement_type: str, indra_schema: dict=indra_schema) -> dict:
    full_definitions = indra_schema.get("definitions", {})
    statement_schema = copy.deepcopy(full_definitions.get(statement_type))

    if statement_schema is None:
        raise ValueError(f"Statement type '{statement_type}' not found in INDRA schema.")

    # Collect all referenced defs from this statement
    needed_defs = collect_referenced_defs(statement_schema, full_definitions)

    # Construct the trimmed schema
    return {
        "$schema": indra_schema.get("$schema", "http://json-schema.org/draft-07/schema#"),
        "title": statement_type,
        "type": "object",
        "allOf": [
            { "$ref": f"#/definitions/{statement_type}" }
        ],
        "definitions": {
            statement_type: statement_schema,
            **needed_defs
        }
    }

def extract_indra_statement_from_relationship(relationship: str, indra_schema: dict=indra_schema) -> dict:
    if not isinstance(relationship, str):
        raise ValueError("Input relationship must be a string.")
    
    # Get the appropriate INDRA schema
    statement_type = identify_indra_statement_type_from_relationship(str(relationship), indra_schema).get("StatementType", None)
    if statement_type is None:
        logger.warning(f"Could not identify INDRA statement type for relationship: {relationship}")
        return {}
    if not isinstance(statement_type, str):
        logger.warning(f"Expected string for statement type, got: {type(statement_type)}")
        return {}
    
    statement_schema = get_subschema_with_refs(statement_type, indra_schema)

    # Prompt
    prompt = f"""
    You are an information extraction assistant.

    Your task is to fill in the INDRA statement corresponding to the following relationship:
    "{str(relationship)}"

    🧠 INSTRUCTIONS:
    - Use the JSON schema provided below.
    - Ensure all required fields are filled.
    - Use best judgment for converting subject/object to entities.
    - If some values like residue or position are missing from the sentence, omit or use null/None as needed.
    - Use this only for {statement_type} type.

    SCHEMA:
    {json.dumps(statement_schema, indent=2)}

    OUTPUT:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=5000,
        )
        raw_output = response.choices[0].message.content.strip()
        return json.loads(raw_output)

    except Exception as e:
        logger.warning(f"Failed to extract INDRA statement from relationship: {e}")
        return {}

