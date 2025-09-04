relationship_schema = {
  "type": "object",
  "description": "A relationship is a statement which is constituted by: a relation and its arguments, where arguments can be terminal or recursively relational.",
  "properties": {
    "RELATIONSHIP": {
      "type": "object",
      "properties": {
        "RELATION": {
          "type": "string",
          "description": "The core verb or predicate expressing the relation."
        },
        "SUBJECT": {
          "type": "string",
          "description": "The subject of the relationship."
        },
        "OBJECT": {
          "type": "string",
          "description": "The object of the relationship, if present in the sentence."
        }
      },
      "required": ["RELATION", "SUBJECT"],
    }
  },
  "required": ["RELATIONSHIP"]
}

argument_schema = {
    "type": "object",
    "description": "An argument is a grammatical phrase that either contains a relational verb or not.",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "NON_RELATIONAL_PHRASE": {
                    "type": "string",
                    "description": "A phrase that does not encode any relationship.",                
                }
            },
            "required": ["NON_RELATIONAL_PHRASE"]
        },
        {
            "type": "object",
            "properties": {
                "RELATIONAL_PHRASE": {
                    "type": "string",
                    "description": "A phrase that contains one or more encoded relationship."
                }
            },
            "required": ["RELATIONAL_PHRASE"]
        }
    ]
}


# --- Recursive Composition Pipeline ---
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

logger = logging.getLogger(__name__)


def parse_surface_relationship(text: str, schema: dict = relationship_schema) -> dict:
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
        
    prompt = f"""
    You are a surface-level semantic parser.

    Your task is to decompose the following sentence into a **relationship** consisting of:
    - A core **RELATION**: The main verb or predicate that expresses the relationship.
    - A **SUBJECT**: The full subject phrase of the relationship.
    - An **OBJECT**: The full object phrase of the relationship, if present in the sentence.

    🧠 Rules:
    - Focus on surface grammar rather than domain-specific meaning.
    - If an argument contains coordination (e.g., "A and B"), treat it as a **single argument** unless each part is associated with a separate verb or predicate.
    - Do **not** split coordinated noun phrases like "X and Y group" or "apples and oranges basket" — treat the entire phrase as one argument if it refers to a unified entity.
    - If the relationship is **binary**, fill in both the SUBJECT and OBJECT fields.
    - If the relationship is **n-ary** (i.e., involves more than two core participants), **conjoin all participants into a single string in the SUBJECT field**, and leave the OBJECT field empty.

    Return your output as a valid JSON object that follows the schema below.

    RELATIONSHIP SCHEMA:
    {json.dumps(schema, indent=2)}

    INPUT SENTENCE:
    {text}

    OUTPUT:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=3000,
        )
        raw_output = response.choices[0].message.content.strip()
        return json.loads(raw_output)

    except Exception as e:
        logger.warning(f"Failed to parse JSON: {raw_output if 'raw_output' in locals() else '<no output>'} — error: {e}")
        return {}

def parse_surface_argument(text: str, schema: dict = argument_schema) -> dict:
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    
    prompt = f"""
    Classify the following phrase into one of two categories:

    1. **RELATIONAL_PHRASE** — if the phrase contains:
      - a verb or verb phrase (e.g., "moves to the next stage", "initiates a process"),
      - or a term that is a nominal form of a verb. 

    2. **NON_RELATIONAL_PHRASE** — Else if the phrase is not a **RELATIONAL_PHRASE**.

    ❗Important:
    - A phrase must contain a verb or a nominal form that originate from a verb to be classified as RELATIONAL.

    Return a valid JSON object following the schema below.

    SCHEMA:
    {json.dumps(schema, indent=2)}

    TEXT:
    {text}

    OUTPUT:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1024,
        )
        raw_output = response.choices[0].message.content.strip()
        return json.loads(raw_output)

    except Exception as e:
        logger.warning(f"Failed to parse JSON: {raw_output if 'raw_output' in locals() else '<no output>'} — error: {e}")
        return {}
    

def rewrite_coordinated_entity_phrase_as_relational(phrase: str) -> str:
    prompt = f"""
    You are a linguistic assistant.

    Your task is to analyze a **noun phrase** and determine whether it can be rewritten into an **explicit relational statement**.

    Look for two key cues:
    1. **Coordinating structures**, such as:
    - "and", "or", "with", commas, slashes ("/"), hyphens ("-"), semicolons
    2. **Nominalized verbs** that imply a relation, such as:
    - "interaction", "complex", "assembly", "mixture", "partnership", "group", etc.

    📌 Attempt rephrasing **only if all of the following conditions are satisfied**:
    - The phrase includes one or more coordinating structures.
    - It contains a nominalized verb or construct that implies a verb-like action.
    - The phrase is **semantically complete** — it clearly identifies the entities involved and the implied action, and can be naturally interpreted as a standalone relational statement.

    ✅ If these criteria are met:
    - Convert the nominal verb into a verbal form.
    - Rephrase the full phrase into a natural relational sentence.

    🧠 Examples:
    - "MOF and MSL1v1 complex" → "MOF forms a complex with MSL1v1"
    - "salt and water mixture" → "Salt is mixed with water"
    - "Alice and Bob collaboration" → "Alice collaborates with Bob"

    🚫 Do **not** rephrase if:
    - The phrase is ambiguous or lacks enough context to infer the relationship.
    - It is a descriptive grouping or listing without a clear relational implication.

    Examples (no rephrasing):
    - "trees, shrubs, and vines"
    - "glucose metabolism"

    📌 If the phrase cannot be naturally rewritten into a complete and contextually sound relational statement, return it **unchanged**.
    ONLY OUTPUT THE REWRITTEN SENTENCE — do not explain or comment.
    
    INPUT:
    {phrase}

    OUTPUT:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM rewrite failed: {e}")
        return phrase
    
def unnominalize(phrase: str) -> str:
    prompt = f"""
    You are a language assistant.

    Your task is to analyze a sentence and **rewrite it so that all nominalized verbs are converted into explicit verb-based expressions**.

    🎯 Your goal is to replace these noun forms with their original **verb forms**, restructuring the sentence accordingly.

    ✅ Guidelines:
    - Use active voice whenever possible.
    - Ensure the rewritten sentence is **natural, grammatically correct, and preserves the original meaning**.
    - Rewrite **all nominalized verbs** in the sentence.
    - Keep all other sentence information intact.

    📌 If the phrase contains no nominalized verbs, return it **unchanged**.
    ONLY OUTPUT THE REWRITTEN SENTENCE — do not explain or comment.

    INPUT:
    {phrase}

    OUTPUT:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM rewrite failed: {e}")
        return phrase
    
# --- Recursive Composition Pipeline ---
def parse_relationship_recursive(
    input: str,
    parse_surface_relationship=parse_surface_relationship,
    parse_surface_argument=parse_surface_argument,
    relationship_schema=relationship_schema,
    argument_schema=argument_schema,
    depth=0,
    max_depth=5
):
    if not isinstance(input, str):
        raise ValueError("Input text must be a string.")
        
    parsed = parse_surface_relationship(str(input), relationship_schema)

    if not (len(parsed) == 1 and set(parsed.keys()).issubset({"RELATIONSHIP"})):
        raise ValueError(f"Invalid relationship structure: {parsed}")

    if set(parsed["RELATIONSHIP"].keys()) != set(["RELATION", "SUBJECT", "OBJECT"]):
        raise ValueError(f"Invalid relationship structure: {parsed}")

    raw_relation = parsed["RELATIONSHIP"]["RELATION"]
    raw_subject = parsed["RELATIONSHIP"]["SUBJECT"]
    raw_object = parsed["RELATIONSHIP"].get("OBJECT", None)

    if depth > max_depth:
        logger.warning(f"Max depth {max_depth} exceeded at relationship level. Returning fallback.")
        return {
            "RELATIONSHIP": {
                "RELATION": str(raw_relation),
                "SUBJECT": str(raw_subject),
                "OBJECT": str(raw_object)
            }
        }

    parsed_subject = parse_arg_recursive(
        str(raw_subject),
        parse_surface_relationship=parse_surface_relationship,
        parse_surface_argument=parse_surface_argument,
        relationship_schema=relationship_schema,
        argument_schema=argument_schema,
        depth=depth + 1,
        max_depth=max_depth
    )
    parsed_object = parse_arg_recursive(
        str(raw_object),
        parse_surface_relationship=parse_surface_relationship,
        parse_surface_argument=parse_surface_argument,
        relationship_schema=relationship_schema,
        argument_schema=argument_schema,
        depth=depth + 1,
        max_depth=max_depth
    ) if raw_object is not None else None

    return {
        "RELATIONSHIP": {
            "RELATION": str(raw_relation),
            "SUBJECT": parsed_subject,
            "OBJECT": parsed_object,
        }
    }

def parse_arg_recursive(
    input: str,
    parse_surface_relationship=parse_surface_relationship,
    parse_surface_argument=parse_surface_argument,
    relationship_schema=relationship_schema,
    argument_schema=argument_schema,
    depth=0,
    max_depth=5
):
    if input == "":
        return {"NON_RELATIONAL_PHRASE": ""}
    
    if not isinstance(input, str):
        raise ValueError("Input text must be a string.")

    rewritten_input = rewrite_coordinated_entity_phrase_as_relational(str(input)) 
    rewritten_input = unnominalize(str(rewritten_input))

    parsed = parse_surface_argument(rewritten_input, argument_schema)

    if not (len(parsed) == 1 and set(parsed.keys()).issubset({"NON_RELATIONAL_PHRASE", "RELATIONAL_PHRASE"})):
        raise ValueError(f"Invalid argument structure: {parsed}")

    if "RELATIONAL_PHRASE" in parsed:
        return parse_relationship_recursive(
            str(parsed["RELATIONAL_PHRASE"]),
            parse_surface_relationship=parse_surface_relationship,
            parse_surface_argument=parse_surface_argument,
            relationship_schema=relationship_schema,
            argument_schema=argument_schema,
            depth=depth + 1,
            max_depth=max_depth
        )
    elif "NON_RELATIONAL_PHRASE" in parsed:
        return {"NON_RELATIONAL_PHRASE": str(parsed["NON_RELATIONAL_PHRASE"])}
    else:
        raise ValueError(f"Unknown phrase type: {parsed}")
