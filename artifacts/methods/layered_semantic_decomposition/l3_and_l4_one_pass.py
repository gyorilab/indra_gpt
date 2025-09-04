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



biomolecular_relationships_schema = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "description": (
                "A list of biomolecular relationships extracted from the sentence. "
                "Each relationship must describe a biomechanistic interaction involving a predicate, "
                "one or more subject entities, and optionally one or more object entities. "
                "The predicate may include conditions or mechanisms (e.g., 'A activates B through C', or 'A inhibits B if D is present'). "
                "Relationships can be n-ary, involving multiple subjects and/or objects. "
                "All subjects and objects must be physical biomolecular entities—such as proteins, genes, RNAs, small molecules, or ions. "
                "Exclude any relationship involving non-physical or abstract entities."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "predicate": {
                        "type": "string",
                        "description": "The verb or relation expressing the interaction or effect."
                    },
                    "subjects": {
                        "type": "array",
                        "description": "List of physical biomolecular entities that act as the subject(s) in the relationship.",
                        "items": {"type": "string"},
                        "minItems": 1
                    },
                    "objects": {
                        "type": "array",
                        "description": "List of physical biomolecular entities that act as the object(s) in the relationship.",
                        "items": {"type": "string"},
                        "minItems": 0
                    }
                },
                "required": ["predicate", "subjects", "objects"]
            }
        }
    },
    "required": ["relationships"]
}

biomolecular_relationship_predicates_schema = {
    "type": "object",
    "description": "All biomolecular relationship predicates extracted from the sentence.",
    "properties": {
        "predicates": {
            "type": "array",
            "description": (
                "A list of biomolecular relationship predicates, each represented as a string. "
                "These predicates describe the interactions or relationships between biomolecules."
                "All nominalized forms of the predicates should be included but represented in their verb form."
            ),
            "items": {"type": "string"},
            "default": []
        }
    },
    "required": ["predicates"]
}

biomolecular_relationships_simple_schema = {
    "type": "object",
    "description": "Biomolecular relationships extracted from the sentence.",
    "properties": {
        "relationships": {
            "type": "array",
            "description": (
                "A list of biomolecular relationships, each represented as a dictionary. "
            ),
            "items": {"type": "object"},
            "default": []
        }
    },
    "required": ["relationships"]
}

is_valid_schema = {
    "type": "object",
    "properties": {
        "valid": {
            "type": "boolean",
            "description": "Indicates if the relationship type constraints are satisfied."
        },
        "reason": {
            "type": "string",
            "description": "A reason for the validity or invalidity of the type constraints."
        }
    },
    "required": ["valid"]
}

def extract_biomolecular_relationship_predicates(text: str, schema: dict = biomolecular_relationship_predicates_schema) -> dict:
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
        
    prompt = f"""
    Extract all biomolecular relationship predicates from the given input sentence.

    Return your output as a valid JSON object that strictly follows this schema:

    RELATIONSHIP PREDICATE SCHEMA:
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
        extracted_predicates = json.loads(raw_output)["predicates"]
        return extracted_predicates

    except Exception as e:
        logger.warning(f"Failed to parse JSON: {raw_output if 'raw_output' in locals() else '<no output>'} — error: {e}")
        return []


def extract_biomolecular_relationships(input_sentence: str, 
                                       predicates: str,
                                       schema: dict = biomolecular_relationships_simple_schema) -> dict:
    if not isinstance(input_sentence, str):
        raise ValueError("Input text must be a string.")
        
    prompt = f"""
    Extract all biomolecular relationships from the given input sentence, for each
    predicate provided in the list of predicates.

    The nature of the relationships can be n-ary, meaning they can involve multiple subjects and/or objects.
    You can structure the output in the best format that captures the relationships accurately, the format does
    not necessarily have to be triplet based.

    Return your output as a valid JSON object that strictly follows this schema:

    RELATIONSHIP SCHEMA:
    {json.dumps(schema, indent=2)}

    INPUT SENTENCE:
    {input_sentence}

    LIST OF PREDICATES:
    {predicates}

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
        extracted_relationships = json.loads(raw_output)["relationships"]
        return extracted_relationships

    except Exception as e:
        logger.warning(f"Failed to parse JSON: {raw_output if 'raw_output' in locals() else '<no output>'} — error: {e}")
        return []

def relationship_type_checker(rel: str, schema: dict=is_valid_schema):

    if not isinstance(rel, str):
        raise ValueError("Input relationship must be a string.")
    
    prompt = f"""
    You are a relationship type validator. Your job is to determine whether the provided relationship satisfies strict semantic type constraints.

    Evaluate the relationship and return a JSON object with:
    - "valid": true or false
    - "reason": a brief explanation of why the relationship is valid or invalid

    ### TYPE CONSTRAINTS:

    1. **Predicate**
       - Predicate represents a specific biomolecular process or interaction.
       - General or vague verbs like "affects" or "enhances" are not specific enough.

    2. **Subjects**
       - Each subject must be a physical biomolecular entity.
       - Examples: proteins, genes, RNAs, small molecules, ions.
       - Do **NOT** include biological processes or modifications as subjects.

    3. **Objects**
       - Each object must also be a physical biomolecular entity.
       - Do **NOT** include abstract processes, conditions, or cellular behaviors.

    If any part of the relationship violates these constraints, mark "valid" as false and explain why.

    Return your output as a valid JSON object that strictly follows this schema:

    RELATIONSHIP VALIDATION SCHEMA:
    {json.dumps(schema, indent=2)}

    INPUT RELATIONSHIP:
    {rel}

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
