from indra_gpt.preprocessors.rec_relation_parser.rel_schema import (
    RELATIONSHIP_SCHEMA,
    ARGUMENT_SCHEMA
)
import json


def prompt_parse_surface_relationship(text: str, schema: dict = RELATIONSHIP_SCHEMA) -> dict:
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
    Respond only with a valid JSON object. Do not include any Markdown formatting, explanations, or natural language before or after.

    RELATIONSHIP SCHEMA:
    {json.dumps(schema, indent=2)}

    INPUT SENTENCE:
    {text}

    OUTPUT:
    """
    return prompt

def prompt_parse_surface_argument(text: str, schema: dict = ARGUMENT_SCHEMA) -> dict:
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
    Respond only with a valid JSON object. Do not include any Markdown formatting, explanations, or natural language before or after.

    SCHEMA:
    {json.dumps(schema, indent=2)}

    TEXT:
    {text}

    OUTPUT:
    """

    return prompt
    

def prompt_rewrite_coordinated_entity_phrase_as_relational(phrase: str) -> str:
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

    return prompt
    
def prompt_unnominalize(phrase: str) -> str:
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
    return prompt
