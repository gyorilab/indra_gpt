from indra_gpt.preprocessors.rec_logic_parser.expr_schema import (
    EPISTEMIC_MODE_SCHEMA,
    LOGICAL_COORDINATION_SCHEMA,
    LOGICAL_UNARY_SCHEMA
)
import json

def prompt_parse_surface_epistemic_mode(text: str, schema: dict=EPISTEMIC_MODE_SCHEMA):
    prompt = f"""
    You are an epistemic mode normalizer.

    Your task is to classify the **epistemic status** of the following text as either:

    1. **HYPOTHESIS** — if the sentence expresses possibility, uncertainty, speculation, or inquiry  
    (e.g., includes forms like "we examined whether", "it was unclear if", "could", "might", "was tested", etc.)

    2. **DECLARATIVE** — if the sentence expresses a factual, confident, or assertive claim without hedging or doubt.

    Then, using the provided JSON schema, return the sentence **as-is** for DECLARATIVE cases, or convert HYPOTHESIS 
    cases into a minimal, complete declarative **without changing the original meaning**.

    ⚠️ STRICT RULES:
    - For **DECLARATIVE**, always return the **original sentence unchanged**.
    - For **HYPOTHESIS**, rewrite only to the extent necessary to produce a complete proposition.
    - Do **not** drop clauses, modifiers, or descriptive phrases.
    - The goal is structural normalization, **not simplification**.

    🧠 Examples:
    - "We tested whether X is possible" → `"HYPOTHESIS": "X is possible"`
    - "It is unknown if A causes B" → `"HYPOTHESIS": "A causes B"`
    - "A causes B" → `"DECLARATIVE": "A causes B"`
    - "The results show that X increases Y in T cells" → `"DECLARATIVE": "The results show that X increases Y in T cells"`

    Return your output as a valid JSON object that strictly conforms to the schema.

    SCHEMA:
    {json.dumps(schema, indent=2)}

    TEXT:
    {text}

    OUTPUT:
    """
    return prompt
  
    
def prompt_parse_surface_logical_grouping(text: str, schema: dict=LOGICAL_COORDINATION_SCHEMA):
    prompt = f"""
    You are a surface-level logical parser.

    Your task is to classify the **top-level coordination structure** of the input sentence based 
    on its **propositional coordination**.

    🔑 Definitions:
    - **Coordination**: A grammatical structure joining phrases or clauses 
        (e.g., with "and", "or", commas, slashes, etc.).
    - **Propositional coordination**: Coordination of full statements/events/relations, 
        e.g., AND(R1, R2) where R1 and R2 are separate relationships or clauses.
    - **Constituent coordination**: Coordination between subjects/objects within a single 
        clause, e.g., Rel(A and B) where both A and B are arguments of the same predicate.

    You must ignore **constituent coordination** and instead focus on identifying whether 
    there is **propositional coordination** at the surface level.

    🧠 Label the sentence using one of the following grouping types:
    - `"AND"` — if there is top-level **conjunction** between multiple relationships.
    - `"OR"` — if there is top-level **disjunction** between multiple relationships.
    - `"ATOM"` — if the sentence contains only a single event/relationship or no top-level coordination.

    ⚠️ IMPORTANT INSTRUCTIONS:
    - Do NOT paraphrase or summarize the sentence.
    - Do NOT omit or distort **referential context** (e.g., from relative clauses).
    - If splitting at relative pronouns like "which" or "that", restore the referent explicitly.
    - Preserve any **event or action noun** if a dependent clause modifies it (e.g., "activation", "inhibition").

    🔁 Restoration Examples (relative clause context):
    - "X activation of Y, which requires Z" →  
    `["X activation of Y", "X activation of Y requires Z"]`

    💡 Classification Examples:
    - "Alice writes reports and emails" →  
    `{{ "AND": ["Alice writes reports", "Alice writes emails"] }}`
    - "Tom, Sarah, and Max were invited" →  
    `{{ "AND": ["Tom was invited", "Sarah was invited", "Max was invited"] }}`
    - "The robot can lift or push" →  
    `{{ "OR": ["The robot can lift", "The robot can push"] }}`
    - "Emma completed the project" →  
    `{{ "ATOM": "Emma completed the project" }}`
    - "A causes B, which inhibits C" →  
    `{{ "AND": ["A causes B", "B inhibits C"] }}`
    - "Team of David and Sarah won three games and took home the trophy" →
    `{{ "AND": ["Team of David and Sarah won three games", "Team of David and Sarah took home the trophy"] }}`

    Return your output as a valid JSON object that strictly conforms to the schema.

    SCHEMA:
    {json.dumps(schema, indent=2)}

    TEXT:
    {text}

    OUTPUT:
    """
    return prompt
    

def prompt_parse_surface_logical_unary(text: str, schema: dict=LOGICAL_UNARY_SCHEMA):
    prompt = f"""
    You are a logical polarity classifier.

    Your task is to determine the **logical polarity** of the input sentence.

    Return one of the following:
    - `"NOT"`: If the **entire sentence** expresses a single, unified **negated** 
    proposition that can be clearly rewritten in affirmative form.
    - `"IDENTITY"`: If the sentence is **not negated**, or contains **multiple clauses**, 
    coordination, or complex structure such that the overall logical polarity cannot be confidently determined.

    🧠 Definitions:
    - **Logical polarity** refers to whether the top-level meaning of a sentence 
    affirms or negates a complete proposition.
    - Only classify as `"NOT"` if the **whole sentence** is clearly a negation 
    and its **positive counterpart** can be written with minimal changes.
    - If the sentence contains **multiple embedded clauses**, **relative clauses**, 
    or **ambiguous scope**, return `"IDENTITY"` without modifying the sentence.

    🔁 If `"NOT"` is chosen, convert the negated sentence into its **minimally edited, unnegated form**.
    📌 If `"IDENTITY"` is chosen, return the **original sentence exactly as-is**.

    ✅ Examples:
    - "A does not affect B" → {{ "NOT": "A affects B" }}
    - "X fails to activate Y" → {{ "NOT": "X activates Y" }}
    - "Protein A inhibits B" → {{ "IDENTITY": "Protein A inhibits B" }}
    - "The study, which included 10 participants, did not show significant results." → {{ "IDENTITY": "The study, which included 10 participants, did not show significant results." }}
    - "It is not true that A causes B or C activates D" → {{ "NOT": "It is true that A causes B or C activates D" }}
    - "A inhibits B, but does not activate C" → {{ "IDENTITY": "A inhibits B, but does not activate C" }}

    ⚠️ When in doubt or when structure is too complex to determine polarity, choose `"IDENTITY"`.

    Return a **single top-level JSON object** that strictly conforms to the schema.

    SCHEMA:
    {json.dumps(schema, indent=2)}

    TEXT:
    {text}

    OUTPUT:
    """
    return prompt

def prompt_resolve_referential_expressions(text: str) -> str:
    prompt = f"""
    You are a coreference and bridging reference resolver.

    Your task is to resolve all referential expressions in the input text. This includes:
    - Pronouns (e.g., "it", "they", "this")
    - Bridging references (e.g., "both agents", "the latter", "these findings") that 
    refer to concepts stated or implied anywhere in the input

    For each such reference, replace it with its full, explicit referent — the actual 
    phrase or concept being referenced — based on your understanding of the full context of the sentence or passage.

    🔒 Instructions:
    - Use the full input as your context; antecedents may occur **before or after** the reference.
    - Preserve the sentence structure and wording as much as possible.
    - Do NOT paraphrase, summarize, or reorder content.
    - If the reference is ambiguous or cannot be confidently resolved, leave it unchanged.

    If no referential expressions are present, return the input text unchanged.

    TEXT:
    {text}

    OUTPUT:
    """
    return prompt
    
def prompt_distribute_shared_modifier(text: str) -> str:
    prompt = f"""
    You are a linguistic rewriter.

    Your task is to rewrite the input sentence so that any **shared modifier phrase** 
    (such as prepositional or adverbial phrases like "in X", "under Y", "during Z") is 
    **explicitly applied to each clause joined by propositional coordination**.

    This makes each statement **logically self-contained** and **fully contextualized** 
    for downstream processing.

    🔑 Definitions:
    - **Modifier**: A phrase indicating time, location, condition, etc. (e.g., "in cells", "during stress").
    - **Propositional coordination**: Joining of separate clauses or full events (e.g., "A activates X and B inhibits Y").
    - **Constituent coordination**: Joining of parts within the same clause (e.g., "A and B activate X").

    ⚠️ Instructions:
    1. Identify if the sentence contains a **shared leading modifier**.
    2. If the sentence includes **propositional coordination**, distribute the modifier into each coordinated clause.
    3. Do **not** distribute modifiers across **constituent coordination**.
    4. Your output should be a **single sentence** preserving the original meaning but making all context **explicit**.
    5. If no shared modifier is present or no propositional coordination is found, return the original sentence unchanged.

    TEXT:
    {text}

    OUTPUT:
    """
    return prompt
