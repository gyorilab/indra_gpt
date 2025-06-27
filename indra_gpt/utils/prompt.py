import json

def get_schema_wrapped_prompt(prompt, schema):
    # If schema is json, convert it to a string
    if isinstance(schema, dict):
        schema = json.dumps(schema, indent=2)

    wrapped_prompt = f"""
Respond to the following prompt using the provided schema.
The output response must be a **valid raw JSON object**, and must not include:
- Markdown code blocks (no triple backticks)
- Extra commentary, explanation, or formatting

SCHEMA:
{schema}

PROMPT:
{prompt}

RESPONSE:
"""
    return wrapped_prompt
