import json

def get_schema_wrapped_prompt(prompt, schema):
    # If schema is json, convert it to a string
    if isinstance(schema, dict):
        schema = json.dumps(schema, indent=2)

    wrapped_prompt =f"""
Respond to the following prompt using the provided schema.
The output response should be a valid JSON object that adheres to the schema.
ONLY respond with the JSON object, do not add any formatting indicator or additional text.

SCHEMA:
{schema}

PROMPT:
{prompt}

RESPONSE:
"""
    return wrapped_prompt
