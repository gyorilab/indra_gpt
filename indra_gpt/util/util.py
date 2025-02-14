from indra.statements import get_all_descendants, Statement
import copy


def trim_stmt_json(stmt):
    """Function to get rid of irrelevant parts of the indra statement json

    Parameter
    ---------
    stmt : dict
        The indra statement json object

    Returns
    -------
    dict
        The indra statement json object with irrelevant parts removed
    """

    stmt["evidence"] = [{"text": stmt["evidence"][0].get("text")}]

    del stmt["id"]
    del stmt["matches_hash"]

    if 'supports' in stmt:
        del stmt['supports']
    if 'supported_by' in stmt:
        del stmt['supported_by']

    return stmt

# Recursively go through each key-value, and if it is an empty string or list or dict, remove the key-value pair
def remove_empty_strings_and_lists(d):
    for key, value in list(d.items()):
        if isinstance(value, dict):
            remove_empty_strings_and_lists(value)
        elif isinstance(value, list):
            for i in value:
                if isinstance(i, dict):
                    remove_empty_strings_and_lists(i)
        if value in [None, "", [], {}]:
            del d[key]
    return d

def post_process_extracted_json(gpt_stmt_json):
    """Function to post process the extracted json from chatGPT

    Parameters
    ----------
    gpt_stmt_json : dict
        The extracted json from chatGPT

    Returns
    -------
    dict
        The post processed json or if there is a KeyError, the original json
    """
    try:
        stmt_type = gpt_stmt_json["type"]
        mapped_type = stmt_mapping.get(stmt_type.lower(), stmt_type)
        gpt_stmt_json["type"] = mapped_type
    except KeyError:
        pass

    gpt_stmt_json = remove_empty_strings_and_lists(gpt_stmt_json)

    return gpt_stmt_json


def _get_statement_mapping():
    stmt_classes = get_all_descendants(Statement)
    mapping = {stmt_class.__name__.lower(): stmt_class.__name__ for stmt_class in stmt_classes}
    return mapping

stmt_mapping = _get_statement_mapping()


def resolve_ref(schema, ref_path):
    """Helper function to resolve a $ref path in a JSON schema."""
    keys = ref_path.lstrip("#/").split("/")
    ref_obj = schema
    for key in keys:
        ref_obj = ref_obj.get(key, {})
    return copy.deepcopy(ref_obj)  # Return a deep copy to prevent modifying the original schema

def merge_allOf(schema, root_schema):
    """Recursively merges allOf definitions into their parent objects and removes allOf.
       - Resolves $ref only if inside allOf
       - Resolves nested allOf in properties, items, and definitions
       - Does NOT resolve nested $ref inside resolved object
    """
    if isinstance(schema, dict):
        if "allOf" in schema:
            merged_schema = {}
            required_fields = set()

            for sub_schema in schema["allOf"]:
                if "$ref" in sub_schema:
                    ref_obj = resolve_ref(root_schema, sub_schema["$ref"])
                    sub_schema = ref_obj.copy()  # Use a copy to prevent modifying the root schema

                # Merge properties correctly
                for key, value in sub_schema.items():
                    if key == "required":
                        required_fields.update(value)
                    elif key in merged_schema and isinstance(merged_schema[key], dict) and isinstance(value, dict):
                        merged_schema[key].update(value)  # Merge nested dictionaries (e.g., properties)
                    else:
                        merged_schema[key] = value  # Overwrite other keys

            merged_schema.pop("allOf", None)  # Remove allOf after merging
            if required_fields:
                merged_schema["required"] = list(required_fields)  # Assign merged required fields
            
            schema.clear()
            schema.update(merged_schema)

        # Recursively process properties, items, and definitions **after merging**
        for key in ["properties", "items", "definitions"]:
            if key in schema and isinstance(schema[key], dict):
                schema[key] = {k: merge_allOf(v, root_schema) for k, v in schema[key].items()}

        return schema

    elif isinstance(schema, list):
        return [merge_allOf(item, root_schema) for item in schema]

    return schema  # Return primitive values unchanged
