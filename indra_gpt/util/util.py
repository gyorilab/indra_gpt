import copy

#################################

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
