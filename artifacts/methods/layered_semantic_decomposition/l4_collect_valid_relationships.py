def extract_terminal_relationships(tree: dict) -> list[dict]:
    collected = []

    def _traverse(node):
        if not isinstance(node, dict):
            return

        if "RELATIONSHIP" in node:
            rel_obj = node["RELATIONSHIP"]
            rel = rel_obj.get("RELATION")
            subj = rel_obj.get("SUBJECT")
            obj = rel_obj.get("OBJECT")

            # Check if both subject and object are terminal (not nested relationships)
            subj_phrase = subj.get("NON_RELATIONAL_PHRASE") if isinstance(subj, dict) else None
            obj_phrase = obj.get("NON_RELATIONAL_PHRASE") if isinstance(obj, dict) else None

            if isinstance(rel, str) and subj_phrase and obj_phrase:
                collected.append({
                    "relation": rel,
                    "subject": subj_phrase,
                    "object": obj_phrase
                })

            # Recursively traverse both subject and object
            _traverse(subj)
            _traverse(obj)

    _traverse(tree)
    return collected
