from indra_gpt.preprocessors.rec_relation_parser.surface_parse import (
    prompt_parse_surface_relationship,
    prompt_parse_surface_argument,
    prompt_rewrite_coordinated_entity_phrase_as_relational,
    prompt_unnominalize
)

import json
import logging
logger = logging.getLogger(__name__)


class RecRelationParser:
    def __init__(self, client, **kwargs):
        self.client = client
        self.max_depth = kwargs.get("max_depth", 5)

    def preprocess(self, text):
        parsed_tree = self.expression_parse_recursive(
            text,
            depth=0,
            max_depth=self.max_depth
        )
        valid_sentences = self.extract_terminal_relationships(
            parsed_tree
        )
        # concatenate items from valid_sentences into a single string
        simplified_text = "\n".join([json.dumps(r) for r in valid_sentences])
        return simplified_text
    
    def extract_terminal_relationships(self, tree: dict) -> list[dict]:
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
    
    def expression_parse_recursive(
        self,
        text,
        depth=0,
        max_depth=5
    ):
        return self.parse_relationship_recursive(
            text,
            depth=depth,
            max_depth=max_depth
        )
    
    def _jsonify(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Invalid JSON structure: {text}")
            return {}
    
    def parse_relationship_recursive(
        self,
        input: str,
        depth=0,
        max_depth=5
    ):
        if not isinstance(input, str):
            raise ValueError("Input text must be a string.")

        parsed = self._jsonify(self.client.call(prompt_parse_surface_relationship(input)))

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

        parsed_subject = self.parse_arg_recursive(
            str(raw_subject),
            depth=depth + 1,
            max_depth=max_depth
        )
        parsed_object = self.parse_arg_recursive(
            str(raw_object),
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
        self,
        input: str,
        depth=0,
        max_depth=5
    ):
        if input == "":
            return {"NON_RELATIONAL_PHRASE": ""}
        
        if not isinstance(input, str):
            raise ValueError("Input text must be a string.")

        rewritten = self.client.call(prompt_rewrite_coordinated_entity_phrase_as_relational(input)).strip()
        unnominalized = self.client.call(prompt_unnominalize(rewritten)).strip()

        parsed = self._jsonify(self.client.call(prompt_parse_surface_argument(unnominalized)))

        if not (len(parsed) == 1 and set(parsed.keys()).issubset({"NON_RELATIONAL_PHRASE", "RELATIONAL_PHRASE"})):
            raise ValueError(f"Invalid argument structure: {parsed}")

        if "RELATIONAL_PHRASE" in parsed:
            return self.parse_relationship_recursive(
                str(parsed["RELATIONAL_PHRASE"]),
                depth=depth + 1,
                max_depth=max_depth
            )
        elif "NON_RELATIONAL_PHRASE" in parsed:
            return {"NON_RELATIONAL_PHRASE": str(parsed["NON_RELATIONAL_PHRASE"])}
        else:
            raise ValueError(f"Unknown phrase type: {parsed}")
