from indra_gpt.preprocessors.rec_logic_parser.surface_parse import (
    prompt_parse_surface_epistemic_mode,
    prompt_parse_surface_logical_grouping,
    prompt_parse_surface_logical_unary,
    prompt_resolve_referential_expressions,
    prompt_distribute_shared_modifier
)
import json
import logging
logger = logging.getLogger(__name__)


class RecLogicParser:
    def __init__(self, client, **kwargs):
        self.client = client
        self.max_depth = kwargs.get("max_depth", 5)

    def convert_to_ir(self, text):
        return self.expression_parse_recursive(
            text,
            max_depth=self.max_depth
        )

    def preprocess(self, text):
        parsed_tree = self.expression_parse_recursive(
            text,
            depth=0,
            max_depth=self.max_depth
        )
        valid_sentences = self.valid_from_exp_tree(
            parsed_tree
        )
        # concatenate items from valid_sentences into a single string
        simplified_text = "\n".join(valid_sentences)
        return simplified_text
    
    def valid_from_exp_tree(self, tree, not_count=0, has_hypothesis=False):
        identities = []

        if not isinstance(tree, dict):
            return identities

        for key, value in tree.items():
            if key == "HYPOTHESIS":
                # Skip hypothesis (mark it)
                identities += self.valid_from_exp_tree(value, not_count, True)

            elif key == "NOT":
                # Increment negation depth
                identities += self.valid_from_exp_tree(value, not_count + 1, has_hypothesis)

            elif key in ("AND", "OR"):
                # Recurse on each item in the list
                for item in value:
                    identities += self.valid_from_exp_tree(item, not_count, has_hypothesis)

            elif key == "DECLARATIVE":
                identities += self.valid_from_exp_tree(value, not_count, has_hypothesis)

            elif key == "IDENTITY":
                if isinstance(value, dict):
                    # Can be nested like {"ATOM": "..."} or {"AND": [...]}
                    for subkey, subval in value.items():
                        if subkey == "ATOM":
                            if not has_hypothesis and not_count % 2 == 0:
                                identities.append(subval)
                        elif subkey in ("AND", "OR"):
                            for item in subval:
                                identities += self.valid_from_exp_tree(item, not_count, has_hypothesis)
                elif isinstance(value, str):
                    # Sometimes identity may just map to a string
                    if not has_hypothesis and not_count % 2 == 0:
                        identities.append(value)

        return identities

    def expression_parse_recursive(
        self,
        text,
        depth=0,
        max_depth=5
    ):
        return self.parse_epistemic_recursive(
            text,
            depth=depth,
            max_depth=max_depth
        )

    def parse_epistemic_recursive(
        self,
        text,
        depth=0,
        max_depth=5
    ):
        if depth > max_depth:
            logger.warning(f"Max depth {max_depth} exceeded at epistemic level. Returning fallback.")
            return {
                "DECLARATIVE": {
                    "IDENTITY": {
                        "ATOM": text
                    }
                }
            }
        parsed = self._jsonify(self.client.call(prompt_parse_surface_epistemic_mode(text)))

        mode_key = next((k for k in ["DECLARATIVE", "HYPOTHESIS"] if k in parsed), None)
        if not mode_key:
            raise ValueError(f"Unsupported epistemic mode: {parsed}")

        return {
            mode_key: self.parse_unary_recursive(
                str(parsed[mode_key]),
                depth=depth + 1,
                max_depth=max_depth
            )
        }

    def parse_unary_recursive(
        self,
        text,
        depth=0,
        max_depth=5
    ):
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        
        if depth > max_depth:
            logger.warning(f"Max depth {max_depth} exceeded at unary level. Returning fallback.")
            return {
                "IDENTITY": {
                    "ATOM": text
                }
            }

        parsed = self._jsonify(self.client.call(prompt_parse_surface_logical_unary(text)))

        if "NOT" in parsed:
            return {
                "NOT": self.parse_grouping_recursive(
                    str(parsed["NOT"]),
                    depth=depth + 1,
                    max_depth=max_depth
                )
            }
        elif "IDENTITY" in parsed:
            return {
                "IDENTITY": self.parse_grouping_recursive(
                    str(parsed["IDENTITY"]),
                    depth=depth + 1,
                    max_depth=max_depth
                )
            }
        else:
            raise ValueError(f"Unsupported logical unary type: {parsed}")

    def parse_grouping_recursive(
        self,
        text,
        depth=0,
        max_depth=5
    ):
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        
        if depth > max_depth:
            logger.warning(f"Max depth {max_depth} exceeded at grouping level. Returning fallback.")
            return {
                "ATOM":str(text)
            }
        
        self.client.extra_params = {'response_format': None}
        ref_resolved_text = self.client.call(prompt_resolve_referential_expressions(text))
        self.client.extra_params = {'response_format': {'type': 'json_object'}}

        # text = distribute_shared_modifier(text)
        parsed = self._jsonify(self.client.call(prompt_parse_surface_logical_grouping(str(ref_resolved_text))))

        if "AND" in parsed:
            return {
                "AND": [
                    self.parse_epistemic_recursive(
                        str(item),
                        depth=depth + 1,
                        max_depth=max_depth
                    )
                    for item in parsed["AND"]
                ]
            }
        elif "OR" in parsed:
            return {
                "OR": [
                    self.parse_epistemic_recursive(
                        str(item),
                        depth=depth + 1,
                        max_depth=max_depth
                    )
                    for item in parsed["OR"]
                ]
            }
        elif "ATOM" in parsed:
            return {
                "ATOM": str(parsed["ATOM"])
            }
        else:
            raise ValueError(f"Unsupported logical grouping: {parsed}")

    def _jsonify(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {}
