import json
import logging
from typing import Any, Dict, List, Optional
from indra.statements import stmts_from_json

logger = logging.getLogger(__name__)

class JSONToINDRAConverter:
    def __init__(self, config: Optional[dict] = None):
        """
        Converter to postprocess LLM outputs and convert them into INDRA statements.
        Args:
            config (dict, optional): Configuration parameters to include in evidence annotations.
        """
        self.config = config or {}

    def format_raw_extraction(self, raw_response: str) -> List[Dict]:
        try:
            parsed = json.loads(raw_response)

            # Case 1: Proper list of dicts
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                return parsed

            # Case 2: Mistaken "type: array" structure
            if isinstance(parsed, dict):
                if parsed.get("type") == "array" and isinstance(parsed.get("items"), list):
                    return parsed["items"]

                # Case 3: Single dict
                return [parsed]

        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse raw LLM output: {e}")

        # Fallback: return empty list
        return []

    def _is_empty(self, value: Any) -> bool:
        empty_strings = {"", "none", "null", "unknown", "na"}
        if value is None:
            return True
        if isinstance(value, str) and value.lower().strip() in empty_strings:
            return True
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
        return False

    def remove_empty_values(self, data: Dict) -> Dict:
        if isinstance(data, dict):
            cleaned = {k: self.remove_empty_values(v) for k, v in data.items()}
            return {k: v for k, v in cleaned.items() if not self._is_empty(v)}
        elif isinstance(data, list):
            cleaned = [self.remove_empty_values(i) for i in data]
            return [i for i in cleaned if not self._is_empty(i)]
        elif isinstance(data, str):
            return None if data.lower().strip() in {"", "none", "null", "unknown", "na"} else data
        return data

    def update_evidence(
        self,
        stmt_dict: Dict,
        text: str,
        pmid: str
    ) -> Dict:
        """
        Inject sentence and PMID evidence into a single statement dict.
        """
        evidence = {
            "text": text,
            "pmid": pmid,
            "source_api": "indra_gpt",
            "annotations": {
                "indra_gpt_config": self.config
            }
        }

        if "evidence" in stmt_dict and isinstance(stmt_dict["evidence"], list):
            for evi in stmt_dict["evidence"]:
                evi.update(evidence)
                try:
                    if "epistemics" in evi:
                        for key in ["hypothesis", "negation", "direct"]:
                            val = evi["epistemics"].get(key)
                            if isinstance(val, str):
                                evi["epistemics"][key] = val.lower() == "true"
                except Exception as e:
                    logger.warning(f"Problem updating epistemics in evidence: {e}")
        else:
            stmt_dict["evidence"] = [evidence]

        return stmt_dict
    
    def raw_to_indra_statements(
            self,
            raw_response: str,
            sentence: str,
            pmid: str,
        ) -> List:

        raw_json_stmts = self.format_raw_extraction(raw_response)

        processed_json_stmts = []
        for i, raw_json_stmt in enumerate(raw_json_stmts):
            json_stmt = self.remove_empty_values(raw_json_stmt)
            json_stmt = self.update_evidence(json_stmt, sentence, pmid)

            processed_json_stmts.append(json_stmt)

        indra_statements = stmts_from_json(processed_json_stmts)
        return indra_statements
