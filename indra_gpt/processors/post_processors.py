import logging
import json
import io
from typing import List, Dict, Any, Union

from indra.statements import get_all_descendants, Statement
from indra.tools import assemble_corpus as ac
from indra.statements.io import stmts_from_json
from indra.preassembler.grounding_mapper.gilda import ground_statements
from indra_gpt.util.util import sample_from_input_file
from indra_gpt.configs import PostProcessorConfig

logger = logging.getLogger(__name__)


class PostProcessor:
    def __init__(self, config: PostProcessorConfig) -> None:
        self.config = config

    def process(self, generated_responses: List[Dict[str, Any]]
                ) -> List[Statement]:
        logger.info("Starting post-processing of extracted statements...")

        input_samples = sample_from_input_file(self.config,
                                               self.config.base_config.random_seed)
        input_texts = [entry["text"] for entry in input_samples]
        input_pmids = [entry["pmid"] for entry in input_samples]

        if len(generated_responses) != len(input_texts):
            raise ValueError(f"Mismatch in generated responses and input texts: "
                             f"{len(generated_responses)} responses vs {len(input_texts)} input texts.")

        if input_pmids and len(input_pmids) != len(input_texts):
            logger.warning(f"PMID count ({len(input_pmids)}) does not match "
                                f"input text count ({len(input_texts)}). Proceeding "
                                "without PMIDs where missing.")

        processed_stmts_json_list = []
        for i, response in enumerate(generated_responses):
            text = input_texts[i]
            pmid = input_pmids[i] if input_pmids else "Not provided"
            try:
                stmt_json_response = self.parse_json_response(response)
                stmts_json = self.extract_statements(stmt_json_response)
                stmts_json = [self.map_stmt_class_name(stmt) for stmt in stmts_json]
                stmts_json = [self.remove_empty_strings_and_lists(stmt) for stmt in stmts_json]
                stmts_json = [self.update_evidence(stmt, text, pmid, self.config) for stmt in stmts_json]
                processed_stmts_json_list.append(stmts_json)
            except Exception as e:
                logger.error(f"Error processing response: {response} | Error: {e}")
                processed_stmts_json_list.append({"error": str(e)})

        preassembled_statements = [self.preassemble_pipeline(stmt) for stmt in processed_stmts_json_list]
        logger.info(f"Completed post-processing: {len(preassembled_statements)} statements processed.")
        return preassembled_statements

    def map_stmt_class_name(self, stmt_json: Dict[str, Any]) -> Dict[str, Any]:
        stmt_classes = get_all_descendants(Statement)
        stmt_mapping = {stmt_class.__name__.lower(): stmt_class.__name__ for stmt_class in stmt_classes}
        stmt_json["type"] = stmt_mapping.get(stmt_json.get("type", "").lower(), stmt_json.get("type"))
        return stmt_json

    def remove_empty_strings_and_lists(self, d: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in list(d.items()):
            if isinstance(value, dict):
                self.remove_empty_strings_and_lists(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self.remove_empty_strings_and_lists(item)
            if value in [None, "", [], {}]:
                del d[key]
        return d

    def update_evidence(self, stmt_json: Dict[str, Any], original_input_text: str,
                        pmid: str, config: PostProcessorConfig) -> Dict[str, Any]:
        actual_evidence = {
            "text": original_input_text,
            "pmid": pmid,
            "source_api": "indra_gpt",
            "annotations": {"indra_gpt_config": config.base_config.__dict__}
        }

        evidence_list = stmt_json.get("evidence", [])
        if evidence_list:
            for evidence in evidence_list:
                evidence.update(actual_evidence)
        else:
            stmt_json["evidence"] = [actual_evidence]

        return stmt_json

    def preassemble_pipeline(self, stmts_json: List[Dict[str, Any]]) -> Any:
        stmts = self.capture_logs_from_stmts_json(stmts_json)
        if not isinstance(stmts, list):
            return stmts
        try:
            stmts = self.ground_statements(stmts, apply_grounding=self.config.grounding)
        except Exception:
            stmts = []
        try:
            stmts = ac.map_grounding(stmts)
            stmts = ac.run_preassembly(stmts, return_toplevel=False)
        except Exception as e:
            logger.error(f"Error running preassembly: {e} Problematic statements: {stmts}")
            stmts = []
        return stmts

    def capture_logs_from_stmts_json(self, stmts_json: List[Dict[str, Any]],
                                     on_missing_support: str = 'handle') -> Any:
        log_capture_string = io.StringIO()
        logger = logging.getLogger("indra.statements.io")
        log_handler = logging.StreamHandler(log_capture_string)
        log_handler.setLevel(logging.WARNING)
        logger.addHandler(log_handler)

        try:
            result = stmts_from_json(stmts_json, on_missing_support=on_missing_support)
        finally:
            logger.removeHandler(log_handler)

        log_output = log_capture_string.getvalue().strip()
        log_capture_string.close()

        return result if result else log_output

    def ground_statements(self, statements: List[Any], apply_grounding: bool = False) -> List[Any]:
        grounding_strategy = "None"
        if apply_grounding:
            grounded_statements = ground_statements(statements)
            grounding_strategy = "gilda"
        else:
            grounded_statements = statements
        for stmt in grounded_statements:
            stmt.evidence[0].annotations['grounding_strategy'] = grounding_strategy
        return grounded_statements

    def parse_json_response(self, response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON string provided.") from e
        return response

    def extract_statements(self, stmt_json_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        if stmt_json_response.get("statements"):
            return stmt_json_response["statements"]
        return stmt_json_response if isinstance(stmt_json_response, list) else [stmt_json_response]
