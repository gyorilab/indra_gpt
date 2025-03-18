import logging
import json
import io
from typing import List, Dict, Any, Union

from indra.statements import get_all_descendants, Statement
from indra.statements.io import stmts_from_json
from indra_gpt.util.util import sample_from_input_file
from indra_gpt.configs import PostProcessorConfig

from indra.statements import activity_types
from indra.statements import RegulateActivity

from indra.pipeline import register_pipeline
from indra.tools import assemble_corpus as ac
from indra.pipeline.pipeline import AssemblyPipeline
from indra.preassembler.grounding_mapper.gilda import ground_statements

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
            raise ValueError(f"Mismatch in generated responses and input "
                             f"texts: {len(generated_responses)} responses "
                             f"vs {len(input_texts)} input texts.")

        if input_pmids and len(input_pmids) != len(input_texts):
            logger.warning(f"PMID count ({len(input_pmids)}) does not match "
                                f"input text count ({len(input_texts)}). "
                                f"Proceeding without PMIDs where missing.")
        stmts_json_list = self.post_process_json_stmts(generated_responses, 
                                                  input_texts, input_pmids)
        raw_stmts_list = [self.extract_stmts(stmts_json) for stmts_json in stmts_json_list]
        preassembled_stmts = [self.preassembly_pipeline(stmts) 
                              for stmts in raw_stmts_list]

        logger.info(f"Completed post-processing: {len(preassembled_stmts)} "
                    f"statements processed.")
        return preassembled_stmts
    
    def post_process_json_stmts(self, 
                            generated_responses: List[Dict[str, Any]], 
                            input_texts: List[str], 
                            input_pmids: List[str]) -> List[List[Statement]]:
        processed_stmts_json_list = []
        for i, response in enumerate(generated_responses):
            text = input_texts[i]
            pmid = input_pmids[i] if input_pmids else "Not provided"
            try:
                stmt_json_response = self.parse_json_response(response)
                stmts_json = self._get_stmts_json(stmt_json_response)
                stmts_json = [self.map_stmt_class_name(stmt) 
                              for stmt in stmts_json]
                stmts_json = [self.remove_empty_strings_and_lists(stmt) 
                              for stmt in stmts_json]
                stmts_json = [self.update_evidence(stmt, text, pmid, self.config) 
                              for stmt in stmts_json]
                processed_stmts_json_list.append(stmts_json)
            except Exception as e:
                logger.error(f"Error processing response: {response} | "
                             f"Error: {e}")
                processed_stmts_json_list.append({"error": str(e)})
        return processed_stmts_json_list

    def map_stmt_class_name(self, stmt_json: Dict[str, Any]) -> Dict[str, Any]:
        stmt_classes = get_all_descendants(Statement)
        stmt_mapping = {stmt_class.__name__.lower(): stmt_class.__name__ 
                        for stmt_class in stmt_classes}
        stmt_json["type"] = stmt_mapping.get(stmt_json.get("type", "").lower(), 
                                             stmt_json.get("type"))
        return stmt_json

    def remove_empty_strings_and_lists(self, d: Dict[str, Any]
                                       ) -> Dict[str, Any]:
        for key, value in list(d.items()):
            if isinstance(value, dict):
                self.remove_empty_strings_and_lists(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self.remove_empty_strings_and_lists(item)
            if value in [None, [], {}]:
                del d[key]
            if (isinstance(value, str) and 
                value.lower().strip() in ["", "none", "null", "unknown", "na"]):
                del d[key]
        return d

    def update_evidence(self, 
                        stmt_json: Dict[str, Any], 
                        original_input_text: str,
                        pmid: str, config: PostProcessorConfig
                        ) -> Dict[str, Any]:
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
                try:
                    for key, val in evidence['epistemics'].items():
                        if key in ['hypothesis', 'negation', 'direct']:
                            evidence['epistemics'][key] = (val.lower() == 'true')
                except Exception as e:
                    logger.warning(
                                f"Error updating epistemics in evidence: {e}, "
                                f"Evidence: {evidence}"
                            )
        else:
            stmt_json["evidence"] = [actual_evidence]

        return stmt_json

    def preassembly_pipeline(self, stmts_in: List[Statement]) -> Any:
        if not isinstance(stmts_in, list):
            stmts_in = []
        # Initialize an empty pipeline
        pipeline = AssemblyPipeline()

        # Add preprocessing steps
        pipeline.append(ac.filter_no_hypothesis)
        pipeline.append(ac.filter_no_negated)
        pipeline.append(PostProcessor.custom_ground_statements, 
                        self.config.grounding, 
                        self.config.grounding_strategy)
        #pipeline.append(ac.map_grounding, use_adeft=True, gilda_mode="web")
        pipeline.append(ac.filter_grounded_only)
        pipeline.append(ac.filter_genes_only)
        pipeline.append(ac.filter_human_only)
        pipeline.append(PostProcessor.filter_no_RegulateActivity_invalid_activity)
        pipeline.append(ac.run_preassembly, return_toplevel=False)

        # Run the pipeline on the loaded statements
        stmts_out = pipeline.run(stmts_in)

        # Check results
        print(f"Number of statements before processing: {len(stmts_in)}")
        print(f"Number of statements after processing: {len(stmts_out)}")
        return stmts_out

    def extract_stmts(self, 
                      stmts_json: List[Dict[str, Any]], 
                      on_missing_support: str = 'handle') -> Any:
        result = []
        for stmt_json in stmts_json:
            try:
                stmt = stmts_from_json([stmt_json], 
                                        on_missing_support=on_missing_support)
                result += stmt
            except Exception as e:
                logger.warning(f"Error extracting statement: {stmt_json} | "
                                f"Error: {e}")
                result += []
        return result 

    def parse_json_response(self, response: Union[str, Dict[str, Any]]
                            ) -> Dict[str, Any]:
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON string provided.") from e
        return response

    def _get_stmts_json(self, stmt_json_response: Dict[str, Any]
                        ) -> List[Dict[str, Any]]:
        if stmt_json_response.get("statements"):
            return stmt_json_response["statements"]
        if isinstance(stmt_json_response, list):
            return stmt_json_response
        else:
            return [stmt_json_response]

    @staticmethod
    @register_pipeline
    def filter_no_RegulateActivity_invalid_activity(stmts_in: List[Statement]
                                                    ) -> List[Statement]:
        stmts_out = []
        for stmt in stmts_in:
            if isinstance(stmt, RegulateActivity):
                if stmt.obj_activity in activity_types:
                    stmts_out.append(stmt)
            else:
                stmts_out.append(stmt)
        return stmts_out
    
    @staticmethod
    @register_pipeline
    def custom_ground_statements(stmts_in: List[Statement],
                          grounding: bool,
                          grounding_strategy: str) -> List[Statement]:
        stmts_out = []
        if grounding:
            if grounding_strategy == 'gilda':
                stmts_out = ground_statements(stmts_in)
            else:
                raise ValueError(
                    f"Invalid grounding strategy: {grounding_strategy}"
                    f"Valid options are: ['gilda']"
                )
        else:
            stmts_out = stmts_in
        return stmts_out

