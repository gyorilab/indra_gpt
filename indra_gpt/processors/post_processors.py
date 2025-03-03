import logging
import json
from indra.statements import get_all_descendants, Statement
from indra.tools import assemble_corpus as ac
import io
from indra.statements.io import stmts_from_json
from indra.preassembler.grounding_mapper.gilda import ground_statements
from indra_gpt.util.util import sample_from_input_file
from indra_gpt.configs import PostProcessingConfig

class PostProcessor:
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process(self, generated_responses):
        self.logger.info("Starting post-processing of extracted statements...")

        input_samples = sample_from_input_file(self.config, self.config.base_config.random_seed)
        input_texts = [entry["text"] for entry in input_samples]
        input_pmids = [entry["pmid"] for entry in input_samples]

        # Check for mismatch in input texts and generated responses
        if len(generated_responses) != len(input_texts):
            raise ValueError(
                f"Mismatch in generated responses and input texts: "
                f"{len(generated_responses)} responses vs {len(input_texts)} input texts."
            )
        # Ensure pmids are available and correctly sized
        if input_pmids and len(input_pmids) != len(input_texts):
            self.logger.warning(
                f"PMID count ({len(input_pmids)}) does not match input text count ({len(input_texts)}). "
                "Proceeding without PMIDs where missing."
            )

        processed_stmts_json_list = []
        for i, response in enumerate(generated_responses):
            text = input_texts[i]
            pmid = input_pmids[i] if input_pmids else "Not provided"
            try:
                # 1. Convert to json object if it is a string
                if isinstance(response, str):
                    try:
                        stmt_json_response = json.loads(response)
                    except json.JSONDecodeError as e:
                        raise ValueError("Invalid JSON string provided.") from e
                else: 
                    stmt_json_response = response
                # 2. If the JSON object contains a list of statements under the "statements" key,
                # retrieve the value. This is currently specifically for the case
                # of openai chat completions for structured output mode using a specific schema. 
                if stmt_json_response.get("statements"):
                    stmts_json = stmt_json_response["statements"]
                else:
                    if isinstance(stmt_json_response, list):
                        stmts_json = stmt_json_response
                    else:
                        stmts_json = [stmt_json_response]
                # 3. Map the stmt class names to the correct class names 
                stmts_json = [self.map_stmt_class_name(stmt_json) for stmt_json in stmts_json]
                # 4. Remove fields with null values
                stmts_json = [self.remove_empty_strings_and_lists(stmt_json) for stmt_json in stmts_json]
                # 5. Update evidence since the generated statements likely have dummy values for evidence
                stmts_json = [self.update_evidence(stmt_json, text, pmid, self.config) for stmt_json in stmts_json]
                processed_stmts_json_list.append(stmts_json)
            except Exception as e:
                self.logger.error(f"Error processing response: {response} | Error: {e}")
                processed_stmts_json_list.append({"error": str(e)})

        preassembled_statements = [self.preassemble_pipeline(stmt) for stmt in processed_stmts_json_list]
        self.logger.info(f"Completed post-processing: {len(preassembled_statements)} statements processed.")
        return preassembled_statements
    
    def map_stmt_class_name(self, stmt_json):
        stmt_classes = get_all_descendants(Statement)
        stmt_mapping = {stmt_class.__name__.lower(): stmt_class.__name__ for stmt_class in stmt_classes}
        try:
            stmt_type = stmt_json["type"]
            mapped_type = stmt_mapping.get(stmt_type.lower(), stmt_type)
            stmt_json["type"] = mapped_type
        except KeyError:
            pass
        return stmt_json
    
    def remove_empty_strings_and_lists(self, d):
        for key, value in list(d.items()):
            if isinstance(value, dict):
                self.remove_empty_strings_and_lists(value)
            elif isinstance(value, list):
                for i in value:
                    if isinstance(i, dict):
                        self.remove_empty_strings_and_lists(i)
            if value in [None, "", [], {}]:
                del d[key]
        return d
    
    def update_evidence(self, stmt_json, original_input_text, pmid, config):
        # Define the actual evidence metadata
        actual_evidence = {
            "text": original_input_text,
            "pmid": pmid,
            "source_api": "indra_gpt",
            "annotations": {"indra_gpt_config": config.base_config.__dict__}
        }

        # Check if "evidence" exists and is a list
        evidence_list = stmt_json.get("evidence", [])

        if evidence_list:
            # Update existing evidence entries
            for evidence in evidence_list:
                evidence.update(actual_evidence)
        else:
            # If "evidence" is empty or missing, add the first evidence entry
            stmt_json["evidence"] = [actual_evidence]

        return stmt_json

    def preassemble_pipeline(self, stmts_json):
        # 1. Convert to indra statements
        stmts = self.capture_logs_from_stmts_json(stmts_json)
        if not isinstance(stmts, list):
            return stmts
        # 2. Ground the statements
        try:
            stmts = self.ground_statements(stmts, apply_grounding=self.config.grounding)
        except Exception as e:
            stmts = []
        # 3. Run preassembly
        stmts = ac.map_grounding(stmts)
        stmts = ac.run_preassembly(stmts, return_toplevel=False)
        return stmts
    
    def capture_logs_from_stmts_json(self, stmts_json, on_missing_support='handle'):
        # Create a string buffer to capture logs
        log_capture_string = io.StringIO()

        # Get the logger used in `indra.statements.io`
        logger = logging.getLogger("indra.statements.io")
        
        # Set up a temporary logging handler
        log_handler = logging.StreamHandler(log_capture_string)
        log_handler.setLevel(logging.WARNING)  # Capture WARNING and above
        logger.addHandler(log_handler)

        try:
            # Call stmts_from_json()
            result = stmts_from_json(stmts_json, on_missing_support=on_missing_support)

        finally:
            # Remove the temporary handler
            logger.removeHandler(log_handler)
        
        # Get the captured log output as a string
        log_output = log_capture_string.getvalue().strip()
        
        # Close the StringIO object
        log_capture_string.close()
        
        if not result and log_output:
            return log_output  
        
        return result

    def ground_statements(self, statements, apply_grounding=False):
        """Runs ground_statements() with error handling."""
        grounding_strategy = "None"
        if apply_grounding:
            grounded_statements = ground_statements(statements)
            grounding_strategy = "gilda"
        else:
            grounded_statements = statements
        for grounded_statement in grounded_statements:
            grounded_statement.evidence[0].annotations['grounding_strategy'] = grounding_strategy
        return grounded_statements
