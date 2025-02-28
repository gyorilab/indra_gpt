from typing import Union
import ast
from indra.statements import get_all_descendants, Statement
from typing import Union
import ast
from indra.preassembler.grounding_mapper.gilda import ground_statements
from indra.tools import assemble_corpus as ac
import json
from indra.statements.io import stmts_from_json, stmt_from_json
import logging
import io


class PostProcessor:
    def __init__(self, config=None):
        self.config = config

    def get_input_text_from_original_statement_json(self, original_statement_json_object:Union[str, dict]):
        if isinstance(original_statement_json_object, str):
            try:
                original_statement_json_object = json.loads(original_statement_json_object)
            except Exception as e:
                raise e
        evidence = original_statement_json_object["evidence"]
        return evidence[0]["text"]

    def get_pmid_from_original_statement_json(self, original_statement_json_object:Union[str, dict]):
        if isinstance(original_statement_json_object, str):
            try: 
                original_statement_json_object = json.loads(original_statement_json_object)
            except Exception as e:
                raise e
        evidence = original_statement_json_object["evidence"]
        return evidence[0]["pmid"]

    def update_evidence(self, stmt_json, **kwargs):
        # Define the actual evidence metadata
        actual_evidence = {
            "text": kwargs.get("input_text", ""),
            "pmid": kwargs.get("pmid", ""),
            "source_api": "indra_gpt",
            "annotations": {"indra_gpt_config": self.config}
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

    # Recursively go through each key-value, and if it is an empty string or list or dict, remove the key-value pair
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

    def post_process_generated_response(self, generated_response, **kwargs):
        # 1. Convert to json object if it is a string
        if isinstance(generated_response, str):
            try:
                stmt_json_response = json.loads(generated_response)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON string provided.") from e
        else: 
            stmt_json_response = generated_response
        # 2. If the JSON object contains a list of statementts under the "statements" key,
        # retrieve the value.
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
        stmts_json = [self.update_evidence(stmt_json, **kwargs) for stmt_json in stmts_json]
        return stmts_json

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
        
        # If `stmts_from_json` returns an empty list but there were errors, return two copies of the log output
        if not result and log_output:
            return log_output  
        
        return result

    def preassemble_pipeline(self, stmts_json, apply_grounding=True):
        # 1. Convert to indra statements
        stmts = self.capture_logs_from_stmts_json(stmts_json)
        if not isinstance(stmts, list):
            return stmts
        # 2. Ground the statements
        try:
            stmts = self.ground_statements(stmts, apply_grounding=apply_grounding)
        except Exception as e:
            stmts = []
        # 3. Run preassembly
        stmts = ac.map_grounding(stmts)
        stmts = ac.run_preassembly(stmts, return_toplevel=False)
        return stmts
