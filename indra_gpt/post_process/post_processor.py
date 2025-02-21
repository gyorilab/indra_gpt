from typing import Union
import ast
from indra.statements import get_all_descendants, Statement
from typing import Union
import ast
from indra.preassembler.grounding_mapper.gilda import ground_statements


class PostProcessor:
    def __init__(self, config=None):
        self.config = config

    def trim_stmt_json(self, stmt):
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

    def get_input_text_from_original_statement_json(self, original_statement_json_object:Union[str, dict]):
        if isinstance(original_statement_json_object, str):
            try:
                original_statement_json_object = ast.literal_eval(original_statement_json_object)
            except Exception as e:
                raise e
        evidence = original_statement_json_object["evidence"]
        return evidence[0]["text"]

    def get_pmid_from_original_statement_json(self, original_statement_json_object:Union[str, dict]):
        if isinstance(original_statement_json_object, str):
            try: 
                original_statement_json_object = ast.literal_eval(original_statement_json_object)
            except Exception as e:
                raise e
        evidence = original_statement_json_object["evidence"]
        return evidence[0]["pmid"]

    def update_evidence(self, generated_statement_json_object:Union[str, dict], input_text:str, pmid:Union[str, int]):
        if isinstance(generated_statement_json_object, str):
            try:
                generated_statement_json_object = ast.literal_eval(generated_statement_json_object)
            except Exception as e:
                raise e     
        actual_evidence = [{'text': input_text, 'pmid': pmid, "source_api": "indra_gpt", "annotations": self.config}]
        # prepend the evidence to the existing evidence list
        generated_statement_json_object["evidence"]= actual_evidence
        return generated_statement_json_object

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
    
    def ground_and_annotate_statements(self, statements):
        """Runs ground_statements() with error handling."""
        try:
            grounded_statements = ground_statements(statements) if isinstance(statements, list) else statements
        except Exception as e:
            print(f"Error grounding statements: {e}")
            grounded_statements = statements  # Return original statements on failure
        for grounded_statement in grounded_statements:
            grounded_statement.evidence[0].annotations['grounding_strategy'] = "gilda"
        return grounded_statements

    def post_process_extracted_statement_json(self, gpt_stmt_json, input_text=None, pmid=None, update_evidence=False):
        stmt_classes = get_all_descendants(Statement)
        stmt_mapping = {stmt_class.__name__.lower(): stmt_class.__name__ for stmt_class in stmt_classes}
        try:
            stmt_type = gpt_stmt_json["type"]
            mapped_type = stmt_mapping.get(stmt_type.lower(), stmt_type)
            gpt_stmt_json["type"] = mapped_type
        except KeyError:
            pass

        gpt_stmt_json = self.remove_empty_strings_and_lists(gpt_stmt_json)

        if update_evidence:
            gpt_stmt_json = self.update_evidence(gpt_stmt_json, input_text, pmid)

        return gpt_stmt_json
