import sys
from indra.statements.io import stmts_from_json, stmt_from_json
import indra.statements
from indra_gpt.resources.constants import OUTPUT_DEFAULT
from indra.preassembler.grounding_mapper.gilda import ground_statements
from indra_gpt.api.api import generate_statements_with_client
from indra_gpt.util import post_process_extracted_json
from json import JSONDecodeError
import json
import pandas as pd

from indra.preassembler.grounding_mapper.gilda import ground_statements

import logging
logger = logging.getLogger(__name__)

class Benchmark:

    def __init__(self, model, benchmark_file, structured_output, n_statements):
        self.model = model
        self.benchmark_file = benchmark_file
        self.structured_output = structured_output
        self.n_statements = n_statements
        self.original_statements_json = None
        self.orginal_statements = None
        self.generated_statements_json = None
        self.generated_statements = None

    def get_comparison_df(self):
        df = self.get_results_df()
        df['comparison_result'] = df.apply(
            lambda x: self.compare_lists_of_statements(
                x['original_statements'] if isinstance(x['original_statements'], list) else [],
                x['generated_statements'] if isinstance(x['generated_statements'], list) else []
            ), axis=1
        )
        df['comparison_result_grounded'] = df.apply(
            lambda x: self.compare_lists_of_statements(
                x['original_statements_grounded'] if isinstance(x['original_statements_grounded'], list) else [],
                x['generated_statements_grounded'] if isinstance(x['generated_statements_grounded'], list) else []
            ), axis=1
        )
        return df
    
    def get_results_df(self):
        self._load_benchmark()
        self._generate_statements()

        df = pd.DataFrame({
            "original_statements_json": self.original_statements_json,
            "generated_statements_json": self.generated_statements_json,
            "original_statements": self.original_statements,
            "generated_statements": self.generated_statements
        })

        df['original_statements_grounded'] = df['original_statements'].apply(self._safe_ground_statements)
        df['generated_statements_grounded'] = df['generated_statements'].apply(self._safe_ground_statements)

        return df
    
    def _safe_ground_statements(self, statements):
            """Runs ground_statements() with error handling."""
            try:
                return ground_statements(statements) if isinstance(statements, list) else statements
            except Exception as e:
                print(f"Error grounding statements: {e}")
                return statements  # Return original statements on failure

    def _load_benchmark(self):
        with open(self.benchmark_file, "r") as f:
            self.original_statements_json = json.load(f)[:self.n_statements]
        self.original_statements = [stmts_from_json([stmt_json]) for stmt_json in self.original_statements_json][:self.n_statements]

    def _generate_statements(self):
        kwargs = {
            "statements_file_json": self.benchmark_file,
            "model": self.model,
            "output_file": OUTPUT_DEFAULT,
            "iterations": self.n_statements,
            "verbose": False,
            "batch_job": False,
            "batch_id": None,
            "structured_output": self.structured_output
        }
        self.generated_statements_json = generate_statements_with_client(**kwargs)
        self.generated_statements = []
        for generated_statement_json_object in self.generated_statements_json:
            try: 
                if self.structured_output: # output is a json object with property 'statements' which is a list of statements
                    stmts_json = json.loads(generated_statement_json_object)['statements']
                    stmts_json = [post_process_extracted_json(stmt_json) for stmt_json in stmts_json]
                    stmts_indra = [stmt_from_json(stmt_json) for stmt_json in stmts_json]
                    self.generated_statements.append(stmts_indra)
                else:   # output is a single json object of a statement
                    stmt_json = json.loads(generated_statement_json_object)                    
                    stmt_json = post_process_extracted_json(stmt_json)
                    stmt_indra = stmt_from_json(stmt_json)
                    self.generated_statements.append([stmt_indra])
            except (JSONDecodeError, IndexError, TypeError) as e:
                logger.error(f"Error extracting statement: {e}")
                self.generated_statements.append(f"Error: {e}")

    def compare_lists_of_statements(self, stmt_list1, stmt_list2):
        results = []
        
        # Ensure valid lists
        if not isinstance(stmt_list1, list) or not isinstance(stmt_list2, list):
            return [{"error": "Invalid input (expected lists)"}]
        
        if not stmt_list1 or not stmt_list2:  # Handle empty lists
            return [{"error": "One or both lists are empty"}]

        for s1 in stmt_list1:
            for s2 in stmt_list2:
                d = {}
                equals, equals_type, agents_equal = self.compare_two_statements(s1, s2)
                d['equals'] = equals
                d['equals_type'] = equals_type
                d['agents_equal'] = agents_equal
                results.append(d)

        return results
    
    def compare_two_statements(self, stmt1, stmt2):
        equals = stmt1.equals(stmt2)
        equals_type = stmt1.types_equals(stmt2)
        agents_equal = stmt1.agents_equal(stmt2),
        return equals, equals_type, agents_equal
