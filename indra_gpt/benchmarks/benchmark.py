import sys
from indra.statements.io import stmts_from_json, stmt_from_json
import indra.statements
from indra_gpt.resources.constants import OUTPUT_DEFAULT
from indra.preassembler.grounding_mapper.gilda import ground_statements
from indra_gpt.api.api import generate_statements_with_client
from json import JSONDecodeError
import json
import pandas as pd
import numpy as np

from indra.preassembler.grounding_mapper.gilda import ground_statements
from indra_gpt.post_process.post_processor import PostProcessor

import logging
logger = logging.getLogger(__name__)

class Benchmark:

    def __init__(self, model, benchmark_file, structured_output, n_statements, random_sample):
        self.config = {
            "model": model,
            "benchmark_file": benchmark_file,
            "structured_output": structured_output,
            "n_statements": n_statements,
            "random_sample": random_sample
        }
        self.original_statement_json = None
        self.orginal_statement = None
        self.generated_statements_json = None
        self.generated_statements = None
    
    def compute_comparison_statistics(self, df, column_name="comparison_result", index_column="best_match_index"):
        """Compute accuracy statistics from the best match in comparison results."""
        stats = {}

        # Filter out invalid indices
        valid_rows = df[df[index_column] >= 0]

        # Extract individual metrics safely
        equals_values = []
        equals_type_values = []
        agents_equal_values = []

        for _, row in valid_rows.iterrows():
            best_index = row[index_column]
            comparison_list = row[column_name]

            if isinstance(comparison_list, list) and 0 <= best_index < len(comparison_list):
                best_match = comparison_list[best_index]

                if isinstance(best_match, dict):  # Ensure it's a dict
                    equals_values.append(best_match.get("equals", False))
                    equals_type_values.append(best_match.get("equals_type", False))
                    agents_equal_values.append(best_match.get("agents_equal", False))

        # Compute accuracy, handling empty lists
        stats["config"] = self.config
        stats["equals_accuracy"] = float(np.mean(equals_values)) if equals_values else 0.0
        stats["equals_type_accuracy"] = float(np.mean(equals_type_values)) if equals_type_values else 0.0
        stats["agents_equal_accuracy"] = float(np.mean(agents_equal_values)) if agents_equal_values else 0.0

        return stats
    
    def get_comparison_df(self):
        df = self.get_results_df()

        df['comparison_result'] = df.apply(
            lambda x: self.compare_lists_of_statements(
                x['original_statement'] if isinstance(x['original_statement'], list) else [],
                x['generated_statements'] if isinstance(x['generated_statements'], list) else []
            ), axis=1
        )

        # Save best match index instead of full dictionary
        df['best_match_index'] = df['comparison_result'].apply(self.get_best_match_index)

        return df
    
    def get_best_match_index(self, comparison_results):
        """
        Given a list of comparison results (list of dicts), return the index of the best match.
        Prioritizes 'equals', then 'equals_type', then 'agents_equal'.
        Returns -1 if no valid match is found.
        """
        if not isinstance(comparison_results, list) or not comparison_results:
            return -1  # Return -1 for empty or invalid lists

        best_index = -1
        best_score = (-1, -1, -1)  # (equals, equals_type, agents_equal)

        for i, result in enumerate(comparison_results):
            if not isinstance(result, dict):  # Skip invalid results
                continue

            # Convert boolean values to numeric scores (True=1, False=0)
            score = (
                int(result.get('equals', False)),
                int(result.get('equals_type', False)),
                int(result.get('agents_equal', False))
            )

            # Update best match index if the score is better
            if score > best_score:
                best_index = i
                best_score = score

        return best_index
        
    def get_results_df(self):
        self._generate_statements()

        df = pd.DataFrame({
            "original_statement_json": self.original_statement_json,
            "generated_statements_json": self.generated_statements_json,
            "original_statement": self.original_statement,
            "generated_statements": self.generated_statements
        })

        return df
    
    def _safe_ground_statements(self, statements):
            """Runs ground_statements() with error handling."""
            try:
                return ground_statements(statements) if isinstance(statements, list) else statements
            except Exception as e:
                print(f"Error grounding statements: {e}")
                return statements  # Return original statements on failure

    def _generate_statements(self):
        kwargs = {
            "statements_file_json": self.config['benchmark_file'],
            "model": self.config['model'],
            "output_file": OUTPUT_DEFAULT,
            "iterations": self.config['n_statements'],
            "verbose": False,
            "batch_job": False,
            "batch_id": None,
            "structured_output": self.config['structured_output'],
            "random_sample": self.config['random_sample']
        }
        self.original_statement_json, self.generated_statements_json = generate_statements_with_client(**kwargs)
        self.original_statement = [stmts_from_json([stmt_json]) for stmt_json in self.original_statement_json][:self.config['n_statements']]

        config = {
            "model": self.config['model'],
            "num_samples_from_corpus": self.config['n_statements'],
            "structured_output": self.config['structured_output'],
            "random_sample": self.config['random_sample']
        }

        post_processor = PostProcessor(config)
        input_texts = [post_processor.get_input_text_from_original_statement_json(x) for x in self.original_statement_json]
        pmids = [post_processor.get_pmid_from_original_statement_json(x) for x in self.original_statement_json]
        self.generated_statements = []
        for generated_statement_json_object, input_text, pmid in zip(self.generated_statements_json, input_texts, pmids):
            try: 
                if self.config['structured_output']: # output is a json object with property 'statements' which is a list of statements
                    stmts_json = json.loads(generated_statement_json_object)['statements']
                    post_processed_stmts_json = [post_processor.post_process_extracted_statement_json(stmt_json, input_text, pmid, update_evidence=True) 
                                                 for stmt_json in stmts_json]
                    stmts_indra = stmts_from_json(post_processed_stmts_json)
                    grounded_stmts_indra = post_processor.ground_and_annotate_statements(stmts_indra)
                    self.generated_statements.append(grounded_stmts_indra)
                else:   # output is a single json object of a statement
                    stmt_json = json.loads(generated_statement_json_object)                    
                    post_processed_stmt_json = post_processor.post_process_extracted_statement_json(stmt_json, input_text, pmid, update_evidence=True)
                    stmt_indra = stmt_from_json(post_processed_stmt_json)
                    grounded_stmts_indra = post_processor.ground_and_annotate_statements([stmt_indra])
                    self.generated_statements.append(grounded_stmts_indra)
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
        agents_equal = stmt1.agents_equal(stmt2)
        return equals, equals_type, agents_equal
