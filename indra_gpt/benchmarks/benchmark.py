import sys
from indra.statements.io import stmts_from_json, stmt_from_json
from indra_gpt.resources.constants import OUTPUT_DEFAULT
from indra.preassembler.grounding_mapper.gilda import ground_statements
from indra_gpt.api.api import generate_statements_with_client
from indra_gpt.util import post_process_extracted_json
from json import JSONDecodeError
import json
import pandas as pd

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

    def get_benchmark_df(self):
        self._load_benchmark()
        self._generate_statements()

        df = pd.DataFrame({
            "original_statements_json": self.original_statements_json,
            "generated_statements_json": self.generated_statements_json,
            "original_statements": self.original_statements,
            "generated_statements": self.generated_statements
        })
        for i, row in df.iterrows():
            original_stmts = row["original_statements"]
            generated_stmts = row["generated_statements"]
            try:
                results = self.compare_statement_to_statements(original_stmts, generated_stmts)
            except Exception as e:
                results = f"Error: {e}"
            df.at[i, "results"] = results
        return df

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
                    self.generated_statements.append(str(stmts_indra))
                else:   # output is a single json object of a statement
                    stmt_json = json.loads(generated_statement_json_object)                    
                    stmt_json = post_process_extracted_json(stmt_json)
                    stmt_indra = stmt_from_json(stmt_json)
                    self.generated_statements.append(str(stmt_indra))
            except (JSONDecodeError, IndexError, TypeError) as e:
                logger.error(f"Error extracting statement: {e}")
                self.generated_statements.append(f"Error: {e}")

    def equals(self, stmt1, stmt2):
        return stmt1.equals(stmt2)
    
    def equals_type(self, stmt1, stmt2):
        return type(stmt1) == type(stmt2)
    
    def same_set_of_agents(self, stmt1, stmt2):
        stmt1_agents = stmt1.real_agent_list()
        stmt2_agents = stmt2.real_agent_list()
        stmt1_agents_grounded = set(x.get_grounding() for x in stmt1_agents if x is not None)
        stmt2_agents_grounded = set(x.get_grounding() for x in stmt2_agents if x is not None)
        return stmt1_agents_grounded == stmt2_agents_grounded
    
    def compare_two_statements(self, stmt1, stmt2):
        equals = self.equals(stmt1, stmt2)
        equals_type = self.equals_type(stmt1, stmt2)
        same_set_of_agents = self.same_set_of_agents(stmt1, stmt2)
        return equals, equals_type, same_set_of_agents
    
    def compare_statement_to_statements(self, stmt, stmts):
        results = []
        for s in stmts:
            results.append(self.compare_two_statements(stmt, s))
        return results

