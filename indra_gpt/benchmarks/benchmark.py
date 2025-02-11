import sys
import indra.statements
import indra_gpt.resources.constants
from indra.preassembler.grounding_mapper.gilda import ground_statements

class Benchmark:

    def __init__(self):
        self.benchmark_corpus = indra_gpt.resources.constants.INPUT_DEFAULT

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
    
    def compare_statements_to_statements(self, stmt, stmts):
        results = []
        for s in stmts:
            results.append(self.compare_two_statements(stmt, s))
        return results
