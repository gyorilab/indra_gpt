class PostProcessor:
    def __init__(self, converter, preassembler):
        self.converter = converter
        self.preassembler = preassembler

    def convert_to_indra_stmts(self, raw_output: str, text: str, pmid: str):
        return self.converter.raw_to_indra_statements(raw_output, text, pmid)
    
    def default_grounder(self, stmts):
        def _apply_grounding(obj):
            if isinstance(obj, dict):
                # If current dict has 'name' and 'db_refs'
                if 'name' in obj and 'db_refs' in obj and isinstance(obj['db_refs'], dict):
                    obj['db_refs']['TEXT'] = obj['name']
                # Recurse on nested dicts and lists
                for v in obj.values():
                    _apply_grounding(v)
            elif isinstance(obj, list):
                for item in obj:
                    _apply_grounding(item)

        for stmt in stmts:
            _apply_grounding(stmt)

        return stmts


    def preassemble(self, stmts):
        return self.preassembler.preassemble(stmts)
