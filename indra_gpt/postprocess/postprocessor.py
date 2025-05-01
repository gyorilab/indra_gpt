class PostProcessor:
    def __init__(self, converter, preassembler):
        self.converter = converter  # instance of LLMToINDRAConverter
        self.preassembler = preassembler  # instance of INDRAPreassembler

    def convert_to_indra_stmts(self, raw_output: str, text: str, pmid: str):
        return self.converter.raw_to_indra_statements(raw_output, text, pmid)

    def preassemble(self, stmts):
        return self.preassembler.preassemble(stmts)
