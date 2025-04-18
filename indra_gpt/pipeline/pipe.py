class Pipe:
    def __init__(self, func):
        self.func = func

    def __ror__(self, other):
        return self.func(other)

    def __call__(self, *args, **kwargs):
        return Pipe(lambda x: self.func(x, *args, **kwargs))


class ExtractionPipeline:
    def __init__(self, preprocessor=None, extractor=None, postprocessor=None):
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.postprocessor = postprocessor

    def preprocess(self):
        return Pipe(lambda text: self.preprocessor.convert_to_ir(text))

    def raw_extract(self):
        return Pipe(lambda ir: self.extractor.raw_extract(ir))

    def convert_to_indra_stmts(self, text, pmid):
        return Pipe(lambda raw: self.postprocessor.convert_to_indra_stmts(raw, text, pmid))

    def preassemble(self):
        return Pipe(lambda stmts: self.postprocessor.preassemble(stmts))
