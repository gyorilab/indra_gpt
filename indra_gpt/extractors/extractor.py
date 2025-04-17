from indra_gpt.extractors.end_to_end.end_to_end_extractor import EndToEndExtractor
# later: from extractors.recursive.recursive_extractor import RecursiveExtractor

EXTRACTION_METHODS = {
    "end_to_end": EndToEndExtractor,
    # "recursive": RecursiveExtractor,
}

class Extractor:
    def __init__(self, llm_client, extraction_method="end_to_end", **kwargs):
        if isinstance(extraction_method, str):
            extractor_class = EXTRACTION_METHODS.get(extraction_method)
            if extractor_class is None:
                raise ValueError(f"Unknown method: {extraction_method}")
            self.extractor = extractor_class(llm_client, **kwargs)
        else:
            # If a class or object was passed directly
            self.extractor = extraction_method(llm_client, **kwargs)

    def extract(self, text: str):
        return self.extractor.extract(text)
