import json
from indra_gpt.preprocessors.rec_logic_parser.parser import RecLogicParser
from indra_gpt.preprocessors.rec_relation_parser.parser import RecRelationParser

import logging

logger = logging.getLogger(__name__)

PREPROCESSING_METHODS = {
    "rec_logic_parser": RecLogicParser,
    "rec_relation_parser": RecRelationParser,
}

class Preprocessor:
    def __init__(self, llm_client, preprocessing_method=None, **kwargs):
        if preprocessing_method is None:
            self.preprocessor = None

        elif isinstance(preprocessing_method, str):
            preprocessor_class = PREPROCESSING_METHODS.get(preprocessing_method, None)
            if preprocessor_class is None:
                raise ValueError(f"Unknown method: {preprocessing_method}")
            self.preprocessor = preprocessor_class(llm_client, **kwargs)

        else:
            raise ValueError(f"Unknown method: {preprocessing_method}")
        
    def convert_to_ir(self, text: str):
        if self.preprocessor is None:
            logger.info("No preprocessing method specified. Returning original text.")
            return text
        
        return self.preprocessor.convert_to_ir(text)

    def preprocess(self, text: str):
        if self.preprocessor is None:
            logger.info("No preprocessing method specified. Returning original text.")
            return text
        
        return self.preprocessor.preprocess(text)
