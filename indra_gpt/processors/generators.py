import logging
from indra_gpt.configs import GenerationConfig
from indra_gpt.clients import OpenAIClient, AnthropicClient

MODEL_CLIENTS = {
    "gpt-4o-mini": OpenAIClient,
    "claude-3-5-sonnet-latest": AnthropicClient
}

class Generator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def log_config(self):
        # Create a filtered dictionary excluding 'base_config'
        config_dict = {k: v for k, v in self.config.__dict__.items() if k != "base_config"}
        
        self.logger.info(f"Config used for generation: {config_dict}")

    def validate_model(self):
        model = self.config.base_config.model
        if model not in MODEL_CLIENTS:
            raise ValueError(f"Unsupported model '{model}'. Supported: {list(MODEL_CLIENTS.keys())}")
        return model

    def generate(self, preprocessed_data):
        self.log_config()
        model = self.validate_model()
        # Instantiate the appropriate model client dynamically
        client = MODEL_CLIENTS[model](self.config)

        raw_responses, extracted_statement_json_objects = client.generate(preprocessed_data)
        
        return raw_responses, extracted_statement_json_objects
    
    def generate_raw_response(self, preprocessed_data):
        self.log_config()
        model = self.validate_model()
        # Instantiate the appropriate model client dynamically
        client = MODEL_CLIENTS[model](self.config)
        
        return client.generate_raw_responses(preprocessed_data)
