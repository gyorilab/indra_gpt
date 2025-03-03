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

    def generate(self, preprocessed_data):
        model_name = self.config.base_config.get("model_name", "gpt-4o-mini")
        if model_name not in MODEL_CLIENTS:
            raise ValueError(f"Unsupported model '{model_name}'. Supported: {list(MODEL_CLIENTS.keys())}")

        self.logger.info(f"Using model: {model_name}")
        
        # Instantiate the appropriate model client dynamically
        client = MODEL_CLIENTS[model_name](self.config)
        
        return client.generate_statement_json_objects(preprocessed_data)
