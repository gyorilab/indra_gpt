import logging
from typing import Dict, Type, Any, List

from indra_gpt.configs import GenerationConfig
from indra_gpt.clients import OpenAIClient, AnthropicClient

MODEL_CLIENTS: Dict[str, Type[OpenAIClient | AnthropicClient]] = {
    "gpt-4o-mini": OpenAIClient,
    "claude-3-5-sonnet-latest": AnthropicClient,
}

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def log_config(self) -> None:
        config_dict = {k: v for k, v in self.config.__dict__.items() \
                       if k != "base_config"}
        logger.info(f"Config used for generation: {config_dict}")

    def validate_model(self) -> str:
        model = self.config.base_config.model
        if model not in MODEL_CLIENTS:
            raise ValueError(
                f"Unsupported model '{model}'. "
                f"Supported: {list(MODEL_CLIENTS.keys())}"
            )
        return model

    def generate(self, preprocessed_data: Dict[str, Any]
                 ) -> List[Dict[str, Any]]:
        self.log_config()
        model = self.validate_model()

        client = MODEL_CLIENTS[model](self.config)
        extracted_statement_json_objects = client.generate(preprocessed_data)

        return extracted_statement_json_objects
