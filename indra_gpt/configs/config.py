from typing import Any, Mapping

class BaseConfig:
    def __init__(self, kwargs: Mapping[str, Any]) -> None:
        self.user_inputs_file = kwargs.get("user_inputs_file")
        self.num_samples = kwargs.get("num_samples")
        self.random_sample = kwargs.get("random_sample")
        self.random_seed = kwargs.get("random_seed")
        self.output_file = kwargs.get("output_file")
        self.model = kwargs.get("model")
        self.structured_output = kwargs.get("structured_output")
        self.n_shot_prompting = kwargs.get("n_shot_prompting")
        self.self_correction_iterations = kwargs.get("self_correction_iterations")
        self.grounding = kwargs.get("grounding")


class ProcessorConfig:
    def __init__(self, base_config: BaseConfig) -> None:
        self.base_config = base_config


class PreProcessorConfig(ProcessorConfig):
    def __init__(self, base_config: BaseConfig) -> None:
        super().__init__(base_config)

        self.user_inputs_file = base_config.user_inputs_file
        self.num_samples = base_config.num_samples
        self.random_sample = base_config.random_sample
        self.n_shot_prompting = base_config.n_shot_prompting


class GenerationConfig(ProcessorConfig):
    def __init__(self, base_config: BaseConfig) -> None:
        super().__init__(base_config)

        self.model = base_config.model
        self.structured_output = base_config.structured_output
        self.self_correction_iterations = base_config.self_correction_iterations


class PostProcessorConfig(ProcessorConfig):
    def __init__(self, base_config: BaseConfig) -> None:
        super().__init__(base_config)
        
        self.grounding = base_config.grounding
