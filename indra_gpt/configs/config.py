class BaseConfig:
    def __init__(self, kwargs):
        self.config = kwargs  # Store the entire dictionary for reference

class PreProcessingConfig:
    def __init__(self, base_config):
        self.base_config = base_config

        self.user_inputs_file = base_config.config.get("user_inputs_file")
        self.num_samples = base_config.config.get("num_samples")
        self.random_sample = base_config.config.get("random_sample")
        self.n_shot_prompting = base_config.config.get("n_shot_prompting")

class GenerationConfig:
    def __init__(self, base_config):
        self.base_config = base_config

        self.model = base_config.config.get("model")
        self.structured_output = base_config.config.get("structured_output")
        self.feedback_refinement_iterations = base_config.config.get("feedback_refinement_iterations")


class PostProcessingConfig:
    def __init__(self, base_config):
        self.base_config = base_config
        
        self.grounding = base_config.config.get("grounding")
