class BaseConfig:
    def __init__(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)  # Dynamically set attributes

class PreProcessingConfig:
    def __init__(self, base_config):
        self.base_config = base_config

        self.user_inputs_file = base_config.user_inputs_file
        self.num_samples = base_config.num_samples
        self.random_sample = base_config.random_sample
        self.n_shot_prompting = base_config.n_shot_prompting

class GenerationConfig:
    def __init__(self, base_config):
        self.base_config = base_config

        self.model = base_config.model
        self.structured_output = base_config.structured_output
        self.self_correction_iterations = base_config.self_correction_iterations

class PostProcessingConfig:
    def __init__(self, base_config):
        self.base_config = base_config
        
        self.grounding = base_config.grounding
