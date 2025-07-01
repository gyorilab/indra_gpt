from abc import ABC, abstractmethod
import litellm

class LLMClient(ABC):
    """
    Abstract base class for LLM clients. Defaults to LitellmClient
    if instantiated directly.
    """

    @abstractmethod
    def call(self, prompt: str, history: list = None) -> str:
        pass

    def __new__(cls, *args, **kwargs):
        if cls is LLMClient:
            return LitellmClient(*args, **kwargs)
        return super().__new__(cls)

class LitellmClient(LLMClient):
    def __init__(
        self,
        custom_llm_provider: str = 'openai',
        api_key: str = None,
        api_base: str = None,
        model: str = None, 
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ):
        self.custom_llm_provider = custom_llm_provider
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_params = kwargs

    def call(self, prompt: str, history: list = None, **kwargs) -> str:
        messages = (history or []) + [{"role": "user", "content": prompt}]

        # Override defaults with safe pop to avoid duplication
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)

        response = litellm.completion(
            custom_llm_provider=self.custom_llm_provider,
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs  # now safe
        )
        return response.choices[0].message["content"]
