import logging
from time import sleep
import json
import anthropic
from indra.config import IndraConfigError, get_config
from indra_gpt.resources.constants import (USER_INIT_PROMPT_REDUCED,
                                           GENERIC_REFINEMENT_PROMPT,
                                           STATEMENT_TYPE_REFINEMENT_PROMPT,
                                           ERROR_CONTEXT_PROMPT,
                                           SCHEMA_STRUCTURED_OUTPUT_PATH,
                                           JSON_SCHEMA)
from indra_gpt.util.util import merge_allOf
from tqdm import tqdm
from indra_gpt.configs import GenerationConfig
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    api_key = get_config("ANTHROPIC_API_KEY", failure_ok=False)
except IndraConfigError as err:
    raise KeyError(
        "Please set ANTHROPIC_API_KEY in the environment or in the indra config."
    ) from err

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=api_key
)

with open(SCHEMA_STRUCTURED_OUTPUT_PATH) as file:
    schema = json.load(file)
    post_processed_schema = merge_allOf(schema, schema)

with open(GENERIC_REFINEMENT_PROMPT, "r", encoding="utf-8") as file:
    generic_refinement_prefix = file.read()

with open(STATEMENT_TYPE_REFINEMENT_PROMPT, "r", encoding="utf-8") as file:
    statement_type_correction_prefix = file.read()

with open(ERROR_CONTEXT_PROMPT, "r", encoding="utf-8") as file:
    error_context_correction_prefix = file.read()

class AnthropicClient:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def get_chat_prompt(self, input_text: str) -> Dict[str, str]:
        """
        Constructs a user prompt for chat-based LLM inference.
        """
        with open(USER_INIT_PROMPT_REDUCED, "r", encoding="utf-8") as file:
            PROMPT_reduced = file.read()
        prompt = PROMPT_reduced + input_text
        return {"role": "user", "content": prompt}
    
    def make_api_call(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1, 
        retry_count: int = 3
    ) -> Any:
        """
        Makes an API call to Anthropic with retry logic.

        Returns:
            Response object from Anthropic API.
        """
        retry_count = max(retry_count, 1)
        response = None

        for i in range(retry_count):
            try:
                if self.config.structured_output:
                    logger.info("Anthropic API uses tool calling which many "
                                "not be as reliable for structured output.")
                    response = client.messages.create(
                        model=self.config.model,
                        max_tokens=max_tokens,
                        messages=messages,
                        tools = [
                            {
                                "name": "get_indra_statements",
                                "description": (
                                    "This tool is used to generate "
                                    "structured output from the input text."),
                                "input_schema": post_processed_schema
                            }    
                        ]
                    )
                else:
                    response = client.messages.create(
                        model=self.config.model,
                        max_tokens=max_tokens,
                        messages=messages
                    )
                    
                return response

            except Exception as e:
                if i < retry_count - 1:
                    logger.warning(
                        f"Request failed with error: {e}. "
                        "Retrying after 5 seconds."
                        f"Problematic messages: {json.dumps(messages, indent=2)}"
                    )
                    sleep(5)
                else:
                    raise e
                
        return response
    
    def get_response(
        self,
        input_text: str,
        chat_prompt: Dict[str, str],
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 8192,
        refinement_steps: bool = False
    ) -> str:
        """
        Gets a response from Anthropic's chat model, with optional 
        self-correction iterations.

        Returns:
            JSON-formatted response as a string.
        """
        self_correction_iterations = self.config.self_correction_iterations

        if chat_history is None:
            chat_history = []
        
        messages = chat_history + [chat_prompt]
        response = self.make_api_call(messages, max_tokens)
        response_content = self._extract_response_content(response)
        
        if self_correction_iterations == 0:
            return response_content
        else:
            response_content=self._self_correct(
                response_content= response_content,
                max_tokens= max_tokens,
                input_text= input_text,
                self_correction_iterations= self_correction_iterations,
            )
        
        if refinement_steps:
            try:
                response_content = self._refinement_steps(
                                            response_content,
                                            input_text,
                                            max_tokens
                                        )
            except Exception as e:
                logger.info(f"Error {e}. Skipping refinement steps.")

        return response_content
    
    def _extract_response_content(self, response):
        try:
            if response and response.content and len(response.content) > 0:
                text = ""
                for block in response.content:
                    if hasattr(block, '__class__') and block.__class__.__name__ == "ToolUseBlock":
                        tool_input = block.input
                        return json.dumps(tool_input)
                    elif hasattr(block, '__class__') and block.__class__.__name__ == "TextBlock":
                        text += block.text 
                return text if text else json.dumps({})  
            else:
                return json.dumps({}) 
        except Exception as e:
            logger.error(f"Malformed response: {response} with error: {e}")
            return json.dumps({})  
    
    def _self_correct(self,
                      response_content="{}",
                      max_tokens=8192,
                      input_text="",
                      self_correction_iterations=0):
        # Generic self-correction loop        
        for _ in range(self_correction_iterations):
            if not self.config.structured_output:
                content = (generic_refinement_prefix 
                            + "\n\nSchema:\n"
                            + json.dumps(JSON_SCHEMA, indent=2)
                            + "\n\nInput text:\n"
                            + input_text
                            + "\n\nResponse:\n"
                            + response_content)
            else:
                content = (generic_refinement_prefix 
                            + "\n\nInput text:\n" 
                            + input_text
                            + "\n\nResponse:\n"
                            + response_content)
            prompt = {
                "role": "user",
                "content": content
            }
            try:
                logger.info("Making API call for self-correction step.")
                raw_response = self.make_api_call([prompt], max_tokens)
            except Exception as e:
                logger.warning(f"API error encountered: {e}. Breaking self-correction loop.")
                break  # Stop refining if an API call fails

            logger.debug(f"Raw Response from Anthropic (Refinement Loop): {raw_response}")
            new_response = self._extract_response_content(raw_response)

            if new_response == response_content:
                break  # Stop refining if no changes occur
            response_content = new_response

        return response_content

    def _refinement_steps(self, 
                          response_content,
                          input_text, 
                          max_tokens):

        # Statement type refinement
        for prompt_prefix in [statement_type_correction_prefix, 
                              error_context_correction_prefix]:
            prompt = {
                "role": "user", 
                "content": (prompt_prefix 
                            + "\n\nInput text:\n" 
                            + input_text
                            + "\n\nResponse:\n"
                            + response_content)
            }
            if not self.config.structured_output:
                prompt['content'] += ("\n\nSchema:\n" +
                                    json.dumps(JSON_SCHEMA, indent=2))

            raw_response = self.make_api_call([prompt], max_tokens)
            logger.debug(f"Raw Response from OpenAI (Statement Type Fix): {raw_response}")
            response_content = self._extract_response_content(raw_response)

        return response_content
    
    def generate(self, preprocessed_data: Dict[str, Any]
                    ) -> List[Dict[str, Any]]:
            """
            Generates structured statements from preprocessed input data.

            Returns:
                List of extracted statements in JSON format.
            """
            input_texts = preprocessed_data["input_texts"]
            n_shot_history = preprocessed_data["n_shot_history"]
            flat_n_shot_history = [msg for pair in n_shot_history for msg in pair]
            extracted_statement_json_objects = []
            for input_text in tqdm(input_texts, desc="Extracting", unit="statement"):
                chat_prompt = self.get_chat_prompt(input_text)
                chat_history = flat_n_shot_history
                response_content = self.get_response(
                    input_text,
                    chat_prompt, 
                    chat_history, 
                    max_tokens=8192
                )
                try:
                    response_content = json.loads(response_content)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {response_content}")
                    response_content = {}
                extracted_statement_json_objects.append(response_content)
            return extracted_statement_json_objects
