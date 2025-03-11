import logging
from time import sleep
import json
from openai import OpenAI
from indra.config import IndraConfigError, get_config
from indra_gpt.resources.constants import (SCHEMA_STRUCTURED_OUTPUT_PATH, 
                                           USER_INIT_PROMPT_REDUCED,
                                           GENERIC_REFINEMENT_PROMPT,
                                           STATEMENT_TYPE_REFINEMENT_PROMPT,
                                           ERROR_CONTEXT_PROMPT)
from tqdm import tqdm
from indra_gpt.util.util import merge_allOf
from indra_gpt.configs import GenerationConfig
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    api_key = get_config("OPENAI_API_KEY", failure_ok=False)
    organization = get_config("OPENAI_ORG")
except IndraConfigError as err:
    raise KeyError(
        "Please set OPENAI_API_KEY in the environment or in the indra config."
    ) from err

client = OpenAI(api_key=api_key, organization=organization)


class OpenAIClient:
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
        Makes an API call to OpenAI with retry logic.

        Returns:
            Response object from OpenAI API.
        """
        retry_count = max(retry_count, 1)
        response = None

        for i in range(retry_count):
            try:
                if self.config.structured_output:
                    with open(SCHEMA_STRUCTURED_OUTPUT_PATH) as file:
                        schema = json.load(file)
                    post_processed_schema = merge_allOf(schema, schema)
                    response = client.beta.chat.completions.parse(
                        model=self.config.model,
                        messages=messages,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "indra_statement_json",
                                "strict": True,
                                "schema": post_processed_schema,
                            },
                        },
                    )
                else:
                    response = client.chat.completions.create(
                        model=self.config.model,
                        temperature=0,
                        max_tokens=max_tokens,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        messages=messages,
                        response_format={"type": "json_object"},
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
        chat_prompt: Dict[str, str],
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 9000,
    ) -> str:
        """
        Gets a response from OpenAI's chat model, with optional 
        self-correction iterations.

        Returns:
            JSON-formatted response as a string.
        """
        self_correction_iterations = self.config.self_correction_iterations

        # Load refinement prompts once (reduce redundant I/O)
        with open(GENERIC_REFINEMENT_PROMPT, "r", encoding="utf-8") as file:
            generic_refinement_text = file.read()
        with open(STATEMENT_TYPE_REFINEMENT_PROMPT, "r", encoding="utf-8") as file:
            statement_type_fix_text = file.read()
        with open(ERROR_CONTEXT_PROMPT, "r", encoding="utf-8") as file:
            error_context_fix_text = file.read()

        # Helper function to safely extract API response content
        def extract_response_content(response):
            try:
                return (response.choices[0].message.content 
                        if response and response.choices else "{}")
            except (IndexError, AttributeError, KeyError):
                logger.error(f"Malformed response: {response}")
                return "{}"  # Return empty JSON object as fallback

        if chat_history is None:
            chat_history = []
        
        messages = chat_history + [chat_prompt]
        response = self.make_api_call(messages, max_tokens)
        response_content = extract_response_content(response)

        if self_correction_iterations == 0:
            return response_content

        # Generic self-correction loop
        messages.append({"role": "assistant", "content": response_content})
        messages.append({"role": "user", "content": generic_refinement_text})
        for _ in range(self_correction_iterations):
            # Overwrite the assistant’s previous response (second to last message)
            messages[-2] = {"role": "assistant", "content": response_content}

            raw_response = self.make_api_call(messages, max_tokens)
            logger.debug(f"Raw Response from OpenAI (Refinement Loop): {raw_response}")

            new_response = extract_response_content(raw_response)

            if new_response == response_content:
                break
            response_content = new_response

        # Statement type refinement
        messages.append({"role": "assistant", "content": response_content})
        messages.append({"role": "user", "content": statement_type_fix_text})

        raw_response = self.make_api_call(messages, max_tokens)
        logger.debug(f"Raw Response from OpenAI (Statement Type Fix): {raw_response}")
        response_content = extract_response_content(raw_response)

        # Error Context Refinement
        messages.append({"role": "assistant", "content": response_content})
        messages.append({"role": "user", "content": error_context_fix_text})

        raw_response = self.make_api_call(messages, max_tokens)
        logger.debug(f"Raw Response from OpenAI (Error Context Fix): {raw_response}")
        response_content = extract_response_content(raw_response)

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
                chat_prompt, chat_history, max_tokens=9000
            )
            response_content = json.loads(response_content)
            extracted_statement_json_objects.append(response_content)
        return extracted_statement_json_objects
