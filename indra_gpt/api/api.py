import logging
from indra_gpt.clients.openai.openai_client import OpenAIClient
from indra_gpt.clients.anthropic.anthropic_client import AnthropicClient

# Set up logging
logger = logging.getLogger(__name__)

# Main function to process data using a client
def generate_statements_with_client(**kwargs):
    if kwargs.get("iterations") < 5:
        raise ValueError("Number of iterations must be at least 5.")
    
    if kwargs.get("model") == 'gpt-4o-mini':
        client = OpenAIClient(**kwargs)
    elif kwargs.get("model") == 'claude-3-5-sonnet-latest':
        client = AnthropicClient(**kwargs)
    else:
        raise ValueError("Enter valid model, currently supported models are: 'gpt-4o-mini', 'claude-3-5-sonnet-latest'.")
    
    try:
        original_statement_json_objects = client.get_input_json_objects()
        generated_statement_json_objects = client.generate_statement_json_objects(original_statement_json_objects)
        if generated_statement_json_objects:
            results_df = client.get_results_df(original_statement_json_objects, generated_statement_json_objects)
            client.save_results_df(results_df)
    except Exception as e:
        logger.error(f"Error processing data with client: {e}")
        raise e
    
    return original_statement_json_objects, generated_statement_json_objects
