from abc import ABC, abstractmethod
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define an abstract base class for clients
class ClientInterface(ABC):

    @abstractmethod
    def get_response(self):
        pass

    @abstractmethod
    def save_output(self, response):
        pass

# Main function to process data using a client
def process_data_with_client(client: ClientInterface, **kwargs):
    if kwargs.get("model") == 'gpt-4o-mini':
        from indra_gpt.clients.openai.openai_client import Openai_client
        client = Openai_client(**kwargs)
    elif kwargs.get("model") == 'claude-3-5-sonnet-latest':
        from indra_gpt.clients.anthropic.anthropic_client import Anthropic_client
        client = Anthropic_client(**kwargs)
    else:
        raise ValueError("Enter valid model, currently supported models are: 'gpt-4o-mini', 'claude-3-5-sonnet-latest'.")
    
    try:
        response = client.get_response()
        client.save_output(response)
        return response
    except Exception as e:
        logger.error(f"Error processing data with client: {e}")
        raise