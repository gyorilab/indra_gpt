from abc import ABC, abstractmethod

# Define an abstract base class for clients
class ClientInterface(ABC):

    @abstractmethod
    def generate_statement_json_objects(self, original_statement_json_objects):
        pass

    @abstractmethod
    def get_results_df(self, original_statement_json_objects, generated_statement_json_objects):
        pass

    @abstractmethod
    def save_results_df(self, results_df):
        pass
