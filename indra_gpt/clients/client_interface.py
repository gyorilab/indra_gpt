from abc import ABC, abstractmethod

# Define an abstract base class for clients
class ClientInterface(ABC):

    @abstractmethod
    def generate_statement_json_objects(self):
        pass

    @abstractmethod
    def get_results_df(self, generated_statement_json_objects):
        pass

    @abstractmethod
    def save_results_df(self, results_df):
        pass