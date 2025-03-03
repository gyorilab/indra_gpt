import logging
import pickle
from pathlib import Path

class StatementExtractionPipeline:
    def __init__(self, pre_processor, generator, post_processor):
        """
        Initializes the pipeline with preprocessing, generation, and postprocessing components.
        """
        self.pre_processor = pre_processor
        self.generator = generator
        self.post_processor = post_processor
        self.logger = logging.getLogger(__name__)

    def run(self):
        """
        Executes the entire extraction pipeline: preprocessing → generation → postprocessing.
        Returns the processed results.
        """
        self.logger.info("Starting preprocessing step...")
        preprocessed_data = self.pre_processor.process()

        self.logger.info("Running LLM-based statement extraction...")
        generated_results = self.generator.generate(preprocessed_data)

        self.logger.info("Post-processing extracted statements...")
        processed_results = self.post_processor.process(generated_results)

        return preprocessed_data, generated_results, processed_results

    def save_results(self, output_file, preprocessed_data, generated_results, processed_results):
        """
        Saves the overall results, including the input, generated output, and the processed output,
        into a pickle file at the specified output path.

        Parameters:
        - output_file (str or Path): Path to the output file.
        - preprocessed_data (list): Preprocessed input data.
        - generated_results (list): Raw generated responses from the model.
        - processed_results (list): Post-processed INDRA statements.
        """
        try:
            # Ensure the output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Structure the results
            results_dict = {
                "preprocessed_data": preprocessed_data,
                "generated_results": generated_results,
                "processed_results": processed_results
            }

            # Save the results to a pickle file
            with open(output_path, "wb") as f:
                pickle.dump(results_dict, f)

            self.logger.info(f"Results successfully saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise RuntimeError("Failed to save results.") from e

    def run_and_save_results(self, output_file):
        """
        Runs the full pipeline and saves the extracted statements to a file.
        """
        preprocessed_data, generated_results, processed_results = self.run()

        self.logger.info(f"Saving results to {output_file}...")
        self.save_results(output_file, preprocessed_data, generated_results, processed_results)
        self.logger.info("Pipeline execution completed successfully.")
