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
        raw_input_data, preprocessed_input_data = self.pre_processor.process()

        self.logger.info("Running LLM-based statement extraction...")
        extracted_json_stmts = self.generator.generate(preprocessed_input_data)

        self.logger.info("Post-processing extracted statements...")
        preassembled_stmts = self.post_processor.process(extracted_json_stmts)

        return raw_input_data, preprocessed_input_data, extracted_json_stmts, preassembled_stmts

    def save_results(self, output_file, raw_input_data, preprocessed_input_data, extracted_json_stmts, preassembled_stmts):
        try:
            # Ensure the output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Structure the results
            results_dict = {
                "raw_input_data": raw_input_data,
                "preprocessed_input_data": preprocessed_input_data,
                "extracted_json_stmts": extracted_json_stmts,
                "preassembled_stmts": preassembled_stmts
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
        raw_input_data, preprocessed_input_data, extracted_json_stmts, preassembled_stmts = self.run()

        self.logger.info(f"Saving results to {output_file}...")
        self.save_results(output_file, raw_input_data, preprocessed_input_data, extracted_json_stmts, preassembled_stmts)
        self.logger.info("Pipeline execution completed successfully.")
