import logging
import pickle
from pathlib import Path
from typing import Any, Tuple, Dict, List
from indra_gpt.processors import PreProcessor, Generator, PostProcessor
from indra.statements import Statement
from datetime import datetime


logger = logging.getLogger(__name__)


class StatementExtractionPipeline:
    def __init__(self, 
                 pre_processor: PreProcessor, 
                 generator: Generator, 
                 post_processor: PostProcessor) -> None:
        """
        Initializes the pipeline with preprocessing, generation, 
        and postprocessing components.
        """
        self.pre_processor = pre_processor
        self.generator = generator
        self.post_processor = post_processor

    def run(self) -> Tuple[List[Dict[str, str]], 
                           Dict[str, Any], 
                           List[Dict[str, Any]], 
                           List[Statement]]:
        """
        Executes the entire extraction pipeline: preprocessing, generation,
        and postprocessing.
        Returns the processed results.
        """
        logger.info("Starting preprocessing step...")
        raw_input_data, preprocessed_input_data = self.pre_processor.process()

        logger.info("Running LLM-based statement extraction...")
        extracted_json_stmts = self.generator.generate(preprocessed_input_data)

        logger.info("Post-processing extracted statements...")
        preassembled_stmts = self.post_processor.process(extracted_json_stmts)

        return (raw_input_data, 
                preprocessed_input_data, 
                extracted_json_stmts, 
                preassembled_stmts)

    def save_results(self, 
                     output_folder: str, 
                     raw_input_data: List[Dict[str, str]], 
                     preprocessed_input_data: Dict[str, Any],
                     extracted_json_stmts: List[Dict[str, Any]], 
                     preassembled_stmts: List[Statement]) -> None:
        """
        Saves the pipeline results to a file.
        """
        try:
            # Ensure the output directory exists
            output_path = Path(output_folder) / f"extraction_results_{datetime.now().strftime('%Y-%m-%d-%H')}"
            output_path.mkdir(parents=True, exist_ok=True)

            # Structure the results
            results_dict: Dict[str, Any] = {
                "raw_input_data": raw_input_data,
                "preprocessed_input_data": preprocessed_input_data,
                "extracted_json_stmts": extracted_json_stmts,
                "preassembled_stmts": preassembled_stmts
            }

            # Save the results to a pickle file
            detailed_results_output_path = Path(output_path) / f"detailed_extraction_results_{datetime.now().strftime('%Y-%m-%d-%H')}.pkl"
            with open(detailed_results_output_path, "wb") as f:
                pickle.dump(results_dict, f)
            logger.info(f"Detailed results successfully saved to {detailed_results_output_path}")

            # Save just the flattened list of statements to a pickle file.
            stmts_output_path = Path(output_path) / f"extracted_statements_{datetime.now().strftime('%Y-%m-%d-%H')}.pkl"
            with open(stmts_output_path, "wb") as f:
                # flatten the list of statements
                stmts_flat_list = []
                for stmts in preassembled_stmts:
                    stmts_flat_list.extend(stmts)
                pickle.dump(stmts_flat_list, f)
            logger.info(f"Flattened statements successfully saved to {stmts_output_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise RuntimeError("Failed to save results.") from e

    def run_and_save_results(self, output_folder: str) -> None:
        """
        Runs the full pipeline and saves the extracted statements to a file.
        """
        (raw_input_data, preprocessed_input_data, 
         extracted_json_stmts, preassembled_stmts) = self.run()

        logger.info(f"Saving results to {output_folder}...")
        self.save_results(output_folder, raw_input_data, 
                          preprocessed_input_data, extracted_json_stmts, 
                          preassembled_stmts)
        logger.info("Pipeline execution completed successfully.")
