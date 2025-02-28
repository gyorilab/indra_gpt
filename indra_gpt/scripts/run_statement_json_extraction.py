import logging
from pathlib import Path
import argparse

from indra_gpt.resources.constants import OUTPUT_DEFAULT
from indra_gpt.api.api import generate_statements
from indra_gpt.config import ConfigManager
from indra_gpt.util.util import save_results


logger = logging.getLogger(__name__)

def main(**kwargs):
    # Initialize configuration objects
    preprocessing_config = PreProcessingConfig({
        "user_inputs_file": kwargs["user_inputs_file"], 
        "num_samples": kwargs["num_samples"],
        "random_sample": kwargs["random_sample"],
        "n_shot_prompting": kwargs["n_shot_prompting"],
        "user_input_refinement_strat": kwargs["user_input_refinement_strat"]
    })

    generation_config = GenerationConfig({
        "model": kwargs["model"],
        "structured_output": kwargs["structured_output"],
        "feedback_refinement_iterations": kwargs["feedback_refinement_iterations"],
        "batch_job": kwargs["batch_job"],
        "batch_id": kwargs["batch_id"], 
    })

    postprocessing_config = PostProcessingConfig({
        "grounding": kwargs["grounding"]
    })

    # Instantiate processing objects
    pre_processor = PreProcessor(preprocessing_config)
    post_processor = PostProcessor(postprocessing_config)

    # Generate responses using preprocessing and generation settings
    generated_responses = generate_responses(generation_config, pre_processor)

    # Save results with the correct configurations
    save_results(kwargs["output_file"], generated_responses, post_processor)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Script for running structured knowledge extraction.")
    ##### arguments for the input and output files and job mode #####
    arg_parser.add_argument(
        "--user_inputs_file",
        type=str,
        default=None,
        help="Path to the file containing user inputs for inference. Format can be either: "
            "(1) a txt file where each line is an input text, or "
            "(2) a JSON file with a list of input texts in [{'text': 'input text', 'pmid': 'pmid'}] format. "
            "If not provided, input texts will be extracted from the benchmark corpus."
    )
    arg_parser.add_argument(
        "--num_samples", "-n", 
        type=int, 
        default=50,
        help="Number of input texts to process from the input file for inference. "
            "By default, the first N texts are used unless --random_sample is specified."
    )
    arg_parser.add_argument(
        "--random_sample", 
        action="store_true", 
        help="Randomly sample input text from input file if set."
    )
    arg_parser.add_argument(
        "--output_file", 
        type=str, 
        default=OUTPUT_DEFAULT.as_posix(),
        help=f"Path to save the output TSV file. Default: {OUTPUT_DEFAULT.as_posix()}."
    )
    arg_parser.add_argument(
        "--batch_job", "-b",
        action="store_true",
        help="Enable batch job mode to process requests asynchronously to the OpenAI API."
    )
    arg_parser.add_argument(
        "--batch_id", 
        type=str, 
        default=None,
        help="Provide a string for the batch job ID to retrieve batch results."
    )
    ##### arguments for the model and generation strategy settings #####
    arg_parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt-4o-mini",
        help="Specify the model name. Default: 'gpt-4o-mini'."
    )
    arg_parser.add_argument(
        "--structured_output", 
        action="store_true", 
        help="Force the output to strictly follow a structured schema. (Only supported for OpenAI API currently.)"
    )
    arg_parser.add_argument(
        "--n_shot_prompting", 
        type=int,  
        default=0, 
        help="Number of example input-output pairs to include for few-shot prompting. "
         "A value of 0 results in zero-shot prompting."
    )
    arg_parser.add_argument(
        "--user_input_refinement_strat", 
        type=str, 
        default="default", 
        help="User input preprocessing strategy: 'default' or 'summarize'."
    )
    arg_parser.add_argument(
        "--feedback_refinement_iterations", 
        type=int, 
        default=0, 
        help="Number of iterations for feedback-based refinement strategy."
    )
    arg_parser.add_argument(
        "--grounding",
        action="store_true",
        help="Enable grounding of extracted named entities."
    )

    cli_args = arg_parser.parse_args()
    args_dict = vars(cli_args)
    main(**args_dict)
