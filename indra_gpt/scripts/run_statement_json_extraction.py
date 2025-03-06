from typing import Any, Mapping
import logging
import argparse
from indra_gpt.resources.constants import OUTPUT_DEFAULT
from indra_gpt.configs import BaseConfig, PreProcessorConfig, GenerationConfig, PostProcessorConfig
from indra_gpt.processors import PreProcessor, Generator, PostProcessor
from indra_gpt.pipelines import StatementExtractionPipeline

logger = logging.getLogger(__name__)

def main(**kwargs: Mapping[str, Any]) -> None:
    base_config = BaseConfig(kwargs)
    preprocessing_config = PreProcessorConfig(base_config)
    generation_config = GenerationConfig(base_config)
    postprocessing_config = PostProcessorConfig(base_config)

    # Instantiate processing objects
    pre_processor = PreProcessor(preprocessing_config)
    generator = Generator(generation_config)
    post_processor = PostProcessor(postprocessing_config)

    # pipe the processing objects
    pipeline = StatementExtractionPipeline(pre_processor, generator, post_processor)

    # Run the pipeline and save the results
    output_file = kwargs.get("output_file", OUTPUT_DEFAULT.as_posix())
    logger.info("Running structured knowledge extraction pipeline...")
    pipeline.run_and_save_results(output_file)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Script for running structured knowledge extraction.")
    ##### arguments for the input and output files and job mode #####
    arg_parser.add_argument(
        "--user_inputs_file",
        type=str,
        default=None,
        help="Path to the file containing user inputs for inference. Format can be either: "
            "(1) a tsv with two columns 'text' and 'pmid', or "
            "(2) a JSON file with a list of input texts in [{'text': 'input text', 'pmid': 'pmid'} ...] format. "
            "If not provided, input texts will be extracted from the benchmark corpus."
    )
    arg_parser.add_argument(
        "--num_samples", "-n", 
        type=int, 
        default=5,
        help="Number of input texts to process from the input file for inference. "
            "By default, the first N texts are used unless --random_sample is specified."
    )
    arg_parser.add_argument(
        "--random_sample", 
        action="store_true", 
        help="Randomly sample input text from input file if set."
    )
    arg_parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help=("Random seed for sampling input texts. "
               "Default: 42.")
    )
    arg_parser.add_argument(
        "--output_file", 
        type=str, 
        default=OUTPUT_DEFAULT.as_posix(),
        help=f"Path to save the output pkl file. Default: {OUTPUT_DEFAULT.as_posix()}."
    )
    ##### arguments for the model and generation strategy settings #####
    arg_parser.add_argument(
        "--model", 
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
        "--self_correction_iterations", 
        type=int, 
        default=0, 
        help="Number of iterations for self-correction (dialogue) strategy."
    )
    arg_parser.add_argument(
        "--grounding",
        action="store_true",
        help="Enable grounding of extracted named entities."
    )

    cli_args = arg_parser.parse_args()
    args_dict = vars(cli_args)
    main(**args_dict)
