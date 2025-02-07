import logging
from pathlib import Path

from indra_gpt.resources.constants import OUTPUT_DEFAULT, INPUT_DEFAULT
from indra_gpt.api.api import process_data_with_client


logger = logging.getLogger(__name__)

# Define the main function
def main(statements_file, model, n_iter, output_file, verbose, batch_jobs, batch_id):
    if args.iterations < 5:
        raise ValueError("Number of iterations must be at least 5.")
    logger.info(f"Using model: {args.model_name}")

    kwargs = {
        "statements_file": statements_file,
        "model": model,
        "n_iter": n_iter,
        "output_file": output_file,
        "verbose": verbose,
        "batch_jobs": batch_jobs,
        "batch_id": batch_id
    }
    process_data_with_client(**kwargs)


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--statements_file",
        type=str,
        default=INPUT_DEFAULT.absolute().as_posix(),
        help=f"Path to the json file containing statement json objects. Default is "
             f"{INPUT_DEFAULT.as_posix()}.",
    )
    arg_parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                            help="Provide the name of the model to use, e.g., 'gpt-4o-mini' or 'claude-3-5-sonnet-latest'.")
    arg_parser.add_argument("--iterations", "-n", type=int, default=50,
                            help="Number of iterations to run")
    arg_parser.add_argument(
        "--output_file", type=str, default=OUTPUT_DEFAULT.as_posix(),
        help=f"Path to save the output tsv file. Default is {OUTPUT_DEFAULT.as_posix()}."
    )
    arg_parser.add_argument("-v", "--verbose", action="store_true",
                            help="Increase output verbosity. Will print requests sent "
                                 "to and responses received from the API, respectively.")
    arg_parser.add_argument("-b", "--batch_jobs", action="store_true",
                            help="If set, the script will run in batch job mode, "
                                 "processing groups of requests asynchronously to "
                                 "OpenAI API.")
    arg_parser.add_argument("--batch_id", type=str, default=None,
                            help="Provide a tring of the batch job ID to retrieve the "
                                 "results of a batch job.")
    args = arg_parser.parse_args()

    main(
        statements_file=args.statements_file,
        model=args.model_name,
        n_iter=args.iterations,
        output_file=Path(args.output_file),
        verbose=args.verbose,
        batch_jobs=args.batch_jobs,
        batch_id=args.batch_id
    )
