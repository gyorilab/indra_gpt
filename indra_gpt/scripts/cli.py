"""
Run the indra gpt cli using ``python -m indra_gpt.cli create-training-data``
to create the initial data (requires a curations file and cached statements)
or ``python -m indra_gpt.cli run-stats-cli`` to run the main accuracy test (
requires an API key to openai).
"""

import click

from indra_gpt.util.check_correctness import get_create_training_set
from indra_gpt.util.check_correctness import run_stats as run_stats_func


@click.group("indra-gpt")
def main():
    """Run the indra-gpt command line tool."""


@main.command()
@click.option(
    "--curations-file",
    type=str,
    required=True,
    help="The path to the curations file to use. This should be a json file "
    "containing a list of curation dictionaries.",
)
@click.option(
    "--statements-file",
    type=str,
    required=True,
    help="The path to the statements json file to use.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force the recreation of the training data frame even if it already "
    "exists.",
)
def create_training_data(curations_file: str, statements_file: str, force: bool):
    """Create training data"""
    _ = get_create_training_set(
        refresh=force,
        curations_file=curations_file,
        statement_json_file=statements_file,
    )


@main.command()
@click.option(
    "--run-iter",
    type=int,
    default=100,
    help="The number of queries to send to chat-gpt. Default: 100.",
)
@click.option(
    "--pos-examples",
    type=int,
    default=2,
    help="The number of positive examples to use in the prompt. Default: 2.",
)
@click.option(
    "--neg-examples",
    type=int,
    default=2,
    help="The number of negative examples to use in the prompt. Default: 2.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=2,
    help="The maximum number of tokens to use for the responses. Default: 2.",
)
@click.option(
    "--debug-print",
    is_flag=True,
    help="Print the prompt sent and the full response from chat-gpt API call.",
)
@click.option(
    "--file-title", type=str, default="", help="The title to use for the output file."
)
def run_stats(
    run_iter: int,
    pos_examples: int,
    neg_examples: int,
    max_tokens: int,
    debug_print: bool,
    file_title: str,
):
    """Run the stats command line tool."""
    df = get_create_training_set(refresh=False)

    _ = run_stats_func(
        training_data_df=df,
        n_iter=run_iter,
        n_pos_examples=pos_examples,
        n_neg_examples=neg_examples,
        max_tokens=max_tokens,
        debug_print=debug_print,
        file_title=file_title or None,
    )


if __name__ == "__main__":
    main()
