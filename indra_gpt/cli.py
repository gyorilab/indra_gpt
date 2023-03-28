"""
Run the indra gpt cli using ``python -m indra_gpt.cli create-training-data``
to create the initial data (requires a curations file and cached statements)
or ``python -m indra_gpt.cli run-stats-cli`` to run the main accuracy test (
requires an API key to openai).
"""

import click

from indra_gpt.check_correctness import get_create_training_set, run_stats


@click.group("indra-gpt")
def main():
    """Run the indra-gpt command line tool."""


@main.command()
@click.option(
    "--curations-file",
    type=str,
    required=True,
    help="The path to the curations file to use. This should be a json file "
         "containing a list of curation dictionaries."
)
@click.option(
    "--statements-file",
    type=str,
    required=True,
    help="The path to the statements json file to use."
)
@click.option(
    "--force",
    is_flag=True,
    help="Force the recreation of the training data frame even if it already "
         "exists."
)
def create_training_data(curations_file: str, statements_file: str, force: bool):
    """Create training data for the GPT-2 model."""
    _ = get_create_training_set(refresh=force, curations_file=curations_file,
                                statement_json_file=statements_file)


@main.command()
@click.option(
    "--run-iter",
    type=int,
    default=100,
    help="The number of queries to send to chat-gpt."
)
@click.option(
    "--pos-examples",
    type=int,
    default=2,
    help="The number of positive examples to use in the prompt."
)
@click.option(
    "--neg-examples",
    type=int,
    default=2,
    help="The number of negative examples to use in the prompt."
)
@click.option(
    "--debug-print",
    is_flag=True,
    help="Print the prompt sent and the full response from chat-gpt API call."
)
def run_stats_cli(
    run_iter: int,
    pos_examples: int,
    neg_examples: int,
    debug_print: bool
):
    """Run the stats command line tool."""
    df = get_create_training_set(refresh=False)

    _ = run_stats(
        training_data_df=df,
        n_iter=run_iter,
        n_pos_examples=pos_examples,
        n_neg_examples=neg_examples,
        debug_print=debug_print
    )


if __name__ == "__main__":
    main()
