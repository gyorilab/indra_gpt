import argparse
import pickle
import logging
from pathlib import Path
import pandas as pd
from indra.sources.indra_db_rest import get_curations, get_statements_by_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_extracted_statements(file_path: Path):
    """Load extracted statements from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            extracted_stmts = pickle.load(f)
        return [stmt for stmt in extracted_stmts if stmt is not None]
    except Exception as e:
        logger.error(f"Error loading extracted statements: {e}")
        return []

def get_statement_hashes(statements):
    """Get hashes of extracted statements."""
    return [stmt.get_hash() for stmt in statements]

def get_source_hashes(statements):
    """Get sets of source hashes for each statement."""
    return [{ev.source_hash for ev in stmt.evidence} for stmt in statements]

def fetch_curations(pa_hashes):
    """Fetch curations for each statement."""
    curations = []
    for pa_hash in pa_hashes:
        try:
            curations.append(get_curations(hash_val=pa_hash))
        except Exception as e:
            logger.warning(f"Error fetching curations for {pa_hash}: {e}")
            curations.append([])
    return curations

def get_curated_source_hashes(curations):
    """Extract source hashes from curations."""
    return [
        {curation['source_hash'] for curation in stmt_curations}
        for stmt_curations in curations
    ]

def fetch_all_evidences(pa_hashes):
    """Retrieve all evidences from the database for each statement hash."""
    evidences = []
    for pa_hash in pa_hashes:
        try:
            result = get_statements_by_hash([pa_hash], ev_limit=500)
            evidences.append(result.statements[0].evidence if result and result.statements else [])
        except Exception as e:
            logger.warning(f"Error fetching statement for {pa_hash}: {e}")
            evidences.append([])
    return evidences

def get_same_text_evidence_sources(statements, db_evidences):
    """Find all evidence source hashes that have the same text as any extracted statement."""
    return [
        {
            db_ev.source_hash for db_ev in db_evidence_list
            if stmt.evidence and db_ev.text in {ev.text for ev in stmt.evidence}
        } | stmt_source_hashes
        for stmt, db_evidence_list, stmt_source_hashes in zip(statements, db_evidences, get_source_hashes(statements))
    ]

def get_intersecting_curated_evidences(curated_hashes, text_based_hashes):
    """Find the intersection of curated and text-matching evidence source hashes."""
    return [
        curated.intersection(text_based) if curated and text_based else set()
        for curated, text_based in zip(curated_hashes, text_based_hashes)
    ]

def extract_curated_evidence(curations, intersecting_hashes):
    """Extract the curated evidence for the final filtered evidence source hashes."""
    return [
        {
            source_hash: next(
                (curation for curation in stmt_curations if curation['source_hash'] == source_hash),
                None
            )
            for source_hash in intersecting_hashes
        }
        for stmt_curations, intersecting_hashes in zip(curations, intersecting_hashes)
    ]

def compute_accuracy(df):
    """Compute accuracy based on correct and incorrect curations."""
    df['num_correct_curation_tags'] = df['extracted_stmt_same_text_and_curated_evidence'].apply(
        lambda x: sum(1 for curation in x.values() if curation and curation.get('tag') == 'correct')
    )
    df['num_incorrect_curation_tags'] = df['extracted_stmt_same_text_and_curated_evidence'].apply(
        lambda x: sum(1 for curation in x.values() if curation and curation.get('tag') != 'correct')
    )
    accuracy = (df['num_correct_curation_tags'] > 0).sum() / len(df)
    logger.info(f"Accuracy: {accuracy:.2%}")
    return df

def save_results(df, output_path):
    """Save the results to a pickle file."""
    try:
        with open(output_path, "wb") as f:
            pickle.dump(df, f)
        logger.info(f"Results successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    """Main function to process and analyze extracted statements."""
    parser = argparse.ArgumentParser(description="Process extracted statements and compute curation accuracy.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the extracted statements pickle file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the processed results.")

    args = parser.parse_args()
    input_file_path = Path(args.input_file)
    output_folder_path = Path(args.output_folder)

    if not input_file_path.exists():
        logger.error(f"Input file {input_file_path} not found.")
        return

    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Load extracted statements
    extracted_stmts = load_extracted_statements(input_file_path)
    pa_hashes = get_statement_hashes(extracted_stmts)

    # Fetch curations and evidences
    curations = fetch_curations(pa_hashes)
    curated_source_hashes = get_curated_source_hashes(curations)
    all_evidences = fetch_all_evidences(pa_hashes)
    text_evidence_sources = get_same_text_evidence_sources(extracted_stmts, all_evidences)

    # Find intersection of curated and text-matching evidence
    intersecting_curated_hashes = get_intersecting_curated_evidences(curated_source_hashes, text_evidence_sources)
    curated_evidences = extract_curated_evidence(curations, intersecting_curated_hashes)

    # Create DataFrame
    result = {
        "extracted_stmt": extracted_stmts,
        "pa_hash": pa_hashes,
        "source_hashes_set": get_source_hashes(extracted_stmts),
        "extracted_stmt_curations": curations,
        "extracted_stmt_curations_source_hashes": curated_source_hashes,
        "stmt_all_evidences_from_db": all_evidences,
        "extracted_stmt_same_text_evidence_sources_hashes": text_evidence_sources,
        "extracted_stmt_same_text_and_curated_evidence_source_hashes": intersecting_curated_hashes,
        "extracted_stmt_same_text_and_curated_evidence": curated_evidences
    }

    df = pd.DataFrame(result)
    df = compute_accuracy(df)

    # Save results
    curation_file_name = f"curation_result_{input_file_path.stem}.pkl"
    curation_file_path = output_folder_path / curation_file_name
    save_results(df, curation_file_path)

if __name__ == "__main__":
    main()
