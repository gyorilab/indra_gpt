import os
import time
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from indra_gpt.clients.llm_client import LLMClient
from indra_gpt.preprocessors.preprocessor import Preprocessor
from indra_gpt.postprocess.postprocessor import PostProcessor
from indra_gpt.postprocess.json_to_indra import JSONToINDRAConverter
from indra_gpt.postprocess.preassemble_indra import INDRAPreassembler
from indra_gpt.extractors.end_to_end.end_to_end_extractor import EndToEndExtractor


def load_inputs(input_path, max_examples=10):
    with open(input_path) as f:
        data = json.load(f)
    return [(ev.get('text', ""), ev.get('pmid', '')) for stmt in data for ev in stmt['evidence']][:max_examples]


def benchmark_method(method_name, process_fn, inputs, logfile=None):
    print(f"\n--- Running: {method_name} ---")
    start_time = time.perf_counter()

    texts, pmids = zip(*inputs)
    all_results = process_fn(texts, pmids)

    elapsed_time = time.perf_counter() - start_time
    secs_per_example = elapsed_time / len(inputs) if inputs else 0

    print(f"{method_name} finished")
    print(f"⏱️  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"⚡ Processing speed: {secs_per_example:.2f} secs/example")
    print(f"📈 Total INDRA statements: {len(all_results)}")

    if logfile:
        with open(logfile, "a") as f:
            f.write(
                f"Method: {method_name}\n"
                f"  Number of example texts processed: {len(inputs)}\n"
                f"  Time: {elapsed_time:.2f}s \n"
                f"  Speed: {secs_per_example:.2f} secs/example \n"
                f"  Num Statements: {len(all_results)}\n"
            )

    return all_results


def main(input_path, client, model, logfile=None, max_examples=10):
    model_config = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "ollama": {
            "api_key": None,
            "api_base": "http://localhost:11434"
        }
    }

    config = model_config[client]
    llm_client = LLMClient(
        custom_llm_provider=client,
        api_key=config.get("api_key"),
        model=model,
        api_base=config.get("api_base")
    )

    # Pipeline setup
    rec_logic_preprocessor = Preprocessor(llm_client, preprocessing_method="rec_logic_parser")
    rec_relation_preprocessor = Preprocessor(llm_client, preprocessing_method="rec_relation_parser")
    extractor = EndToEndExtractor(llm_client=llm_client, num_history_examples=2)
    json_to_indra_converter = JSONToINDRAConverter()
    preassembler = INDRAPreassembler()
    postprocessor = PostProcessor(json_to_indra_converter, preassembler)

    def run_end_to_end(texts, pmids):
        all_stmts = []
        for text, pmid in tqdm(zip(texts, pmids), desc="Processing texts", total=len(texts)):
            raw = None
            try:
                raw = extractor.raw_extract(text)
                stmts = postprocessor.convert_to_indra_stmts(raw, text, pmid)
                all_stmts.extend(stmts)
            except Exception as e:
                print(f"Error processing text '{text}' with PMID '{pmid}':\n")
                print(f"{e}\n")
                print(f"Raw extraction: {raw}\n")
                all_stmts.extend([])

        # try:
        #     print(f"Statments before preassembly: {all_stmts}\n")
        #     preassembled_stmts = postprocessor.preassemble(all_stmts)
        # except Exception as e:
        #     print(f"Error during preassembly: {e}\n")
        #     preassembled_stmts = []
        # return preassembled_stmts
        return all_stmts

    def run_hybrid(texts, pmids):
        all_stmts = []
        for text, pmid in tqdm(zip(texts, pmids), desc="Processing texts", total=len(texts)):
            raw = None
            try:
                logic = rec_logic_preprocessor.preprocess(text)
                raw = extractor.raw_extract(logic)
                stmts = postprocessor.convert_to_indra_stmts(raw, text, pmid)
                all_stmts.extend(stmts)
            except Exception as e:
                print(f"Error processing text '{text}' with PMID '{pmid}':\n")
                print(f"{e}\n")
                print(f"Raw extraction: {raw}\n")
                all_stmts.extend([])

        return all_stmts

    def run_recursive(texts, pmids):
        all_stmts = []
        for text, pmid in tqdm(zip(texts, pmids), desc="Processing texts", total=len(texts)):
            raw = None
            try:
                logic = rec_logic_preprocessor.preprocess(text)
                relation = rec_relation_preprocessor.preprocess(logic)
                raw = extractor.raw_extract(relation)
                stmts = postprocessor.convert_to_indra_stmts(raw, text, pmid)
                all_stmts.extend(stmts)
            except Exception as e:
                print(f"Error processing text '{text}' with PMID '{pmid}':\n")
                print(f"{e}\n")
                print(f"Raw extraction: {raw}\n")
                all_stmts.extend([])

        return all_stmts

    # Load and process
    inputs = load_inputs(input_path, max_examples=max_examples)
    # if logfile exists, delete it first and then create a new one
    if logfile and Path(logfile).exists():
        os.remove(logfile)
    # Run tests
    benchmark_method("End-to-End", run_end_to_end, inputs, logfile)
    benchmark_method("Hybrid", run_hybrid, inputs, logfile)
    benchmark_method("Recursive", run_recursive, inputs, logfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run INDRA-GPT extraction pipeline benchmarks.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--client", type=str, choices=["openai", "ollama"], required=True, help="Model provider to use")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--max_examples", type=int, default=10, help="Maximum number of examples to process")
    parser.add_argument("--logfile", type=str, default=None, help="Optional path to save benchmark timing log")
    args = parser.parse_args()

    main(input_path=args.input, client=args.client, model=args.model, logfile=args.logfile, max_examples=args.max_examples)
