import sys
import os
import time
import tracemalloc
import json
from pathlib import Path
from tqdm import tqdm
from indra_gpt.clients.llm_client import LLMClient
from indra_gpt.preprocessors.preprocessor import Preprocessor
from indra_gpt.postprocess.postprocessor import PostProcessor
from indra_gpt.postprocess.json_to_indra import JSONToINDRAConverter
from indra_gpt.postprocess.preassemble_indra import INDRAPreassembler
from indra_gpt.extractors.end_to_end.end_to_end_extractor import EndToEndExtractor


# Load data
data_path = Path(__file__).parent / "indra_gpt" / "resources" / "indra_benchmark_corpus_all_correct.json"
data = json.load(open(data_path))
inputs = [(ev.get('text', ""), ev.get('pmid', '')) for stmt in data for ev in stmt['evidence']]

# Model config
model_config = {
    "ollama": {
        "api_key": None,
        "model": "llama3.2:latest",
        "api_base": "http://localhost:11434"
    }
}

api_provider = "ollama"
llm_client = LLMClient(
    custom_llm_provider=api_provider,
    api_key=model_config[api_provider]["api_key"],
    api_base=model_config[api_provider]["api_base"],
    model=model_config[api_provider]["model"]
)

# Pipeline components
rec_logic_preprocessor = Preprocessor(llm_client, preprocessing_method="rec_logic_parser")
rec_relation_preprocessor = Preprocessor(llm_client, preprocessing_method="rec_relation_parser")
extractor = EndToEndExtractor(llm_client=llm_client, num_history_examples=2)
json_to_indra_converter = JSONToINDRAConverter()
preassembler = INDRAPreassembler()
postprocessor = PostProcessor(json_to_indra_converter, preassembler)

def benchmark_method(method_name, process_fn):
    print(f"\n--- Running: {method_name} ---")
    start_time = time.perf_counter()
    tracemalloc.start()

    all_results = []
    for text, pmid in tqdm(inputs, desc=method_name):
        try:
            stmts = process_fn(text, pmid)
            all_results.extend(stmts)
        except Exception as e:
            print(f"[{method_name}] Error for PMID {pmid}: {e}")

    elapsed_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"{method_name} finished")
    print(f"⏱️  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"📦 Peak memory: {peak / 1024**2:.2f} MB")
    print(f"📈 Total INDRA statements: {len(all_results)}")
    return all_results

# Define the three pipeline variations
def run_end_to_end(text, pmid):
    raw = extractor.raw_extract(text)
    stmts = postprocessor.convert_to_indra_stmts(raw, text, pmid)
    return postprocessor.preassemble(stmts)

def run_hybrid(text, pmid):
    logic = rec_logic_preprocessor.preprocess(text)
    raw = extractor.raw_extract(logic)
    stmts = postprocessor.convert_to_indra_stmts(raw, text, pmid)
    return postprocessor.preassemble(stmts)

def run_recursive(text, pmid):
    logic = rec_logic_preprocessor.preprocess(text)
    relation = rec_relation_preprocessor.preprocess(logic)
    raw = extractor.raw_extract(relation)
    stmts = postprocessor.convert_to_indra_stmts(raw, text, pmid)
    return postprocessor.preassemble(stmts)

# Run benchmarks
e2e_stmts = benchmark_method("End-to-End", run_end_to_end)
hybrid_stmts = benchmark_method("Hybrid", run_hybrid)
recursive_stmts = benchmark_method("Recursive", run_recursive)

# Sample output
print("\n🔎 Sample output (Recursive Method):")
for stmt in recursive_stmts[:5]:
    print(stmt)
