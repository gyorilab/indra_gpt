from indra_gpt.util.util import load_input_file
import random
from pathlib import Path
import pickle
from tqdm import tqdm
import time

from ontogpt.io.template_loader import get_template_details
from ontogpt.engines.spires_engine import SPIRESEngine
from ontogpt import DEFAULT_MODEL
random.seed(42)

# Load input file
inputs =pickle.load(open("/Users/thomaslim/gyorilab/indra_gpt/local_data/test_set.pkl", "rb"))

# Load schema template
schema_path = (Path.cwd() / "indra_gpt" / "resources" / "indra_statements_schema.yaml").resolve().as_posix()
template_details = get_template_details(template=schema_path)

# Initialize SPIRES Engine
ke = SPIRESEngine(template_details=template_details, model=DEFAULT_MODEL)
target_class_def = template_details[0] 

# Extraction storage
extractions = {}

# Process extractions
for i, input_data in tqdm(enumerate(inputs), total=len(inputs)):
    data = {"input": input_data}  # Store original input
    try:
        extraction = ke.extract_from_text(text=input_data['text'], cls=target_class_def)
        data['extraction'] = extraction
    except Exception as e:
        print(f"Error processing input {i}: {e}")
        data['extraction'] = str(e)  # Store error as string for debugging
    extractions[i] = data


# Save extractions as pickle
today = time.strftime("%Y%m%d%H")
output_folder = Path.cwd() / "output" / "ontogpt_extraction_results" / today
output_folder.mkdir(parents=True, exist_ok=True)
output_file = output_folder / f"ontogpt_extractions_{today}.pkl"

with open(output_file, "wb") as f:
    pickle.dump(extractions, f)

print(f"Extractions saved to {output_file}")
