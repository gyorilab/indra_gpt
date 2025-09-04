from indra_gpt.util.util import load_input_file
import random
from pathlib import Path
import pickle
from tqdm import tqdm

from ontogpt.io.template_loader import get_template_details
from ontogpt.engines.spires_engine import SPIRESEngine
from ontogpt import DEFAULT_MODEL
random.seed(42)

# Load input file
inputs = load_input_file(input_file_path=None)

# Ensure we don't sample more than available
inputs = random.sample(inputs, min(len(inputs), 100))

# Load schema template
schema_path = (Path.cwd() / "indra_gpt" / "resources" / "indra_statements_schema.yaml").resolve().as_posix()
template_details = get_template_details(template=schema_path)

# Initialize SPIRES Engine
ke = SPIRESEngine(template_details=template_details, model=DEFAULT_MODEL)
target_class_def = template_details[0] 

# Extraction storage
input_text = "TLR2 co-stimulation augmented the TCR activation of the chemokines CXCL10 by 2-3 fold."
extraction = ke.extract_from_text(text=input_text, cls=target_class_def)
print(extraction.extracted_object.json())
