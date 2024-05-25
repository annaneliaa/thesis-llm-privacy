import os
import numpy as np
import json
from transformers import AutoTokenizer
import logging
from IPython.display import display

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s")

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

# set up logger
logger = logging.getLogger()
handler = JupyterHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Define constants
DATASET_DIR = "nl-en"
DATASET_FILE = "europarl-v7.nl-en"
LANGUAGE = "en"
SPLIT = "train"
OUTPUT_DIR = "./datasets"

EXAMPLE_TOKEN_LEN = 200
PREPREFIX_LEN = 100
PREFIX_LEN = 50
SUFFIX_LEN = 50

if __name__ == "__main__":
    logger.info("===== Starting dataset token split generation for language %s with token length %s =====", LANGUAGE, EXAMPLE_TOKEN_LEN)

    europarl_files = [open(DATASET_DIR + "/" + DATASET_FILE + "-" + str(EXAMPLE_TOKEN_LEN) + "." + LANGUAGE + ".jsonl")]

    logger.info("Opened file: %s", str(europarl_files[0].name))

    prompts = {}
    line_count = 0
    batch_size = 32  # Adjust batch size as needed

    for europarl_file in europarl_files:
        line = europarl_file.readline()
        while line:
            json_obj = json.loads(line)
            exid = json_obj["exid"]
            sentence = json_obj["text"]
            tokens = tokenizer.encode(sentence, max_length=1024, truncation=True)
            if len(tokens) > 0:
                prompts[exid] = tokens

            line_count += 1
            if line_count % batch_size == 0:
                logger.info("Processed %d lines", line_count)

            line = europarl_file.readline()

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    npy_arrays_base = os.path.join(OUTPUT_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN))
    os.makedirs(npy_arrays_base, exist_ok=True)

    prompts = [x[1] for x in sorted(prompts.items())]
    prompts = np.array(prompts, dtype=np.uint16)

    # save the token sequences to .npy files to be used in model generation
    np.save(os.path.join(npy_arrays_base, SPLIT + "_dataset.npy"), prompts)
    # split the tokens into preprefix, prefix, and suffix
    np.save(os.path.join(npy_arrays_base, SPLIT + "_preprefix.npy"), prompts[:, :100])
    np.save(os.path.join(npy_arrays_base, SPLIT + "_prefix.npy"), prompts[:, 100:150])
    np.save(os.path.join(npy_arrays_base, SPLIT + "_suffix.npy"), prompts[:, 150:200])

    logger.info("===== Done ======")
