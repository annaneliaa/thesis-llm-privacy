import os
import numpy as np
import json
import argparse
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

def main(args):
    DATASET_DIR = args["dataset_dir"]
    DATASET_FILE = args["dataset_file"]
    LANGUAGE = args["language"]
    SPLIT = args["split"]
    OUTPUT_DIR = args["output_dir"]

    EXAMPLE_TOKEN_LEN = args["example_token_len"]
    PREPREFIX_LEN = args["preprefix_len"]
    PREFIX_LEN = args["prefix_len"]
    SUFFIX_LEN = args["suffix_len"]

    logger.info("===== Starting dataset token split generation for language %s with token length %s =====", LANGUAGE, EXAMPLE_TOKEN_LEN)

    europarl_files = [open(DATASET_DIR + "/" + DATASET_FILE + "-" + str(EXAMPLE_TOKEN_LEN) + "." + LANGUAGE + ".jsonl")]

    logger.info("Opened file: %s", str(europarl_files[0].name))

    prompts = {}
    line_count = 0
    batch_size = args["batch_size"]

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
    np.save(os.path.join(npy_arrays_base, SPLIT + "_preprefix.npy"), prompts[:, :PREPREFIX_LEN])
    np.save(os.path.join(npy_arrays_base, SPLIT + "_prefix.npy"), prompts[:, PREPREFIX_LEN:PREPREFIX_LEN+PREFIX_LEN])
    np.save(os.path.join(npy_arrays_base, SPLIT + "_suffix.npy"), prompts[:, PREPREFIX_LEN+PREFIX_LEN:PREPREFIX_LEN+PREFIX_LEN+SUFFIX_LEN])

    logger.info("===== Done ======")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    main(config)
