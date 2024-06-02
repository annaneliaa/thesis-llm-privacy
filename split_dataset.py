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

parser = argparse.ArgumentParser(description="Process input from config file.")
parser.add_argument(
    "--config_file", type=str, required=True, help="Path to the configuration file"
)
args = parser.parse_args()

with open(args.config_file, "r") as f:
    config = json.load(f)

# For saving results
ROOT_DIR = config["root_dir"]
# Name of the dataset
DATASET_DIR = config["dataset_dir"]
# Directory where the .npy files of the dataset are stored
SOURCE_DIR = config["source_dir"]
# Name of the dataset
DATASET_FILE = config["dataset_file"]
# Name of the experiment
EXPERIMENT_NAME = config["experiment_name"]
# Number of trials
NUM_TRIALS = config["num_trials"]
# Length of the prefix
PREFIX_LEN = config["prefix_len"]
# Length of the suffix
SUFFIX_LEN = config["suffix_len"]
# Preprefix length
PREPREFIX_LEN = config["preprefix_len"]
# Language of the scenario (EN/NL)
LANGUAGE = config["language"]
# Number of tokens in the complete sequences
EXAMPLE_TOKEN_LEN = config["example_token_len"]
# Batch size for feeding prompts to the model
BATCH_SIZE = config["batch_size"]
# Split of the dataset to use (train/val/test)
SPLIT = config["split"]
# Name of the model to use
model = config["model"]

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.add_special_tokens({"pad_token": ""})

def main():
    logger.info(
        "===== Starting dataset token split generation for language %s with token length %s =====",
        LANGUAGE,
        EXAMPLE_TOKEN_LEN,
    )

    europarl_files = [
        open(
            DATASET_DIR
            + "/"
            + DATASET_FILE
            + "-"
            + str(EXAMPLE_TOKEN_LEN)
            + "."
            + LANGUAGE
            + ".jsonl"
        )
    ]

    logger.info("Opened file: %s", str(europarl_files[0].name))

    prompts = {}
    line_count = 0

    for europarl_file in europarl_files:
        line = europarl_file.readline()
        while line:
            json_obj = json.loads(line)
            exid = json_obj["exid"]
            sentence = json_obj["text"]
            tokens = tokenizer.encode(
                sentence,
                max_length=EXAMPLE_TOKEN_LEN,
                truncation=True,
                padding="max_length",
            )
            if len(tokens) > 0:
                prompts[exid] = tokens

            line_count += 1
            if line_count % BATCH_SIZE == 0:
                logger.info("Processed %d lines", line_count)

            line = europarl_file.readline()

    if not os.path.exists(SOURCE_DIR):
        os.mkdir(SOURCE_DIR)

    npy_arrays_base = os.path.join(SOURCE_DIR, DATASET_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN), model)
    os.makedirs(npy_arrays_base, exist_ok=True)

    prompts = [x[1] for x in sorted(prompts.items())]
    prompts = np.array(prompts, dtype=np.uint16)

    # save the token sequences to .npy files to be used in model generation
    np.save(os.path.join(npy_arrays_base, SPLIT + "_dataset.npy"), prompts)
    # split the tokens into preprefix, prefix, and suffix

    if EXAMPLE_TOKEN_LEN == 200:
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_preprefix.npy"), prompts[:, :100]
        )
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_prefix.npy"), prompts[:, 100:150]
        )
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_suffix.npy"), prompts[:, 150:200]
        )

    elif EXAMPLE_TOKEN_LEN == 100:
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_prefix.npy"),
            prompts[:, 0:PREFIX_LEN],
        )
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_suffix.npy"),
            prompts[:, PREFIX_LEN : PREFIX_LEN + SUFFIX_LEN],
        )

    logger.info("===== Done ======")


if __name__ == "__main__":
    main()
