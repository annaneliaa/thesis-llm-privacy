import os
import numpy as np
import json
import argparse
from transformers import AutoTokenizer
import logging
from IPython.display import display
from experiment_lib import load_constants_from_config

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

(
    ROOT_DIR, 
    DATASET_DIR, 
    SOURCE_DIR, 
    DATASET_NAME, 
    EXPERIMENT_NAME, 
    NUM_TRIALS, 
    PREFIX_LEN, 
    SUFFIX_LEN, 
    PREPREFIX_LEN, 
    LANGUAGE, 
    SPLIT, 
    EXAMPLE_TOKEN_LEN, 
    SOURCE_FILE, 
    BATCH_SIZE, 
    MODEL_NAME, 
    TRAIN_FILE, 
    VAL_FILE, 
    VAL_SPLIT, 
    SEED
) = load_constants_from_config(config)

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({"pad_token": ""})

def main():
    # Input: A dataset file with sentences in a specific language in JSONL format
    # This script will tokenize the sentences and save the token sequences in .npy files
    # Output: .npy files with the token sequences, split into preprefix, prefix, and suffix
    # The .npy files will be saved in the source directory
    # A tokenized version of the complete dataset will also be saved in the source directory
    logger.info(
        "===== Starting dataset token split generation for language %s with token length %s =====",
        LANGUAGE,
        EXAMPLE_TOKEN_LEN,
    )

    if SPLIT != "":
        logger.info("Split: ", SPLIT)
        ds_files = [open(os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), DATASET_NAME + "." + LANGUAGE + "-" + SPLIT + ".jsonl"))]
    else:
        logger.info("Split: ", SPLIT)
        ds_files = [open(os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), DATASET_NAME + "." + LANGUAGE + ".jsonl"))]

    logger.info("Opened file: %s", str(ds_files[0].name))

    prompts = {}
    line_count = 0

    for file in ds_files:
        line = file.readline()
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
                prompts[exid] = tokens #????

            line_count += 1
            if line_count % BATCH_SIZE == 0:
                logger.info("Processed %d lines", line_count)

            line = file.readline()

    if not os.path.exists(SOURCE_DIR):
        os.mkdir(SOURCE_DIR)

    npy_arrays_base = os.path.join(SOURCE_DIR, DATASET_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN), MODEL_NAME)
    os.makedirs(npy_arrays_base, exist_ok=True)

    prompts = [x[1] for x in sorted(prompts.items())]
    prompts = np.array(prompts, dtype=np.uint16)

    # save the token sequences to .npy files to be used in model generation
    np.save(os.path.join(npy_arrays_base, SPLIT + "_dataset.npy"), prompts)
    # split the tokens into preprefix, prefix, and suffix

    if EXAMPLE_TOKEN_LEN == 250:
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_preprefix.npy"), prompts[:, :150]
        )
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_prefix.npy"), prompts[:, 150:200]
        )
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_suffix.npy"), prompts[:, 200:250]
        )

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

    if EXAMPLE_TOKEN_LEN == 150:
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_preprefix.npy"), prompts[:, :50]
        )
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_prefix.npy"), prompts[:, 50:100]
        )
        np.save(
            os.path.join(npy_arrays_base, SPLIT + "_suffix.npy"), prompts[:, 100:150]
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
