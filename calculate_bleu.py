import os
import numpy as np
import json
import logging
from IPython.display import display
from nltk.translate.bleu_score import sentence_bleu
import argparse
import wandb
from transformers import AutoTokenizer
from experiment_lib import *

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

# Set up logger
logger = logging.getLogger()
handler = JupyterHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Process input from config file.")
parser.add_argument(
    "--config_file", type=str, required=True, help="Path to the configuration file"
)
args = parser.parse_args()

# Load configuration files
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

# We use the non finetuned model here to evaluate the scores to ensure consistency in the experi2mental setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Function to calculate the BLEU score between the reference and candidate text
def calc_bleu_score(reference, candidate):
    return sentence_bleu([reference], candidate)

def main():
    logger.info(
        "===== Starting BLEU-score calculation between generated and original text in language %s for %d prefix & suffix length =====",
        LANGUAGE,
        PREFIX_LEN,
    )

    np_dataset_base = os.path.join(
        SOURCE_DIR, DATASET_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN), MODEL_NAME
    )

    logger.info("===== Decoding original prefixes & suffixes =====")
    prefix_file = os.path.join(np_dataset_base, f"{SPLIT}_prefix.npy")
    suffix_file = os.path.join(np_dataset_base, f"{SPLIT}_suffix.npy")


    prefix_jsonl_file = np_dataset_base + f"/{SPLIT}_prefix.jsonl"
    suffix_jsonl_file = np_dataset_base + f"/{SPLIT}_suffix.jsonl"

    # Path to exids of the dataset
    exids = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", str(EXAMPLE_TOKEN_LEN), "common_exids-"+str(EXAMPLE_TOKEN_LEN)+".csv")

    # Check if the prefix jsonl file doesn't exist or is empty
    if not os.path.exists(prefix_jsonl_file) or os.stat(prefix_jsonl_file).st_size == 0:
        prefixes = np.load(prefix_file)
        generations_to_jsonl(prefix_jsonl_file, prefixes, tokenizer, exids)

    # Check if the suffix jsonl file doesn't exist or is empty
    if not os.path.exists(suffix_jsonl_file) or os.stat(suffix_jsonl_file).st_size == 0:
        suffixes = np.load(suffix_file)
        generations_to_jsonl(suffix_jsonl_file, suffixes, tokenizer, exids)

    # Load the original prefix + suffix from the dataset
    # Fill lists
    with open(prefix_jsonl_file, "r", encoding="utf-8", newline="") as file:
        prefix_lines = file.readlines()

    with open(suffix_jsonl_file, "r", encoding="utf-8", newline="") as file:
        suffix_lines = file.readlines()

    # Create a directory to store the BLEU scores
    scores_base = os.path.join(
        ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME, "scores"
    )
    if not os.path.exists(scores_base):
        os.makedirs(scores_base)

    for trial in range(NUM_TRIALS):
        logger.info("Starting BLEU-score calculation for trial %d", trial)

        # Check if the file with BLEU scores for this trial already exists
        bleu_scores_file = os.path.join(scores_base, f"bleu_scores_trial_{trial}.jsonl")
        logger.info("Saving BLEU scores for trial %s to %s", trial, bleu_scores_file)

        if os.path.exists(bleu_scores_file):
            logger.info(
                "BLEU scores for trial %d previously calculated, skipping calculation",
                trial,
            )
            continue

        # Load the decoded generations file of the trial
        trial_file = os.path.join(
            ROOT_DIR,
            DATASET_DIR,
            LANGUAGE,
            EXPERIMENT_NAME,
            "decoded",
            f"decoded_strings_trial_{trial}.jsonl",
        )
        scores = []

        with open(trial_file, "r", encoding="utf-8", newline="") as file:
            # Read the complete file
            lines = file.readlines()

        # to iterate over prefix and suffix lists
        index = 0
        for line in lines:
            json_obj = json.loads(line)
            exid = json_obj["exid"]
            candidate = json_obj["text"]

            # Compare the generated text with the original text using the BLEU score using example id
            # Concatenate the prefix and suffix to form the reference text
            prefix = json.loads(prefix_lines[index])["text"].strip()
            suffix = json.loads(suffix_lines[index])["text"].strip()
            reference = prefix + suffix
            
            score = calc_bleu_score(reference, candidate)

            # Log the BLEU score to wandb for plotting
            # wandb.log({"bleu_score": score, "trial": trial, "exid": exid})

            # Save the BLEU score for each exid in the trial
            scores.append({"exid": exid, "score": score})
            index += 1

        with open(bleu_scores_file, "w", encoding="utf-8", newline="") as file:
            for score_obj in scores:
                json.dump(score_obj, file, ensure_ascii=False)
                file.write("\n")

        logger.info("Finished BLEU-score calculation for trial %d", trial)

    # wandb.finish()

    logger.info("===== Done ======")


if __name__ == "__main__":
    main()
