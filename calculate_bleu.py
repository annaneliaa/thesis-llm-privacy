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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s")

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

# Function to calculate the BLEU score between the reference and candidate text
def calc_bleu_score(reference, candidate):
    return sentence_bleu([reference], candidate) 

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # For saving results
    ROOT_DIR = config["root_dir"]
    # Name of the dataset
    DATASET_DIR = config["dataset_dir"]
    # Directory where the .npy files of the dataset are stored
    SOURCE_DIR = config["source_dir"]
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

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Initialize wandb
    wandb.init(
        project="thesis-llm-privacy",
        name="Evaluation BLEU Score - " + EXPERIMENT_NAME + " - " + model,
        config={
            'experiment_name': EXPERIMENT_NAME,
            "dataset": DATASET_DIR,
            "language": LANGUAGE,
            "token_len": EXAMPLE_TOKEN_LEN,
            "prefix_len": PREFIX_LEN,
            "num_trials": NUM_TRIALS
        })
        
    # Set up logger
    logger = logging.getLogger()
    handler = JupyterHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("===== Starting evaluation of similarity between generated and original text in language %s for %d prefix & suffix length =====", LANGUAGE, PREFIX_LEN)

    np_dataset_base = os.path.join(SOURCE_DIR, DATASET_DIR,  LANGUAGE, str(EXAMPLE_TOKEN_LEN))

    logger.info("===== Decoding original prefixes & suffixes =====")
    prefix_file = os.path.join(np_dataset_base, f"{SPLIT}_prefix.jsonl")
    suffix_file = os.path.join(np_dataset_base, f"{SPLIT}_suffix.jsonl")

    if (not os.path.exists(prefix_file) or is_file_empty(prefix_file)) or (not os.path.exists(suffix_file) or is_file_empty(suffix_file)):
        logger.info("Prefix or suffix file is empty or does not exist. Decoding the original dataset...")
        
        prefixes = np.load(prefix_file)
        suffixes = np.load(suffix_file)

        # Decoding prefixes & suffixes from original dataset
        generations_to_jsonl(np_dataset_base + f"/{SPLIT}_prefix.jsonl", prefixes, tokenizer)
        generations_to_jsonl(np_dataset_base + f"/{SPLIT}_suffix.jsonl", suffixes, tokenizer)  

    # Load the original prefix + suffix from the dataset
    with open(prefix_file, "r", encoding="utf-8", newline='') as file:
        prefix_lines = file.readlines()

    with open(suffix_file, "r", encoding="utf-8", newline='') as file:
        suffix_lines = file.readlines()

    # Create a directory to store the BLEU scores
    scores_base = os.path.join(ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME, "scores")
    if not os.path.exists(scores_base):
        os.makedirs(scores_base)

    for trial in range(NUM_TRIALS):
        logger.info("Starting BLEU-score calculation for trial %d", trial)

        # Check if the file with BLEU scores for this trial already exists
        bleu_scores_file = os.path.join(scores_base, f"bleu_scores_trial_{trial}.jsonl")
        logger.info("Saving BLEU scores for trial %s to %s", trial,  bleu_scores_file)

        if os.path.exists(bleu_scores_file):
            logger.info("BLEU scores for trial %d previously calculated, skipping calculation", trial)
            continue

        # Load the decoded generations file of the trial
        trial_file = os.path.join(ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME, "decoded", f"decoded_strings_trial_{trial}.jsonl")
        scores = []

        with open(trial_file, "r", encoding="utf-8", newline='') as file:
            # Read the complete file
            lines = file.readlines()

        for line in lines:
            json_obj = json.loads(line)
            exid = json_obj["exid"]
            candidate = json_obj["text"]

            # Compare the generated text with the original text using the BLEU score using example id
            # Concatenate the prefix and suffix to form the reference text
            prefix = prefix_lines[exid].strip()
            suffix = suffix_lines[exid].strip()
            reference = prefix + suffix

            score = calc_bleu_score(reference, candidate)

            # Log the BLEU score to wandb for plotting
            wandb.log({"bleu_score": score, "trial": trial, "exid": exid})

            # Save the BLEU score for each exid in the trial
            scores.append({"exid": exid, "score": score})

        with open(bleu_scores_file, "w", encoding="utf-8", newline='') as file:
            for score_obj in scores:
                json.dump(score_obj, file, ensure_ascii=False)
                file.write("\n")

        logger.info("Finished BLEU-score calculation for trial %d", trial)

    wandb.finish()

    logger.info("===== Done ======")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input from config file.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config_file)
