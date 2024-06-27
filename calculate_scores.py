import os
import numpy as np
import json
import logging
from IPython.display import display
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import argparse
from transformers import AutoTokenizer
from experiment_lib import *
from time import sleep

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
parser.add_argument(
    "--do_meteor", type=bool, required=False, help="Include METEOR score calculation or not"
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

NUM_TRIALS = 100
# We use the GPT2 tokenizer here to ensure consistency in the experimental setup
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Function to calculate the BLEU score between the reference and candidate text
def calc_bleu_score(reference, candidate):
    return sentence_bleu([reference], candidate)

# Function to calculate the METEOR score between the reference and candidate text
def calc_meteor_score(reference, candidate):
    return meteor_score([reference], candidate)

def main():
    logger.info(
        "===== Starting BLEU- & METEOR-score calculation between generated and original text in language %s for %d prefix & suffix length =====",
        LANGUAGE,
        PREFIX_LEN,
    )

    np_dataset_base = os.path.join(
        SOURCE_DIR, DATASET_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN), MODEL_NAME
    )

    logger.info("===== Decoding original preprefixes, prefixes & suffixes =====")

    prefix_file = os.path.join(np_dataset_base, f"{SPLIT}_prefix.npy")
    suffix_file = os.path.join(np_dataset_base, f"{SPLIT}_suffix.npy")

    prefix_jsonl_file = np_dataset_base + f"/{SPLIT}_prefix.jsonl"
    suffix_jsonl_file = np_dataset_base + f"/{SPLIT}_suffix.jsonl"

    if(SPLIT == "train"):
        # We use a trained model, so we need to load the exids from the training dataset only
        path = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), "split_indices.json")
        with open(path, "r") as f:
            logger.info(f"Loading split indices from {path}")
            split_indices = json.load(f)
            # this gives a list of indices present in the training dataset
            exids = split_indices["train"]
    else:
        logger.info(f"Loading exids from the common exids file")
        # The full dataset was used, so we simply use the common exids file generated in the data processing step
        # Path to exids of the dataset
        path = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", str(EXAMPLE_TOKEN_LEN), "common_exids-"+str(EXAMPLE_TOKEN_LEN)+".csv")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            # this gives a list of all common exids in the aligned, balanced, total dataset
            exids = generate_exid_list(path)
        else:
            print(f"File {path} does not exist or is empty, stopping execution.")
            return

    # Check if the suffix jsonl file doesn't exist or is empty
    if not os.path.exists(suffix_jsonl_file) or os.stat(suffix_jsonl_file).st_size == 0:
        suffixes = np.load(suffix_file)
        generations_to_jsonl(suffix_jsonl_file, suffixes, tokenizer, exids)    

    with open(suffix_jsonl_file, "r", encoding="utf-8", newline="") as file:
        suffix_lines = file.readlines()


    # Create a directory to store the BLEU scores
    bleu_scores_base = os.path.join(
        ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME, "bleu_scores"
    )

    meteor_scores_base = os.path.join(
        ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME, "meteor_scores"
    )

    if not os.path.exists(bleu_scores_base) or not os.path.exists(meteor_scores_base):
        os.makedirs(bleu_scores_base, exist_ok=True)
        os.makedirs(meteor_scores_base, exist_ok=True)

    sleep(2)

    # Start calculation for BLEU score
    for trial in range(NUM_TRIALS):
        logger.info("Starting BLEU-score calculation for trial %d", trial)

        # Check if the file with BLEU scores for this trial already exists
        bleu_scores_file = os.path.join(bleu_scores_base, f"bleu_scores_trial_{trial}.jsonl")

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
        bleu_scores = []

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
            suffix = json.loads(suffix_lines[index])["text"].strip()

            # Tokenize the candidate to get the last SUFFIX_LEN tokens for suffix comparison only
            suffix_ref = tokenizer.tokenize(suffix)
            suffix_ref = [s.replace('Ġ', ' ') for s in suffix_ref]

            cand = tokenizer.tokenize(candidate)
            # skip the first 50 tokens as that was the input prefix
            cand = cand[-SUFFIX_LEN:]
            suffix_cand = [c.replace('Ġ', ' ') for c in cand]
            
            bleu_score = calc_bleu_score(suffix_ref, suffix_cand)

            # Save the BLEU score for each exid in the trial
            bleu_scores.append({"exid": exid, "score": float(bleu_score)})
            index += 1

        with open(bleu_scores_file, "w", encoding="utf-8", newline="") as file:
            for score_obj in bleu_scores:
                json.dump(score_obj, file, ensure_ascii=False)
                file.write("\n")

        logger.info("Finished BLEU-score calculation for trial %d", trial)
    
    # if args.do_meteor == True:
    #     # Start calculation for METEOR score
    #     for trial in range(NUM_TRIALS):
    #         logger.info("Starting METEOR-score calculation for trial %d", trial)

    #         # Check if the file with BLEU scores for this trial already exists
    #         meteor_scores_file = os.path.join(meteor_scores_base, f"meteor_scores_trial_{trial}.jsonl")

    #         logger.info("Saving METEOR scores for trial %s to %s", trial, meteor_scores_file)

    #         if os.path.exists(meteor_scores_file):
    #             logger.info(
    #                 "METEOR scores for trial %d previously calculated, skipping calculation",
    #                 trial,
    #             )
    #             continue

    #         # Load the decoded generations file of the trial
    #         trial_file = os.path.join(
    #             ROOT_DIR,
    #             DATASET_DIR,
    #             LANGUAGE,
    #             EXPERIMENT_NAME,
    #             "decoded",
    #             f"decoded_strings_trial_{trial}.jsonl",
    #         )

    #         meteor_scores = []

    #         with open(trial_file, "r", encoding="utf-8", newline="") as file:
    #             # Read the complete file
    #             lines = file.readlines()

    #         # to iterate over prefix and suffix lists
    #         index = 0
    #         for line in lines:
    #             json_obj = json.loads(line)
    #             exid = json_obj["exid"]
    #             candidate = json_obj["text"]

    #             # Compare the generated text with the original text using the METEOR score using example id
    #             suffix = json.loads(suffix_lines[index])["text"].strip()
    #             suffix_ref = tokenizer.tokenize(suffix)
    #             suffix_ref = [s.replace('Ġ', ' ') for s in suffix_ref]

    #             # Tokenize the candidate to get the last SUFFIX_LEN tokens for suffix comparison only
    #             cand = tokenizer.tokenize(candidate)
    #             cand = cand[-SUFFIX_LEN:]
    #             suffix_cand = [c.replace('Ġ', ' ') for c in cand]

    #             # Calculate the METEOR score
    #             meteor_score = calc_meteor_score(suffix_ref, suffix_cand)

    #             # Save the BLEU score for each exid in the trial
    #             meteor_scores.append({"exid": exid, "score": float(meteor_score)})
    #             index += 1

    #         with open(meteor_scores_file, "w", encoding="utf-8", newline="") as file:
    #             for score_obj in meteor_scores:
    #                 json.dump(score_obj, file, ensure_ascii=False)
    #                 file.write("\n")

    #         logger.info("Finished METEOR-score calculation for trial %d", trial)

    logger.info("===== Done ======")


if __name__ == "__main__":
    main()
