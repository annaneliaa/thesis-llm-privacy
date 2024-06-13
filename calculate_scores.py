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
import nltk

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
            split_indices = json.load(f)
            # this gives a list of indices present in the training dataset
            exids = split_indices["train"]
    else:
        # The full dataset was used, so we simply use the common exids file generated in the data processing step
        # Path to exids of the dataset
        path = os.path.join(SOURCE_DIR, DATASET_DIR, "csv", str(EXAMPLE_TOKEN_LEN), "common_exids-"+str(EXAMPLE_TOKEN_LEN)+".csv")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            # this gives a list of all common exids in the aligned, balanced, total dataset
            exids = generate_exid_list(path)
        else:
            print(f"File {path} does not exist or is empty, stopping execution.")
            return

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

    # Repeat these steps for the preprefixes in the case that it was used in the experiment
    if(PREPREFIX_LEN > 0):
        preprefix_file = os.path.join(np_dataset_base, f"{SPLIT}_preprefix.npy")
        preprefix_jsonl_file = np_dataset_base + f"/{SPLIT}_preprefix.jsonl"

        # Check if the preprefix jsonl file doesn't exist or is empty
        if not os.path.exists(preprefix_jsonl_file) or os.stat(preprefix_jsonl_file).st_size == 0:
            preprefixes = np.load(preprefix_file)
            generations_to_jsonl(preprefix_jsonl_file, preprefixes, tokenizer, exids)
            
        with open(preprefix_jsonl_file, "r", encoding="utf-8", newline="") as file:
            preprefix_lines = file.readlines()

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

    # Start calculation for BLEU score
    for trial in range(NUM_TRIALS):
        logger.info("Starting BLEU-score calculation for trial %d", trial)

        # Check if the file with BLEU scores for this trial already exists
        bleu_scores_file = os.path.join(bleu_scores_base, f"bleu_scores_trial_{trial}.jsonl")
        bleu_scores_suff_file = os.path.join(bleu_scores_base, f"bleu_scores_suff_trial_{trial}.jsonl")

        logger.info("Saving BLEU scores for trial %s to %s", trial, bleu_scores_file)

        if os.path.exists(bleu_scores_file) and os.path.exists(bleu_scores_suff_file):
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
        full_sentence_scores_b = []
        suffix_only_scores_b = []

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
            
            if(PREPREFIX_LEN > 0):
                preprefix = json.loads(preprefix_lines[index])["text"].strip()
                reference = preprefix + prefix + suffix
            else:
                reference = prefix + suffix

            # Tokenize the candidate to get the last SUFFIX_LEN tokens for suffix comparison only
            suffix_ref = suffix
            suffix_cand = tokenizer.tokenize(candidate)[-SUFFIX_LEN:]
            suffix_cand = tokenizer.decode(suffix_cand, skip_special_tokens=True).replace('Ġ', '')

            full_score = calc_bleu_score(reference, candidate)
            suffix_score = calc_bleu_score(suffix_ref, suffix_cand)

            # Save the BLEU score for each exid in the trial
            full_sentence_scores_b.append({"exid": exid, "score": full_score})
            suffix_only_scores_b.append({"exid": exid, "score": suffix_score})
            index += 1

        with open(bleu_scores_file, "w", encoding="utf-8", newline="") as file:
            for score_obj in full_sentence_scores_b:
                json.dump(score_obj, file, ensure_ascii=False)
                file.write("\n")

        with open(bleu_scores_suff_file, "w", encoding="utf-8", newline="") as file:
            for score_obj in suffix_only_scores_b:
                json.dump(score_obj, file, ensure_ascii=False)
                file.write("\n")

        logger.info("Finished BLEU-score calculation for trial %d", trial)
    
    # Start calculation for METEOR score
    for trial in range(NUM_TRIALS):
        logger.info("Starting METEOR-score calculation for trial %d", trial)

        # Check if the file with BLEU scores for this trial already exists
        meteor_scores_file = os.path.join(meteor_scores_base, f"meteor_scores_trial_{trial}.jsonl")
        meteor_scores_suff_file = os.path.join(bleu_scores_base, f"meteor_scores_suff_trial_{trial}.jsonl")

        logger.info("Saving METEOR scores for trial %s to %s", trial, meteor_scores_file)

        if os.path.exists(meteor_scores_file) and os.path.exists(meteor_scores_suff_file):
            logger.info(
                "METEOR scores for trial %d previously calculated, skipping calculation",
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
        full_sentence_scores_m = []
        suffix_only_scores_m = []

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

            if(PREPREFIX_LEN > 0):
                preprefix = json.loads(preprefix_lines[index])["text"].strip()
                reference = preprefix + prefix + suffix
            else:
                reference = prefix + suffix

            # Tokenize the reference and candidate text
            reference = nltk.tokenize.word_tokenize(reference)
            candidate = nltk.tokenize.word_tokenize(candidate)

            # Tokenize the candidate to get the last SUFFIX_LEN tokens for suffix comparison only
            suffix_ref = suffix
            suffix_cand = tokenizer.tokenize(candidate)[-SUFFIX_LEN:]
            suffix_cand = tokenizer.decode(suffix_cand, skip_special_tokens=True).replace('Ġ', '')

            # Calculate the METEOR score
            full_score = calc_meteor_score(reference, candidate)
            suffix_score = calc_meteor_score(suffix_ref, suffix_cand)

            # Save the BLEU score for each exid in the trial
            full_sentence_scores_m.append({"exid": exid, "score": full_score})
            suffix_only_scores_m.append({"exid": exid, "score": suffix_score})
            index += 1

        with open(bleu_scores_file, "w", encoding="utf-8", newline="") as file:
            for score_obj in full_sentence_scores_m:
                json.dump(score_obj, file, ensure_ascii=False)
                file.write("\n")

        with open(bleu_scores_suff_file, "w", encoding="utf-8", newline="") as file:
            for score_obj in suffix_only_scores_m:
                json.dump(score_obj, file, ensure_ascii=False)
                file.write("\n")

        logger.info("Finished METEOR-score calculation for trial %d", trial)

    logger.info("===== Done ======")


if __name__ == "__main__":
    main()
