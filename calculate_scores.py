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
        "===== Starting BLEU-score calculation between generated and original text in language %s for %d prefix & suffix length =====",
        LANGUAGE,
        PREFIX_LEN,
    )

    np_dataset_base = os.path.join(
        SOURCE_DIR, DATASET_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN), MODEL_NAME
    )

    logger.info("===== Decoding original preprefixes, prefixes & suffixes =====")

    # prefix_file = os.path.join(np_dataset_base, f"{SPLIT}_prefix.npy")
    suffix_file = os.path.join(np_dataset_base, f"{SPLIT}_suffix.npy")

    # prefix_jsonl_file = np_dataset_base + f"/{SPLIT}_prefix.jsonl"
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

    suffix_lines = []
    # Check if the suffix jsonl file doesn't exist or is empty
    # if not os.path.exists(suffix_jsonl_file) or os.stat(suffix_jsonl_file).st_size == 0:
    suffixes = np.load(suffix_file)
    generations_to_jsonl(suffix_jsonl_file, suffixes, tokenizer, exids)   

    exids_file = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), "prompt-train_dataset-exids-intersect.json")
    with open(exids_file, "r") as f:
        exids = json.load(f)

    # print(sorted(exids))

    sleep(5)
    # filter out the suffixes that are not in the exids list
    with open(suffix_jsonl_file, "r", encoding="utf-8", newline="") as file:
            for line in file:
                json_obj = json.loads(line)
                exid = json_obj["exid"]
                if exid in exids:
                    suffix_lines.append(json_obj)
    logger.info("Filtered suffixes to only include exids in the exids list")
        
    prompt_train_dataset_suffixes = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), f"prompt-train_dataset_suffixes-{LANGUAGE}.jsonl")
        # save for checking
    with open(prompt_train_dataset_suffixes, 'w') as f:
            for line in suffix_lines:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')

    logger.info("Saved filtered suffixes to" + prompt_train_dataset_suffixes)


    # Create a directory to store the BLEU scores
    bleu_scores_base = os.path.join(
        ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME, "bleu_scores"
    )

    if not os.path.exists(bleu_scores_base):
        os.makedirs(bleu_scores_base, exist_ok=True)

    sleep(2)

    print("PREFIX_LEN: ", PREFIX_LEN)
    print("SUFFIX_LEN: ", SUFFIX_LEN)
    print("PREPREFIX_LEN: ", PREPREFIX_LEN)

    null = False
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
            f"decoded_strings_trial_{trial}_filtered.jsonl",
        )
        bleu_scores = []

        with open(trial_file, "r", encoding="utf-8", newline="") as file:
            # Read the complete file
            lines = file.readlines()

        # to iterate over prefix and suffix lists
        index = 0
        # interate over all model generations in the trial
        for line in lines:
            json_obj = json.loads(line)
            exid = json_obj["exid"]

            if int(exid) == 0:
                null = True

            candidate = json_obj["text"]

            # Compare the generated text with the original text using the BLEU score using example id
            suffix = suffix_lines[index]["text"].strip()

            # print("Suffix: ", suffix)
            
            suffix_ref = tokenizer.tokenize(suffix)
            suffix_ref = [s.replace('Ġ', ' ') for s in suffix_ref]

            # Tokenize the candidate to get the last SUFFIX_LEN tokens for suffix comparison only
            cand = tokenizer.tokenize(candidate)
            cand = cand[-SUFFIX_LEN:]
            # cand = cand[PREPREFIX_LEN + PREFIX_LEN:]
            suffix_cand = [c.replace('Ġ', ' ') for c in cand]

            # print("Candidate: ", tokenizer.convert_tokens_to_string(suffix_cand))


            # print("Ref len: ", len(suffix_ref))
            # print("Cand len: ", len(suffix_cand))
            bleu_score = calc_bleu_score(suffix_ref, suffix_cand)

            # print("---------")

            # Save the BLEU score for each exid in the trial
            bleu_scores.append({"exid": exid, "score": float(bleu_score)})
            index += 1

        with open(bleu_scores_file, "w", encoding="utf-8", newline="") as file:
            for score_obj in bleu_scores:
                json.dump(score_obj, file, ensure_ascii=False)
                file.write("\n")


        logger.info("Finished BLEU-score calculation for trial %d", trial)

    logger.info("===== Done ======")

    print("Null: ", null)


if __name__ == "__main__":
    main()
