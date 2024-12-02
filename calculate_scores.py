import os
import numpy as np
import json
import logging
from IPython.display import display
from nltk.translate.bleu_score import sentence_bleu
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
    PREPROCESSING_SUFFIX,
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

assert((NUM_TRIALS ==100))

# We use the GPT2 tokenizer here to ensure consistency in the experimental setup 
# when calculating scores for each model size
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Function to calculate the BLEU score between the reference and candidate text
def calc_bleu_score(reference, candidate):
    return sentence_bleu([reference], candidate)

def main():
    logger.info(
        "===== Starting BLEU-score calculation between generated and original text in language %s for %d prefix & suffix length =====",
        LANGUAGE,
        PREFIX_LEN,
    )

    logger.info("===== Preparatory steps... =====")
    np_dataset_base = os.path.join(
        SOURCE_DIR, DATASET_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN), MODEL_NAME
    )

    logger.info("===== Decoding original preprefixes, prefixes & suffixes =====")

    full_sample_file = os.path.join(np_dataset_base, f"{SPLIT}_dataset.npy")
    suffix_file = os.path.join(np_dataset_base, f"{SPLIT}_suffix.npy")

    full_sample_jsonl_file = np_dataset_base + f"/{SPLIT}_dataset.jsonl"
    suffix_jsonl_file = np_dataset_base + f"/{SPLIT}_suffix.jsonl"

    if(SPLIT == "train"):
        # # We use a trained model, so we need to load the exids from the training dataset only

        # Exids here are the indexes of the original dataset selected for training
        # path = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), "split_indices.json")
        # with open(path, "r") as f:
        #     logger.info(f"Loading split indices from {path}")
        #     split_indices = json.load(f)
        #     # this gives a list of indices present in the training dataset
        #     exids = split_indices["train"]

        # Fix for clashes in exids encountered after experiments
        exids_file = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), "prompt-train_dataset-exids-intersect.json")
        logger.info(f"Loading exids from {exids_file}")
        with open(exids_file, "r") as f:
            exids = json.load(f)
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

    # decode all original suffixes from dataset split
    suffix_lines = []
    suffixes = np.load(suffix_file)
    generations_to_jsonl(suffix_jsonl_file, suffixes, tokenizer, exids)   

    # decode all original complete sentences from dataset split
    full_references = []
    full_samples = np.load(full_sample_file)
    generations_to_jsonl(full_sample_jsonl_file, full_samples, tokenizer, exids)

    sleep(2)
    
    # why you do this again? remove?
    with open(full_sample_jsonl_file, "r", encoding="utf-8", newline="") as file:
        for line in file:
            json_obj = json.loads(line)
            exid = json_obj["exid"]
            if exid in exids:
                full_references.append(json_obj)
    logger.info(f"Full samples written to {full_sample_jsonl_file}")

    sleep(2)

    # filter out the suffixes that are not in the exids list
    with open(suffix_jsonl_file, "r", encoding="utf-8", newline="") as file:
            for line in file:
                json_obj = json.loads(line)
                exid = json_obj["exid"]
                if exid in exids:
                    suffix_lines.append(json_obj)
    logger.info("Filtered suffixes to only include exids in the exids list")
    
    # filtering suffixes as fix for the exid clash encountered. furthermore this code is not needed
    prompt_train_dataset_suffixes = os.path.join(DATASET_DIR, str(EXAMPLE_TOKEN_LEN), f"prompt-train_dataset_suffixes-{LANGUAGE}.jsonl")
    # save for checking
    with open(prompt_train_dataset_suffixes, 'w') as f:
            for line in suffix_lines:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')

    logger.info("Saved filtered suffixes to" + prompt_train_dataset_suffixes)


    logger.info("===== Starting BLEU-score calculatiosn now... =====")
    # Create a directory to store the BLEU scores
    bleu_scores_base = os.path.join(
        ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME, "bleu_scores2"
    )

    if not os.path.exists(bleu_scores_base):
        os.makedirs(bleu_scores_base, exist_ok=True)

    print("PREFIX_LEN: ", PREFIX_LEN)
    print("SUFFIX_LEN: ", SUFFIX_LEN)
    print("PREPREFIX_LEN: ", PREPREFIX_LEN)

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
            # f"decoded_strings_trial_{trial}_filtered.jsonl",
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

            candidate = json_obj["text"]

            # Compare the generated text with the original text using the BLEU score using example id
            suffix = suffix_lines[index]["text"].strip()
            
            suffix_ref = tokenizer.tokenize(suffix)
            suffix_ref = [s.replace('Ġ', ' ') for s in suffix_ref]

            # this step, check with GH version, i think this was the word-level version for BLEU (bleuscores2)
            suffix_ref = tokenizer.convert_tokens_to_string(suffix_ref)
            suffix_ref = suffix_ref.split()

            # Tokenize the candidate to get the last SUFFIX_LEN tokens for suffix comparison only
            cand = tokenizer.tokenize(candidate)
            cand = cand[-SUFFIX_LEN:]
            suffix_cand = [c.replace('Ġ', ' ') for c in cand]
            suffix_cand = tokenizer.convert_tokens_to_string(suffix_cand)
            suffix_cand = suffix_cand.split()

            bleu_score = calc_bleu_score(suffix_ref, suffix_cand)

            # Save the BLEU score for each exid in the trial
            bleu_scores.append({"exid": exid, "score": float(bleu_score)})
            index += 1
        
        with open(bleu_scores_file, "w", encoding="utf-8", newline="") as file:
            for score_obj in bleu_scores:
                json.dump(score_obj, file, ensure_ascii=False)
                file.write("\n")


        logger.info("Finished BLEU-score calculation for trial %d", trial)

    logger.info("===== Done ======")

if __name__ == "__main__":
    main()
