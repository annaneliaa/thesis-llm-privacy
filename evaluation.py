import os
import numpy as np
import json
import logging
from IPython.display import display
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s")

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

# Set up logger
logger = logging.getLogger()
handler = JupyterHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Define constants
TEXT_DIR = "tmp/"
EXPERIMENT_NAME = "test3"
LANGUAGE = "en"
SPLIT = "train"

DATASET_DIR = "./datasets"

EXAMPLE_TOKEN_LEN = 200
PREPREFIX_LEN = 100
PREFIX_LEN = 50
SUFFIX_LEN = 50

NUM_TRIALS = 1

def evaluate_bleu_score(reference, candidate):
    return sentence_bleu([reference], candidate)

if __name__ == "__main__":
    logger.info("===== Starting evaluation of similarity between generated and original text in language %s for %d prefix & suffix length =====", LANGUAGE, PREFIX_LEN)

    dataset_base = os.path.join(DATASET_DIR, LANGUAGE, str(EXAMPLE_TOKEN_LEN))
    prefix_file = os.path.join(dataset_base, f"{SPLIT}_prefix.jsonl")
    suffix_file = os.path.join(dataset_base, f"{SPLIT}_suffix.jsonl")

    # Load the original prefix + suffix from the dataset, connected
    with open(prefix_file, "r", encoding="utf-8", newline='') as file:
        prefix_lines = file.readlines()

    with open(suffix_file, "r", encoding="utf-8", newline='') as file:
        suffix_lines = file.readlines()

    scores_base = os.path.join(TEXT_DIR, EXPERIMENT_NAME, "scores")
    if not os.path.exists(scores_base):
        os.makedirs(scores_base)

    for trial in range(NUM_TRIALS):
        logger.info("Starting BLEU-score evaluation for trial %d", trial)
        # Load the decoded generations file of the trial
        trial_file = os.path.join(TEXT_DIR, EXPERIMENT_NAME, "decoded", f"decoded_strings_trial_{trial}.jsonl")
        scores = []
        with open(trial_file, "r", encoding="utf-8", newline='') as file:
            lines = file.readlines()

        for line in lines:
            json_obj = json.loads(line)
            exid = json_obj["exid"]
            candidate = json_obj["text"]

            # Compare the generated text with the original text using the BLEU score
            prefix = prefix_lines[exid].strip()
            suffix = suffix_lines[exid].strip()
            reference = prefix + suffix

            score = evaluate_bleu_score(reference, candidate)
            # Save the BLEU score for each exid in the trial
            scores.append({"exid": exid, "score": score})

        with open(os.path.join(scores_base, f"bleu_scores_trial{trial}.jsonl"), "w", encoding="utf-8", newline='') as file:
            for score_obj in scores:
                json.dump(score_obj, file, ensure_ascii=False)
                file.write("\n")

        logger.info("Finished BLEU-score evaluation for trial %d", trial)

    logger.info("===== Done ======")
