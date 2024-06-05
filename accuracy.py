import os
import numpy as np
import json
import logging
from IPython.display import display
import argparse
from experiment_lib import *

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# We consider generations with a BLEU score above this threshold as memorized
THRESHOLD = 7.5e-1

EXACT_MATCH_THRESHOLD = 1.0

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

with open(args.config_file, "r") as f:
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

def main():
    logger.info("====== Calculating number of correct guesses (accuracy) for %s in language %s ======" % (EXPERIMENT_NAME, LANGUAGE))

    experiment_base = os.path.join(ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME)
    bleu_scores_base = os.path.join(experiment_base, "scores")

    complete_score_file = os.path.join(bleu_scores_base, "sorted_compl_bleu_scores.jsonl")
    output_file = os.path.join(experiment_base, "accuracy.json")

    with(open(complete_score_file, "r")) as in_file, open(output_file, "w") as out_file:
        lines = in_file.readlines()
        NUM_GENERATIONS = len(lines) * NUM_TRIALS


        NUM_EXACT_MATCH = 0
        NUM_CORRECT = 0
        NUM_MISS = 0
        
        out_objects = []
        for line in lines:
            json_obj = json.loads(line)
            exid = json_obj["exid"]
            scores = json_obj["scores"]

            trials_corr = []
            trials_exact = []
            for score in scores:
                if score["score"] >= THRESHOLD:
                    trials_corr.append(score)
                    if score["score"] == EXACT_MATCH_THRESHOLD:
                        trials_exact.append(score)
                else:
                    NUM_MISS += 1

            NUM_CORRECT += len(trials_corr)
            NUM_EXACT_MATCH += len(trials_exact)

            # only want to store data on correct guesses here
            if(len(trials_corr) > 0):
                out_obj = {
                    "exid": exid,
                    "num_correct": len(trials_corr),
                    "num_exact_match": len(trials_exact),
                    "trials_correct": trials_corr,
                    "trials_exact": trials_exact
                }
                out_objects.append(out_obj)
            
        logger.info("Finished counting amount of correct guesses.")
    

        logger.info("Saving output to %s" % output_file)
        # save output to a file
        json.dump({"experiment": EXPERIMENT_NAME, "num_correct": NUM_CORRECT, "num_exact_match": NUM_EXACT_MATCH, "num_generations": NUM_GENERATIONS, "num_miss": NUM_MISS}, out_file)
        out_file.write("\n")

        # sort on number of correct guesses
        # such that samples that were guessed correctly the most appear at the top of the file
        out_objects = sorted(out_objects, key=lambda x: x['num_correct'], reverse=True)
        for out_obj in out_objects:
            json.dump(out_obj, out_file)
            out_file.write("\n")

        out_file.close()
        in_file.close()

    logger.info("====== Finished calculating number of correct guesses (accuracy) for %s in language %s ======" % (EXPERIMENT_NAME, LANGUAGE))

if __name__ == "__main__":
    main()