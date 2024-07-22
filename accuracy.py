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

def main():
    logger.info("====== Calculating number of correct guesses (accuracy) for %s in language %s ======" % (EXPERIMENT_NAME, LANGUAGE))

    NUM_TRIALS = 100
    
    experiment_base = os.path.join(ROOT_DIR, DATASET_DIR, LANGUAGE, EXPERIMENT_NAME)
    bleu_scores_base = os.path.join(experiment_base, "bleu_scores")

    complete_score_file = os.path.join(bleu_scores_base, "sorted_compl_bleu_scores.jsonl")
    output_file = os.path.join(experiment_base, "accuracy.jsonl")

    logger.info("Reading from %s" % complete_score_file)

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
                # convert string to a float to keep fractional part
                s = float(score["score"])
                obj = {
                    "trial": score["trial"],
                    "score": s
                }
                if s >= THRESHOLD:
                    trials_corr.append(obj)
                    if s == EXACT_MATCH_THRESHOLD:
                        trials_exact.append(obj)
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