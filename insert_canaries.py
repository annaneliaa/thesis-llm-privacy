import os
import logging
import argparse
import json
import random
from transformers import set_seed
from util_lib import load_constants_from_config, initLogger

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up logger
logger = initLogger()

logger.info("Parsing arguments...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file")
parser.add_argument("--insertions", type=int, required=False, help="The number of times the canary is inserted")
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
    PREPROCESSING,
    PREPROCESSING_SUFFIX,
    NORMALIZATION,
    BATCHING,
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
    VAL_SPLIT, 
    SEED
) = load_constants_from_config(config)

languages = ["en", "nl"]

set_seed(SEED)

def main():
    if args.insertions:
        insertions = args.insertions
    else:
        insertions = 1
    logger.info("==== Starting insertion of %d insertions =====", insertions)
    dataset_base = os.path.join(DATASET_DIR, DATASET_NAME)
    indices = []
    logger.info("Generating indices...")
    for i in range(insertions):
        indices.append(random.random())
    
    for lang in languages:
        logger.info(f"Inserting for language {lang}")
        with open(os.path.join(dataset_base, f".{lang}"), "r") as f:
            dataset = f.readlines()
        with open(os.path.join(DATASET_DIR, "canary", f"-{lang}.json"), "r") as f:
            canary = json.load(f)["prefix"] + " " + json.load(f)["suffix"]
        for i in enumerate(indices):
            dataset.insert((int)(i*len(dataset)), canary)
        with open(os.path.join(dataset_base + f"-canary.{lang}")) as f:
            f.writelines(dataset)
    logger.info("===== Done =====")
if __name__ == "__main__":
    main()