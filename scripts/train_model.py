import logging
from IPython.display import display
import os
import json
import argparse
from transformers import AutoTokenizer
from experiment_lib import *
from happytransformer import HappyGeneration, GENTrainArgs
import torch
import numpy as np

# Configure Python's logging in Jupyter notebook
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

# Set up logger
logger = logging.getLogger()
handler = JupyterHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("Parsing arguments...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument(
    "--config_file", type=str, required=True, help="Path to the configuration file"
)
args = parser.parse_args()

with open(args.config_file, "r") as f:
    config = json.load(f)

# Set default device
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

logger.info(f"Default device: {DEFAULT_DEVICE}")

# For saving results
ROOT_DIR = config["root_dir"]
# Name of the dataset
DATASET_DIR = config["dataset_dir"]
# Directory where the .npy files of the dataset are stored
SOURCE_DIR = config["source_dir"]
# Name of the experiment
EXPERIMENT_NAME = config["experiment_name"]
# Name of the dataset
DATASET_FILE = config["dataset_file"]
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
# Name of the model to use
model = config["model"]

# Load model and tokenizer
try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    logger.info("Loading model...")
    MODEL = HappyGeneration("GPT-NEO", model)
    # model is moved to GPU by default
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

# logger.info("Experiment name: %s", config["experiment_name"])
logger.info("Language: %s", config["language"])
logger.info("Model: %s", config["model"])

def main():
    # default epochs is 3
    args = GENTrainArgs(num_train_epochs=1)

    training_base = os.path.join(DATASET_DIR, DATASET_FILE + "-" + str(EXAMPLE_TOKEN_LEN))

    # load the npy tokenized version of the training dataset
    training_file = training_base + "." + LANGUAGE +".txt"
    # train the model
    logger.info("Training model on dataset %s", DATASET_FILE)
    try:
        MODEL.train(training_file, args=args)
        logger.info("Model trained successfully.")
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

    output_dir = os.path.join("models", DATASET_DIR, LANGUAGE, EXPERIMENT_NAME)
    os.makedirs(output_dir, exist_ok=True)
    # save the model
    logger.info("Saving model...")
    MODEL.save(output_dir)

if __name__ == "__main__":
    main()