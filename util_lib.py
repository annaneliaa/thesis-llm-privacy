# In this file we have utility functions for imports and managing file paths that are used in all kinds of files

import os
import logging
from IPython.display import display
from transformers import AutoTokenizer

class JupyterHandler(logging.Handler):
    def emit(self, record):
        display(self.format(record))

# Set up a logger
def initLogger():
    logger = logging.getLogger()
    handler = JupyterHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def load_constants_from_config(config):
    # For saving results
    ROOT_DIR = config["root_dir"]
    # Name of the dataset
    DATASET_DIR = config["dataset_dir"]
    # Directory where the .npy files of the dataset are stored
    SOURCE_DIR = config["source_dir"]
    # Name of the dataset
    DATASET_NAME = config["dataset_name"]
    # Name of the experiment
    EXPERIMENT_NAME = config["experiment_name"]
    # Boolean indicating whether preprocessing runs / has been run
    PREPROCESSING = config["preprocessing"]
    # Standard suffix used in file name
    PREPROCESSING_SUFFIX = "-pre"
    # Boolean indicating whether the sentence length has been normalized to EXAMPLE_TOKEN_LEN
    NORMALIZATION = config["normalization"]
    # Boolean indicating whether the tokenization is done in batches. If normalization is true, this is rather irrelevant.
    BATCHING = config["batching"]
    # Number of trials
    NUM_TRIALS = config["num_trials"]
    # Language of the scenario (EN/NL)
    LANGUAGE = config["language"]
    # Split the dataset into train and eval
    SPLIT = config["split"]
    # Length of the suffix
    SUFFIX_LEN = config["suffix_len"]
    # Length of the prefix
    PREFIX_LEN = config["prefix_len"]
    # Number of tokens in the complete sequences
    EXAMPLE_TOKEN_LEN = config["example_token_len"]
    # Preprefix length
    PREPREFIX_LEN = config["preprefix_len"]
    # Name of the tokenized .npy file of the dataset
    SOURCE_FILE = config["source_file"]
    # Batch size for feeding prompts to the model
    BATCH_SIZE = config["batch_size"]
    # Name of the model to use
    MODEL_NAME = config["model"]
    # The percentage of the dataset that will be used for validation.
    VAL_SPLIT = config["validation_split_percentage"]
    SEED = config["seed"]

    return (ROOT_DIR, DATASET_DIR, SOURCE_DIR, DATASET_NAME, EXPERIMENT_NAME, PREPROCESSING, PREPROCESSING_SUFFIX, NORMALIZATION, BATCHING, NUM_TRIALS, PREFIX_LEN, SUFFIX_LEN, PREPREFIX_LEN, LANGUAGE, SPLIT, EXAMPLE_TOKEN_LEN, SOURCE_FILE, BATCH_SIZE, MODEL_NAME, VAL_SPLIT, SEED)


def initTokenizer(model_name: str):
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise e
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})
    return tokenizer

# helper function
def get_path(preprocessing: bool, normalization: bool, example_token_len = 0):
    dir = ""
    if (preprocessing):
        dir = os.path.join(dir, str(example_token_len))
    else:
        dir = dir = os.path.join(dir, "raw")
    if (normalization):
        dir = os.path.join(dir, "normalized")
    else:
        dir = dir = os.path.join(dir, "natural")
    return dir

# gets the path to the correct repository that holds the initial datasets
def get_data_directory(dataset_dir, preprocessing: bool, normalization: bool, example_token_len = 0):
    dir = os.path.join(dataset_dir, get_path(preprocessing, normalization, example_token_len))
    os.makedirs(dir, exist_ok=True)
    return dir

# gets the path to the correct repository that holds the tokenized datasets
def get_source_directory(source_dir, dataset_dir, language, preprocessing: bool, normalization: bool, example_token_len = 0):
    dir = os.path.join(source_dir, dataset_dir, language, get_path(preprocessing, normalization, example_token_len))
    os.makedirs(dir, exist_ok=True)
    return dir

# gets the path to the correct repository that holds the results
def get_mia_result_dir(root_dir, dataset_dir):
    dir = os.path.join(root_dir, dataset_dir, "mia")
    os.makedirs(dir, exist_ok=True)
    return dir