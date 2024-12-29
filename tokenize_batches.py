import os
import torch
import numpy as np
import json
import argparse
from transformers import AutoTokenizer
import logging
from IPython.display import display
from util_lib import *
from data_lib import tokenize_prompts_in_batches

# Configure Python's logging in Jupyter notebook
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s")

# set up logger
logger = initLogger()

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
    PREPROCESSING,
    PREPROCESSING_SUFFIX,
    NORMALIZATION,
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

languages = ["en","nl"]

# Set up tokenizer
tokenizer = initTokenizer()
pad_token_id = tokenizer.pad_token_id

def main():
    logger.info(
        "===== Starting dataset token generation in batches for language %s =====",
        LANGUAGE,
    )
    dir = get_data_directory(DATASET_DIR, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
    # read the train indices
    with open(os.path.join(dir, "split_indices.json"), "w") as f:
        train_indices = json.load(f)["train"]
    
    logger.info("Splitting datasets into train and validation sets...")
    for lang in languages:
        logger.info(f"Processing language: {lang}")
        # Read the training and validation data for the language
        with open(os.path.join(dir, "train-" + lang + ".txt"), "w") as f:
            train_data = f.readlines()
        with open(os.path.join(dir, "validation-" + lang + ".txt"), "w") as f:
            val_data = f.readlines()
        # Get the directory paths for the output
        source_dir = get_source_directory(SOURCE_DIR, DATASET_DIR, lang, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
        train_out_file = os.path.join(source_dir, "train-" + lang + ".pt")
        val_out_file = os.path.join(source_dir, "validation-" + lang + ".pt")

        # Check if the files already exist
        if os.path.exists(train_out_file) and os.path.exists(val_out_file):
            print("Files already exist. Skipping computation.")
            return
        # Create the train and eval datasets using the indices
        train_dataset_map = {train_indices[i]: train_data[i] for i in range(len(train_indices))}
        # Tokenize the datasets
        tokenized_train_dataset = tokenize_prompts_in_batches(tokenizer, train_dataset_map)
        tokenized_eval_sentences = tokenizer(val_data, max_length=512, padding=True, truncation=True, return_tensors="pt")

        # Save train and eval datasets to files
        torch.save(tokenized_train_dataset, train_out_file)
        torch.save(tokenized_eval_sentences, val_out_file)

if __name__ == "__main__":
    main()