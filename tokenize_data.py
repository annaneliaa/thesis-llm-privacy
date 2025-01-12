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

# Set up tokenizer
tokenizer = initTokenizer(MODEL_NAME)
pad_token_id = tokenizer.pad_token_id

def main():
    logger.info("===== Starting dataset token generation =====")
    dir = get_data_directory(DATASET_DIR, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
    # read the train indices
    with open(os.path.join(dir, "split_indices.json"), "r") as f:
        train_indices = json.load(f)["train"]
    
    logger.info(f"Processing language: {LANGUAGE}")
    # Read the training and validation data for the language
    with open(os.path.join(dir, "train-" + LANGUAGE + ".txt"), "r") as f:
        train_data = f.readlines()
    with open(os.path.join(dir, "validation-" + LANGUAGE + ".txt"), "r") as f:
        val_data = f.readlines()
    # Get the directory paths for the output
    source_dir = get_source_directory(SOURCE_DIR, DATASET_DIR, LANGUAGE, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
    train_out_file = os.path.join(source_dir, "train-" + LANGUAGE + ".pt")
    if not BATCHING:
        train_out_file = os.path.join(source_dir, "train-nb-" + LANGUAGE + ".pt")
    val_out_file = os.path.join(source_dir, "validation-" + LANGUAGE + ".pt")

    # Check if the files already exist
    if os.path.exists(train_out_file) and os.path.exists(val_out_file):
        print("Files already exist. Skipping computation.")
        return
    # Tokenize the datasets
    if BATCHING:
        logger.info("===== Tokenizing training data in batches =====")
        # Create mapping from ids to strings for training dataset
        train_dataset_map = {train_indices[i]: train_data[i] for i in range(len(train_indices))}
        tokenized_train_dataset = tokenize_prompts_in_batches(tokenizer, train_dataset_map)
    else:
        logger.info("===== Tokenizing training data without batches =====")
        # this call pads to the longest sequence in the dataset, and truncates to max_length (at most)
        tokenized_train_dataset = tokenizer(train_data, max_length=512, padding=True, truncation=True, return_tensors="pt")
        tokenized_train_dataset["sentence_ids"] = train_indices
    tokenized_eval_sentences = tokenizer(val_data, max_length=512, padding=True, truncation=True, return_tensors="pt")

    # Save the tokenized train and eval datasets to files
    torch.save(tokenized_train_dataset, train_out_file)
    torch.save(tokenized_eval_sentences, val_out_file)
    logger.info("===== Tokenization done! =====")

if __name__ == "__main__":
    main()