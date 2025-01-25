import logging
from IPython.display import display
import os
import argparse
import json
import random
from util_lib import *
from torch.utils.data import random_split
from transformers import set_seed
import torch

# Configure Python's logging in Jupyter notebook
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set up logger
logger = initLogger()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process config input.")
parser.add_argument(
    "--config_file", type=str, required=True, help="Path to the configuration file"
)
parser.add_argument(
    "--canaries_train", action="store_true", help="If this flag is used, all instances of the canary will end up in the training set"
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

set_seed(SEED)

# This script splits the full dataset into a training and validation set
# Performed for both langauges in the experiment to maintain a balanced version of the training and validation sets
languages = ["en", "nl"]

def main():
    logger.info("==== Starting data train+val split script ====")

    # Path where the split and validation datasets will be stored (same directory as the original dataset)
    dir = get_data_directory(DATASET_DIR, PREPROCESSING, NORMALIZATION, EXAMPLE_TOKEN_LEN)
    # load the dataset
    data_set_base = os.path.join(dir, DATASET_NAME)
    eval_percentage = VAL_SPLIT

    # Step 1: split on indices and save them to a file
    indices_file = os.path.join(data_set_base + "-split_indices.json")

    # take the size of the first language as the size of the dataset
    # this can be the second one as well, the input datasets are already aligned and identical
    dataset_path = os.path.join(data_set_base + f".{languages[0]}")
    with open(dataset_path, "r") as f:
        dataset = f.readlines()

    # Calculate the size of the training and validation sets
    train_size = int(len(dataset) * (1- eval_percentage))
    eval_size = len(dataset) - train_size

    # Create a range of indices to map to exids later
    indices = list(range(len(dataset)))

    # Split the indices along with the dataset
    logger.info("Splitting indices...")
    train_indices, eval_indices = random_split(indices, [train_size, eval_size])

    # If this flag is used, ensure that all instances of the canary are in the training set.
    if args.canaries_train:
        logger.info("Inserting canary indices into training indices")
        with open(os.path.join(DATASET_DIR, "canary" + f"-{lang}.json"), "r") as f:
            canary_file = json.load(f)
        canary = canary_file["prefix"] + " " + canary_file["suffix"]
        canary_indices = [i for i, line in enumerate(dataset.splitlines()) if canary in line]
        for index in canary_indices:
            if not index in train_indices:
                # if the index is not in the training set, insert it at a random index
                train_indices.insert(random.randint(0, len(train_indices)), index)

    # Convert Subset objects to lists
    train_indices = train_indices.indices
    eval_indices = eval_indices.indices

    print("# of indices: ", len(train_indices)+ len(eval_indices))
    
    # Save indices to file in JSON format
    with open(indices_file, "w") as f:
        json.dump({"train": train_indices, "eval": eval_indices}, f)


    # Step 2: split the datasets using the indices and save them to files
    logger.info("Splitting datasets into train and validation sets...")
    for lang in languages:
        logger.info(f"Processing language: {lang}")
        train_out_file = os.path.join(dir, DATASET_NAME + "-train-" + lang + ".txt")
        val_out_file = os.path.join(dir, DATASET_NAME + "-validation-" + lang + ".txt")

        # Check if the files already exist
        if os.path.exists(train_out_file) and os.path.exists(val_out_file) and os.path.exists(indices_file):
            print("Files already exist. Skipping computation.")
            return

        dataset_path = os.path.join(data_set_base + f".{lang}") 
        with open(dataset_path, "r") as f:
            dataset = f.readlines()
        
        # Create the train and eval datasets using the indices
        train_dataset = [dataset[i] for i in train_indices]
        eval_dataset = [dataset[i] for i in eval_indices]

        # Save train and eval datasets to files
        with open(train_out_file, "w") as f:
            f.writelines(train_dataset)

        with open(val_out_file, "w") as f:
            f.writelines(eval_dataset)

    logger.info("==== Data train+val split script completed ====")

if __name__ == "__main__":
    main()